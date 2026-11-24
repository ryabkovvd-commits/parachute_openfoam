"""Isolate canopy geometry using matched, impermeable, frozen CFD calculations."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from aero_diagnostics import input_shape, make_mesh
from parafoil.inflation.constraints import MeanAxisConstraint

NAME = 'nos6_held'
PILOT = [0, 30, 40, 50]
FULL = list(range(0, 51, 5))
NUMERICS = ['system/fvSchemes', 'system/fvSolution',
            'constant/transportProperties', 'constant/turbulenceProperties']


def code_fingerprint():
    root = Path(__file__).resolve().parent
    return {p.relative_to(root).as_posix(): sha(p) for p in sorted(root.rglob('*'))
            if p.is_file() and p.suffix in ('.py', '.txt') and '__pycache__' not in p.parts}


def save(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_case(work):
    work = Path(work).resolve()
    history = work/'out'/f'{NAME}.json'
    rows = json.loads(history.read_text(encoding='utf-8'))
    if not rows:
        raise ValueError(f'Empty source history: {history}')
    row = max(rows, key=lambda r: int(r['pass']))
    row = dict(row)
    inferred = {}
    if 'D0' not in row:
        diameter = float(np.sqrt(24/np.pi))
        if (not np.isclose(float(row['lines']),diameter)
                or any(int(row[k])!=v for k,v in [('gores',96),('rings',12),('stride',4)])):
            raise ValueError('Old source has no D0 and does not match the declared 6 m2, l/D0=1 setup')
        row['D0'] = diameter
        inferred['D0'] = 'declared S0=6 m2; checked nominal line length D0 and 96x12/24-cord topology'
    path = input_shape(work, NAME, row)
    positions = np.load(path, allow_pickle=False)
    mesh = make_mesh(row, float(row['D0']))
    if positions.shape != mesh.position.shape or not np.isfinite(positions).all():
        raise ValueError(f'Invalid source geometry: {path}')
    frame = row.get('coordinate_frame', 'native')
    if frame == 'nosarev_speed':
        from nosarev_frame import to_native
        positions = to_native(positions)
    elif frame not in ('native', None):
        raise ValueError(f'Unknown coordinate frame: {frame}')
    if np.linalg.norm(positions[mesh.confluence[0]]) > 1e-8:
        raise ValueError('This study requires the confluence at the origin')
    mesh.position = positions.copy()
    return row, mesh, dict(source_case=str(work), source_pass=int(row['pass']),
                           source_history_sha256=sha(history), source_shape=str(path),
                           source_shape_sha256=sha(path), source_coordinate_frame=frame,
                           inferred_parameters=inferred)


def plan(angles):
    angles = sorted(set(map(int, angles)))
    if not angles or angles[0] != 0 or any(a < 0 or a > 50 for a in angles):
        raise ValueError('Include 0 degrees and use angles in 0..50')
    result = []
    for angle in angles:
        for family in (['reference'] if angle == 0 else ['reference', 'angle_shape']):
            result.append(dict(id=f'{family}_a{angle:02d}', angle_deg=angle,
                               family=family, source_angle_deg=0 if family=='reference' else angle))
    return result


def prepare(source_root, output, angles, iterations=2000, mesh_factor=1.25):
    source_root, output = Path(source_root).resolve(), Path(output).resolve()
    tasks = plan(angles)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    if iterations < 200 or iterations % 200 or not np.isfinite(mesh_factor) or mesh_factor <= 0:
        raise ValueError('Use iterations >=200, multiple of 200, and positive finite mesh factor')
    sources = {}
    for angle in sorted({t['source_angle_deg'] for t in tasks}):
        row, mesh, provenance = load_case(source_root/f'a{angle:02d}')
        if not np.isclose(float(row['hold']), angle) or not np.isclose(float(row['D0']), np.sqrt(24/np.pi)):
            raise ValueError('Source angle or scale differs from the 6 m2 study')
        if (any(int(row[k]) != v for k,v in [('gores',96),('rings',12),('stride',4)])
                or not np.isclose(float(row['lines']), float(row['D0']))):
            raise ValueError('Use the 96x12 geometry with 24 cords of length D0')
        if row.get('canopy_model') not in (None, 'wall') or row.get('fabric_coefficients'):
            raise ValueError('Use source geometries from the impermeable series')
        if row.get('canopy_model') is None:
            provenance['wall_provenance'] = 'declared source series; old record has no canopy_model field'
        else:
            provenance['wall_provenance'] = 'source record canopy_model=wall'
        sources[angle] = (row, mesh, provenance)
    base = sources[0][0]
    for row, _, _ in sources.values():
        for key in ['D0', 'lines', 'gores', 'rings', 'stride', 'shear']:
            if not np.isclose(float(row.get(key, .02 if key=='shear' else np.nan)),
                              float(base.get(key, .02 if key=='shear' else np.nan))):
                raise ValueError(f'Source geometry definitions differ: {key}')
    output.mkdir(parents=True)
    for task in tasks:
        row, mesh, provenance = sources[task['source_angle_deg']]
        # Only a rigid alignment: no loads, structural solve, or shape relaxation.
        frozen = MeanAxisConstraint(mesh, task['angle_deg']).align_initial(mesh.position)
        target = output/(task['id']+'.npy')
        np.save(target, frozen)
        task.update(provenance, seed_file=target.name, seed_sha256=sha(target),
                    geometry_alignment='rigid rotation of mean canopy axis about confluence',
                    topology={k: base.get(k, .02 if k=='shear' else None)
                              for k in ['D0', 'lines', 'gores', 'rings', 'stride', 'shear']},
                    source_coefficients={k: row[k] for k in ['cx', 'cy', 'mz']},
                    source_convergence='accepted under user-specified coupling limits')
    manifest = dict(study='impermeable frozen geometry', source_root=str(source_root),
                    source_series=source_root.name, tasks=tasks, iterations=iterations,
                    code_sha256=code_fingerprint(),
                    mesh_factor=mesh_factor, V=30., nu=1.46e-5, rho=1.225,
                    cord_mm=3.2, cord_cd=1.2, pressure_gradient='gauss',
                    turbulence_length=1.24, turbulence_intensity=.01,
                    coordinate_frame='native',
                    comparison='angle_shape minus reference at equal angle and common new CFD settings',
                    caveat='Do not subtract old coupled coefficients to infer pure deformation: source numerics may differ.')
    save(output/'manifest.json', manifest)
    print(f"Prepared {len(tasks)} fixed CFD states: {output/'manifest.json'}")
    return manifest


def command(manifest_path, task_index):
    path = Path(manifest_path).resolve()
    data = json.loads(path.read_text(encoding='utf-8'))
    if data.get('code_sha256') != code_fingerprint():
        raise ValueError('CFD code changed after setup; restore the recorded version or prepare a new study')
    if not 0 <= task_index < len(data['tasks']):
        raise ValueError('Task index outside manifest')
    task = data['tasks'][task_index]
    seed = path.parent/task['seed_file']
    if not seed.resolve().is_relative_to(path.parent) or sha(seed) != task['seed_sha256']:
        raise ValueError('Prepared geometry changed or seed is outside setup')
    topo = task['topology']
    options = ['--patch=nos6,cords', '--fixed-shape', '--attitude=legacy',
               f"--угол={task['angle_deg']}", f"--полотнищ={int(topo['gores'])}",
               f"--шаг={int(topo['stride'])}", f"--сдвиг={topo['shear']}",
               f"--V={data['V']}", f"--nu={data['nu']}",
               f"--cord-mm={data['cord_mm']}", f"--cord-cd={data['cord_cd']}",
               f"--mesh={data['mesh_factor']}", '--pressure-gradient=gauss',
               f"--turbulence-length={data['turbulence_length']}",
               f"--turbulence-intensity={data['turbulence_intensity']}"]
    return [sys.executable, str(Path(__file__).with_name('cs.py')), NAME, '0', '0',
            str(data['iterations']), '1', str(seed), *options], data, task


def verify(work):
    work = Path(work).resolve()
    info = json.loads((work/'study_case.json').read_text(encoding='utf-8'))
    _, data, task = command(info['manifest'], info['task_index'])
    row = json.loads((work/'out'/f'{NAME}.json').read_text(encoding='utf-8'))[-1]
    if not row.get('fixed_shape') or row.get('canopy_model') != 'wall':
        raise ValueError('Study result must be a frozen impermeable calculation')
    if row.get('coordinate_frame') not in (None, 'native'):
        raise ValueError('Result coordinate frame changed')
    if not np.isclose(float(row['hold']), task['angle_deg']) or row['iters'] != data['iterations']:
        raise ValueError('Result angle or iteration budget differs from setup')
    for relative in [f'fields/{NAME}/pass01/input_shape.npy', f'out/{NAME}_shape.npy']:
        if sha(work/relative) != task['seed_sha256']:
            raise ValueError(f'Frozen coordinates changed: {relative}')
    if row.get('fabric_coefficients') or row.get('pressure_gradient') != 'gauss':
        raise ValueError('CFD boundary law or pressure scheme changed')
    for field, expected in [('V',data['V']), ('nu',data['nu']), ('rho',data['rho']),
                            ('D0',task['topology']['D0']), ('S0',6.),
                            ('lines',task['topology']['lines']),
                            ('cord_width_m',data['cord_mm']/1000), ('cord_cd',data['cord_cd']),
                            ('turbulence_length_m',data['turbulence_length']),
                            ('turbulence_intensity',data['turbulence_intensity'])]:
        if not np.isclose(float(row[field]), expected):
            raise ValueError(f'Unexpected physical setting: {field}')
    numerical_sha = {name: sha(work/'case'/name) for name in NUMERICS}
    return data, task, row, numerical_sha


def run(manifest_path, index, root):
    cmd, _, task = command(manifest_path, index)
    work = Path(root).resolve()/task['id']
    if work.exists():
        raise FileExistsError(f'Refusing to overwrite {work}')
    work.mkdir(parents=True)
    save(work/'study_case.json', dict(manifest=str(Path(manifest_path).resolve()), task_index=index, command=cmd))
    done = subprocess.run(cmd, cwd=work)
    status = dict(CFD_returncode=done.returncode, verification='not performed',postprocess_returncode=None)
    save(work/'study_run_status.json',status)
    if done.returncode:
        return done.returncode
    verify(work)
    status['verification']='passed'
    post = subprocess.run([sys.executable, str(Path(__file__).with_name('nos6_results.py')),
                           'case', str(work), '--speed=30', '--wake=0', '--gif=off', '--no-summary'])
    status['postprocess_returncode']=post.returncode
    save(work/'study_run_status.json',status)
    return post.returncode


def write_table(path, rows, fields):
    with Path(path).open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def plot_comparison(root, report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    angles = report['angles_deg']
    by_id = {r['id']: r for r in report['cases']}
    differences = {r['angle_deg']: r for r in report['delta_angle_shape_minus_reference']}
    with plt.rc_context({'font.family':'DejaVu Sans', 'svg.fonttype':'none'}):
        fig, axes = plt.subplots(1,3,figsize=(11.2,3.7),constrained_layout=True)
        for key, ax in zip(['cx','cy','mz'],axes):
            for family,style,color,label in [
                ('reference','o--','#0072B2','Форма при 0°'),
                ('angle_shape','s-','#D55E00','Форма при данном угле')]:
                ids = [f'{family}_a{a:02d}' if a or family=='reference' else 'reference_a00' for a in angles]
                # Explicit NaNs keep missing cases as gaps, never as zero or interpolation.
                values = [by_id.get(name,{}).get(key,np.nan) for name in ids]
                ax.plot(angles,values,style,color=color,ms=4,label=label)
            ax.set(xlabel='Угол атаки α, град',ylabel=key)
            ax.axhline(0,color='.4',lw=.6);ax.grid(alpha=.25)
        axes[0].legend(fontsize=8)
        fig.suptitle(report.get('plot_label','Непроницаемый купол: фиксированные формы')+'; '+
                     ('серия собрана' if report['complete'] else 'НЕПОЛНАЯ СЕРИЯ'))
        for suffix in ['png','svg']: fig.savefig(root/f'fixed_shape_polar.{suffix}',dpi=180)
        plt.close(fig)
        fig, axes = plt.subplots(1,3,figsize=(11.2,3.4),constrained_layout=True)
        for key,ax in zip(['cx','cy','mz'],axes):
            ax.plot(angles,[differences.get(a,{}).get(key,np.nan) for a in angles],'o-',color='#0072B2',ms=4)
            ax.axhline(0,color='.4',lw=.7);ax.grid(alpha=.25)
            ax.set(xlabel='Угол атаки α, град',ylabel='Δ'+key)
        fig.suptitle(report.get('plot_label','')+'\nФорма при данном угле минус повёрнутая форма при 0°')
        for suffix in ['png','svg']: fig.savefig(root/f'geometry_effect.{suffix}',dpi=180)
        plt.close(fig)
    save(root/'figure_provenance.json',dict(source='geometry_comparison.json',
         formulas='angle_shape minus reference at the same angle; zero case shared',
         interpolation='none; segments connect available adjacent sampled angles; missing values are gaps',
         uncertainty='No statistical intervals or numerical uncertainty estimates added',
         coordinate_frame='native',source_series=report['source_series'],
         complete=report['complete'],matplotlib_version=matplotlib.__version__))


def collect(root, manifest_path=None):
    root = Path(root).resolve()
    root.mkdir(parents=True,exist_ok=True)
    if manifest_path is None:
        first = next(iter(sorted(root.glob('*/study_case.json'))),None)
        if first is None:
            raise ValueError('No case metadata; pass --manifest to report a completely failed series')
        manifest_path = json.loads(first.read_text(encoding='utf-8'))['manifest']
    expected = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    angles = sorted({t['angle_deg'] for t in expected['tasks']})
    signature_expected = hashlib.sha256(json.dumps(expected,sort_keys=True).encode('utf-8')).hexdigest()
    failures = []
    result, coefficients, shared = [], {}, None
    expected_ids = {t['id'] for t in expected['tasks']}
    for task_expected in expected['tasks']:
        work = root/task_expected['id']
        if not (work/'study_case.json').is_file():
            continue
        try:
            data, task, row, numeric = verify(work)
            signature = hashlib.sha256(json.dumps(data,sort_keys=True).encode('utf-8')).hexdigest()
            if signature != signature_expected or task['id'] != task_expected['id']:
                raise ValueError('Case came from another study/task')
            if shared is not None and numeric != shared:
                raise ValueError('CFD discretization/solver/material settings differ across cases')
        except (OSError,ValueError,KeyError) as exc:
            failures.append(dict(id=task_expected['id'],error=str(exc)))
            continue
        shared = numeric
        values = {k: float(row[k]) for k in ['cx', 'cy', 'mz']}
        coefficients[task['id']] = values
        run_status = work/'study_run_status.json'
        postprocess = json.loads(run_status.read_text(encoding='utf-8')).get('postprocess_returncode') if run_status.exists() else None
        result.append(dict(id=task['id'], angle_deg=task['angle_deg'], family=task['family'],
                           **values, cells=row['cells'],
                           postprocess_returncode=postprocess,
                           force_window_range=row.get('force_last_window_range'),
                           moment_window_range=row.get('moment_last_window_range'),
                           CFD_convergence=row.get('CFD_convergence', 'not recorded')))
    differences = []
    for angle in sorted({r['angle_deg'] for r in result}):
        ref = coefficients.get(f'reference_a{angle:02d}')
        own = coefficients.get(f'angle_shape_a{angle:02d}') if angle else ref
        if ref is not None and own is not None:
            differences.append(dict(angle_deg=angle, **{k: own[k]-ref[k] for k in ref}))
    missing = sorted(expected_ids-set(coefficients))
    report = dict(cases=result, delta_angle_shape_minus_reference=differences,
         missing_cases=missing, failed_cases=failures, expected_count=len(expected_ids),
         completed_count=len(result),complete=not missing and not failures,angles_deg=angles,
         source_series=expected['source_series'],common_numerical_sha256=shared,
         interpretation='Geometry effect under common fixed CFD settings; no new structural equilibrium is computed.')
    save(root/'geometry_comparison.json',report)
    write_table(root/'fixed_shape_cases.csv',result,['id','angle_deg','family','cx','cy','mz','cells','CFD_convergence','postprocess_returncode'])
    write_table(root/'geometry_effect.csv',differences,['angle_deg','cx','cy','mz'])
    plot_comparison(root,report)
    (root/'summary.md').write_text(
        '# Исследование фиксированной формы\n\n'
        f"Принято {len(result)} из {len(expected_ids)} состояний. "
        f"Полнота серии: {'полная' if report['complete'] else 'НЕПОЛНАЯ'}.\n\n"
        'Оси: прежние глобальные, поток +Z; Cx по +Z, Cy по +X, момент по +Y около коуша.\n'
        'Разность: angle_shape − reference при одном угле.\n'
        'Флаг полной серии означает наличие и согласованность результатов; '
        'истории CFD и применённые критерии установления оцениваются отдельно.\n\n'
        'Таблицы: fixed_shape_cases.csv, geometry_effect.csv. '
        'Графики: fixed_shape_polar.png/svg, geometry_effect.png/svg.\n',encoding='utf-8')
    print(root/'geometry_comparison.json')
    print(f"Accepted {len(result)}/{len(expected_ids)}; missing: {', '.join(missing) or 'none'}")
    return report


def compare_holding(legacy, inner, output):
    a, ma, pa = load_case(legacy)
    b, mb, pb = load_case(inner)
    if a.get('fixed_shape') or b.get('fixed_shape'):
        raise ValueError('A fixed CFD run cannot measure a change of structural angle holding')
    for key in ['hold','D0','lines','gores','rings','stride','V','nu','rho','shear',
                'canopy_model','fabric_coefficients','cord_width_m','cord_cd',
                'pressure_gradient','turbulence_length_m','turbulence_intensity']:
        if a.get(key) != b.get(key):
            raise ValueError(f'Unmatched holding comparison: {key}')
    if a.get('attitude_mode') not in (None,'legacy') or b.get('attitude_mode') != 'inner':
        raise ValueError('Expected legacy and inner coupled cases')
    differences, percents, left_native, right_native = {}, {}, {}, {}
    for key in ['cx','cy','mz']:
        # Report signed Cy in a single native coordinate frame.
        x = float(a[key]) * (-1 if key=='cy' and pa['source_coordinate_frame']=='nosarev_speed' else 1)
        y = float(b[key]) * (-1 if key=='cy' and pb['source_coordinate_frame']=='nosarev_speed' else 1)
        left_native[key], right_native[key] = x,y
        differences[key] = y-x
        percents[key] = 100*(y-x)/abs(x) if abs(x)>1e-8 else None
    xa = MeanAxisConstraint(ma, float(a['hold'])).align_initial(ma.position)
    xb = MeanAxisConstraint(mb, float(b['hold'])).align_initial(mb.position)
    distances = np.linalg.norm(xb-xa,axis=1)
    signatures = []
    for folder in (legacy, inner):
        signature = Path(folder)/'out'/f'{NAME}_config.json'
        signatures.append(json.loads(signature.read_text(encoding='utf-8')) if signature.exists() else None)
    versions_match = all(s is not None and bool(s.get('source_sha256')) for s in signatures) and signatures[0].get('source_sha256') == signatures[1].get('source_sha256')
    flag_differences = None
    if all(s is not None for s in signatures):
        left,right = [dict(s.get('physics',{}).get('flags',{})) for s in signatures]
        for flags in (left,right):
            flags.pop('--attitude',None)
        flag_differences = {k: [left.get(k),right.get(k)] for k in set(left)|set(right) if left.get(k)!=right.get(k)}
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    save(output, dict(legacy=pa, inner=pb, coefficient_frame='native', coefficients_legacy=left_native,
                     coefficients_inner=right_native, native_delta=differences,
                     percent_of_abs_legacy=percents,
                     source_versions_match=versions_match, other_flag_differences=flag_differences,
                     controlled_holding_comparison=bool(versions_match and flag_differences=={}),
                     shape_rms_D0=float(np.sqrt(np.mean(distances**2))/a['D0']),
                     shape_max_D0=float(distances.max()/a['D0']),
                     convergence='accepted under user-specified limits',
                     caveat='Final-state comparison; source-version and material identity must also match for causal attribution.'))
    print(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    p = sub.add_parser('plan'); p.add_argument('--pilot',action='store_true'); p.add_argument('--full',action='store_true')
    p = sub.add_parser('prepare'); p.add_argument('source',type=Path); p.add_argument('output',type=Path)
    p.add_argument('--pilot',action='store_true'); p.add_argument('--full',action='store_true'); p.add_argument('--iterations',type=int,default=2000)
    p.add_argument('--mesh-factor',type=float,default=1.25)
    p = sub.add_parser('run'); p.add_argument('manifest',type=Path); p.add_argument('index',type=int); p.add_argument('root',type=Path)
    p = sub.add_parser('collect'); p.add_argument('root',type=Path);p.add_argument('--manifest',type=Path)
    p.add_argument('--allow-incomplete',action='store_true')
    p = sub.add_parser('compare-holding'); p.add_argument('legacy',type=Path); p.add_argument('inner',type=Path); p.add_argument('output',type=Path)
    args = parser.parse_args()
    if args.mode in ('plan','prepare') and args.pilot and args.full:
        parser.error('Choose --pilot or --full, not both')
    if args.mode == 'plan': print(json.dumps(plan(PILOT if args.pilot else FULL),indent=2))
    elif args.mode == 'prepare': prepare(args.source,args.output,PILOT if args.pilot else FULL,args.iterations,args.mesh_factor)
    elif args.mode == 'run': return run(args.manifest,args.index,args.root)
    elif args.mode == 'collect':
        report=collect(args.root,args.manifest)
        return 0 if report['complete'] or args.allow_incomplete else 1
    else: compare_holding(args.legacy,args.inner,args.output)
    return 0


if __name__ == '__main__':
    sys.exit(main())
