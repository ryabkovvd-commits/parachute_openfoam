"""Continue the existing 21 frozen cases to iteration 2000, preserving provenance."""
import argparse
import contextlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE/'coupled'))
import fixed_shape_study as fs
from aero_diagnostics import make_mesh
from cords import read_cord_load
from run_guard import atomic_json, check_cfd, mark_status, table
import vtp

TARGET = 2000
NAME = fs.NAME


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def prepare(root, setup):
    root, setup = Path(root).resolve(), Path(setup).resolve()
    first = next(iter(sorted(root.glob('*/study_case.json'))), None)
    if first is None:
        raise ValueError('No existing study_case.json; use the normal full-series submission')
    source = Path(read(first)['manifest'])
    data = read(source)
    expected = fs.plan(fs.FULL)
    if [{k: t[k] for k in expected[0]} for t in data['tasks']] != expected:
        raise ValueError('Expected the complete 21-state study')
    if int(data['iterations']) > TARGET:
        raise ValueError('Refusing to shorten a study beyond 2000')
    # Checks all seeds and the original solver fingerprint before creating anything.
    for index in range(len(expected)):
        fs.command(source, index)
    setup.mkdir(parents=True, exist_ok=False)
    for task in data['tasks']:
        shutil.copyfile(source.parent/task['seed_file'], setup/task['seed_file'])
    data.update(iterations=TARGET, extension=dict(
        original_manifest=str(source), original_manifest_sha256=fs.sha(source),
        original_iterations=read(source)['iterations'], result_root=str(root),
        tool_sha256=fs.sha(__file__), CFD_convergence='unconfirmed'))
    fs.save(setup/'manifest.json', data)
    print(setup/'manifest.json')


def control_text(text):
    """Change top-level controls, even when several entries share one line."""
    replacements = dict(application='simpleFoam', startFrom='latestTime',
                        stopAt='endTime', endTime=str(TARGET), deltaT='1',
                        writeControl='timeStep', writeInterval='250')
    # Only edit before functions: function objects have their own write intervals.
    parts = re.split(r'(?m)^[ \t]*functions\b', text, maxsplit=1)
    prefix = parts[0]
    for key, value in replacements.items():
        pattern = rf'\b{key}\s+([^;]+);'
        hits = list(re.finditer(pattern, prefix))
        if len(hits) != 1:
            raise ValueError(f'Expected one top-level {key} entry')
        old = hits[0].group(1).strip()
        if key == 'application' and old != 'simpleFoam':
            raise ValueError('Only stationary simpleFoam is supported')
        if key == 'deltaT' and float(old) != 1:
            raise ValueError('Expected deltaT 1 (SIMPLE iteration counter)')
        prefix = re.sub(pattern, f'{key} {value};', prefix)
    return prefix + ('functions'+parts[1] if len(parts) == 2 else '')


def latest(case):
    times = []
    for path in case.iterdir():
        if not path.is_dir():
            continue
        try:
            value = float(path.name)
        except ValueError:
            continue
        if np.isfinite(value):
            times.append((value, path))
    if not times:
        raise ValueError('No saved serial fields')
    value, path = max(times, key=lambda item: item[0])
    if value <= 0 or value > TARGET or value != int(value):
        raise ValueError(f'Unexpected saved iteration: {value}')
    for name in ['U', 'p', 'phi', 'nut', 'k', 'omega']:
        if not any((path/(name+s)).is_file() and (path/(name+s)).stat().st_size
                   for s in ['', '.gz']):
            raise ValueError(f'Incomplete latest restart fields: {path/name}')
    return int(value)


def merge_histories(case):
    """OpenFOAM starts a separate forces/<restartTime> directory on restart."""
    root = case/'postProcessing/forces'
    folders = sorted((p for p in root.iterdir() if p.is_dir() and
                      re.fullmatch(r'\d+(?:\.\d+)?', p.name)), key=lambda p: float(p.name))
    merged = {}
    for filename in ['force.dat', 'moment.dat']:
        rows = {}
        for folder in folders:
            path = folder/filename
            if path.is_file():
                for row in table(path):
                    rows[float(row[0])] = row
        if not rows or max(rows) != TARGET:
            raise ValueError(f'No {filename} sample at iteration {TARGET}')
        merged[filename] = np.array([rows[t] for t in sorted(rows)])
    if not np.array_equal(merged['force.dat'][:, 0], merged['moment.dat'][:, 0]):
        raise ValueError('Force and moment sampling times differ')
    for filename, values in merged.items():
        path = root/'0'/filename
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(path, values, header='Merged SIMPLE iteration history; originals in resume_2000_backup and restart folders')


def refreshed_row(work, old, start, elapsed):
    case = work/'case'
    archive = work/'fields'/NAME/'pass01'
    shutil.copytree(case/str(TARGET), archive/str(TARGET), dirs_exist_ok=True)
    shutil.copytree(case/'constant/polyMesh', archive/'polyMesh', dirs_exist_ok=True)
    shutil.copytree(case/'postProcessing', archive/'postProcessing', dirs_exist_ok=True)
    row = dict(old)
    q = .5*float(row['rho'])*float(row['V'])**2
    validation = check_cfd(case, archive, str(TARGET), q, row['S0'], row['D0'])
    mesh = make_mesh(row, float(row['D0']))
    mesh.position = np.load(archive/'input_shape.npy', allow_pickle=False)
    folder = archive/'postProcessing/surfacePressure'/str(TARGET)
    load, total, dcp = vtp.nodal_load(mesh, folder, q=q, kin=.5*float(row['V'])**2)
    cords = read_cord_load(mesh, folder, float(row['rho']))
    load += cords
    total += cords.sum(axis=0)
    r = mesh.position-mesh.position[mesh.confluence[0]]
    scale = q*row['S0']
    # Same native axes and pressure-to-node transfer as the original fixed driver.
    row.update(iters=TARGET, drag=float(total[2]), side=float(total[0]),
               moment=float(np.cross(r, load).sum(axis=0)[1]), dcp=float(dcp.mean()),
               cord_cx=float(cords.sum(axis=0)[2]/scale),
               cord_cy=float(cords.sum(axis=0)[0]/scale),
               cord_mz=float(np.cross(r, cords).sum(axis=0)[1]/(scale*row['D0'])),
               cord_anchor_force_N=cords[mesh.confluence].sum(axis=0).tolist(),
               minutes=float(old.get('minutes', 0))+elapsed/60,
               CFD_restart_iteration=start, CFD_additional_iterations=TARGET-start,
               settled=None, **validation)
    row.update(cx=row['drag']/scale, cy=row['side']/scale,
               mz=row['moment']/(scale*row['D0']))
    if not np.isfinite([row[k] for k in ['cx','cy','mz','dcp']]).all():
        raise ValueError('Non-finite refreshed loads')
    return row


def execute(command, case, filename):
    print('Running:', ' '.join(map(str, command)), flush=True)
    with (case/filename).open('w', encoding='utf-8') as stream:
        subprocess.run(command, cwd=case, stdout=stream, stderr=subprocess.STDOUT, check=True)


@contextlib.contextmanager
def locked(work):
    import fcntl  # Solver jobs run on the Linux cluster; released also on process death.
    with (work/'.resume_2000.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def run(manifest, index, root):
    manifest, root = Path(manifest).resolve(), Path(root).resolve()
    _, data, task = fs.command(manifest, index)
    if data.get('iterations') != TARGET or data.get('extension', {}).get('tool_sha256') != fs.sha(__file__):
        raise ValueError('Unexpected extension manifest or changed continuation tool')
    if data['extension']['result_root'] != str(root):
        raise ValueError('Extension belongs to another result directory')
    work = root/task['id']
    if work.resolve().parent != root or work.is_symlink():
        raise ValueError('Case must be a real directory inside the selected study')
    if not work.exists():
        return fs.run(manifest, index, root)
    with locked(work):
        _, actual_task, old, _ = fs.verify(work)
        if actual_task['seed_sha256'] != task['seed_sha256'] or actual_task['id'] != task['id']:
            raise ValueError('Existing case geometry differs from target task')
        if old.get('coordinate_frame') not in (None, 'native') or old.get('pass') != 1:
            raise ValueError('Only one frozen pass in native axes can be continued')
        case = work/'case'
        if case.is_symlink() or (case/'constant/dynamicMeshDict').exists():
            raise ValueError('Expected an existing stationary case')
        start = latest(case)
        if int(old['iters']) > TARGET:
            raise ValueError('Refusing to shorten a completed calculation')
        if start < int(old['iters']):
            raise ValueError('Latest saved fields precede the recorded completed calculation')
        # Preserve proof that this is the same mesh; never overwrite a different archive.
        for name in ['points', 'faces', 'owner', 'neighbour', 'boundary']:
            solved = next((case/'constant/polyMesh'/(name+s) for s in ['', '.gz']
                           if (case/'constant/polyMesh'/(name+s)).is_file()), None)
            archived = work/'fields'/NAME/'pass01/polyMesh'/solved.name if solved else None
            if solved is None or not archived.is_file() or fs.sha(solved) != fs.sha(archived):
                raise ValueError(f'Original and archived meshes differ: {name}')
        backup = work/'resume_2000_backup'
        backup.mkdir(exist_ok=True)
        for relative in ['study_case.json', 'study_run_status.json', 'out/'+NAME+'.json',
                         'out/'+NAME+'.fixed', 'case/system/controlDict', 'case/log.simpleFoam',
                         'case/postProcessing/forces/0/force.dat', 'case/postProcessing/forces/0/moment.dat']:
            source, target = work/relative, backup/relative
            if source.is_file() and not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
        receipt = dict(target_iteration=TARGET, restart_iteration=start,
                       manifest=str(manifest), CFD_convergence='unconfirmed', state='running')
        atomic_json(work/'extension_2000.json', receipt)
        began = time.monotonic()
        try:
            if start < TARGET:
                for tool in ['decomposePar', 'mpirun', 'simpleFoam', 'reconstructPar']:
                    if not shutil.which(tool):
                        raise ValueError(f'{tool} unavailable: load the OpenFOAM module in the job')
                control = case/'system/controlDict'
                control.write_text(control_text(control.read_text(encoding='utf-8')), encoding='utf-8')
                execute(['decomposePar','-force','-latestTime'], case, 'log.decomposePar.resume2000')
                execute(['mpirun','-np',str(int(os.environ.get('SLURM_NTASKS','72'))),
                         'simpleFoam','-parallel'], case, 'log.simpleFoam.resume2000')
                execute(['reconstructPar','-time',str(TARGET)], case, 'log.reconstruct.resume2000')
                shutil.copyfile(case/'log.simpleFoam.resume2000', case/'log.simpleFoam')
            if latest(case) != TARGET:
                raise ValueError('Solver did not save a complete iteration-2000 snapshot')
            merge_histories(case)
            row = refreshed_row(work, old, start, time.monotonic()-began)
            atomic_json(work/'out'/f'{NAME}.json', [row])
            mark_status(work/'out', NAME, 'fixed', row)
            atomic_json(work/'study_case.json', dict(manifest=str(manifest), task_index=index,
                        continuation=True, original_metadata=str(backup/'study_case.json')))
            fs.verify(work)
            receipt['state'] = 'completed'
            atomic_json(work/'extension_2000.json', receipt)
        except Exception as error:
            receipt.update(state='failed', error=str(error))
            atomic_json(work/'extension_2000.json', receipt)
            raise
        status = dict(CFD_returncode=0, verification='passed', postprocess_returncode=None)
        atomic_json(work/'study_run_status.json', status)
        post = subprocess.run([sys.executable, str(BASE/'coupled/nos6_results.py'),
                               'case', str(work), '--speed=30', '--wake=0', '--gif=off', '--no-summary'])
        status['postprocess_returncode'] = post.returncode
        atomic_json(work/'study_run_status.json', status)
        return post.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='mode', required=True)
    p = subs.add_parser('prepare'); p.add_argument('root'); p.add_argument('setup')
    p = subs.add_parser('run'); p.add_argument('manifest'); p.add_argument('index', type=int); p.add_argument('root')
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare(args.root, args.setup)
        return 0
    return run(args.manifest, args.index, args.root)


if __name__ == '__main__':
    raise SystemExit(main())
