"""Prepare and collect a fixed-geometry comparison of three permeability laws.

python coupled/permeability_models.py prepare run/nos6wall/8928/a30 run/permeability_models/setup30
python coupled/permeability_models.py collect run/permeability_models/JOB
"""
import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import aero_diagnostics as diagnostics

NAME = 'nos6_held'
CORE = ('points', 'faces', 'owner', 'neighbour', 'boundary')
LABELS = {'df': 'Двухчленный закон', 'darcy': 'Линейный закон', 'quadratic': 'Квадратичный закон'}


def speed(dp, d, c, rho=1.225):
    """Inverse of dp/rho = d*u + c*abs(u)*u, including both limiting laws."""
    if not np.isfinite([d, c, rho]).all() or min(d, c)<0 or d+c<=0 or rho<=0:
        raise ValueError('Invalid permeability coefficients or density')
    jump = np.asarray(dp, dtype=float)/rho
    magnitude = np.abs(jump)
    denominator = d+np.sqrt(d*d+4*c*magnitude)
    result = np.divide(2*magnitude, denominator, out=np.zeros_like(magnitude), where=denominator>0)
    return np.sign(jump)*result


def models(reference_pa=49., rho=1.225):
    if not np.isfinite(reference_pa) or reference_pa<=0:
        raise ValueError('Reference pressure must be positive')
    d, c = 33.2, 28.6
    u = float(speed(reference_pa, d, c, rho))
    return [dict(name='df', D=d, Cw=c), dict(name='darcy', D=d+c*u, Cw=0.),
            dict(name='quadratic', D=0., Cw=c+d/u)]


def fingerprint(folder):
    """Hash the five mesh-defining files; auxiliary zones do not define cell geometry."""
    files = {}
    for name in CORE:
        path = folder/name
        if not path.is_file():
            path = folder/(name+'.gz')
        if not path.is_file() or not path.stat().st_size:
            raise FileNotFoundError(f'Incomplete polyMesh: {folder/name}')
        files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return files


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def law_plot(output, records, rho=1.225, reference_pa=49.):
    output.mkdir(parents=True, exist_ok=True)
    dp = np.linspace(0, 900, 451)
    with (output/'permeability_laws.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.writer(stream)
        writer.writerow(['dp_Pa']+[r['name']+'_Un_m_s' for r in records])
        writer.writerows(zip(dp, *(speed(dp,r['D'],r['Cw'],rho) for r in records)))
    fig, ax = plt.subplots(figsize=(8,5), constrained_layout=True)
    for r in records:
        ax.plot(dp, speed(dp,r['D'],r['Cw'],rho), label=LABELS[r['name']])
    ax.axvline(reference_pa, color='0.5', lw=.8, ls=':')
    ax.set(xlabel='Перепад давления Δp, Па', ylabel='Нормальная скорость протекания, м/с',
           title=f'Законы проницаемости: одинаковая скорость при Δp = {reference_pa:g} Па', xlim=(0,900), ylim=(0,None))
    ax.legend()
    ax.grid(alpha=.25)
    path = output/'permeability_laws.png'
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def prepare(source, output, mesh_source=None):
    source, output = source.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f'Use a new setup directory; nothing overwritten: {output}')
    rows = json.loads((source/'out'/f'{NAME}.json').read_text(encoding='utf-8'))
    row = max(rows,key=lambda r:int(r['pass']))
    diameter = float(row.get('D0',np.sqrt(24/np.pi)))
    expected = dict(gores=96, rings=12, stride=4)
    if any(int(row[k])!=v for k,v in expected.items()) or not np.isclose(diameter,np.sqrt(24/np.pi)):
        raise ValueError('This pilot requires a closed 6 m2 canopy, 96 gores, 12 rings and 24 cords')
    if not np.isclose(float(row['hold']),30.) or not np.isclose(float(row['lines']),diameter):
        raise ValueError('Use the held 30-degree case with l=1')
    shape = diagnostics.input_shape(source, NAME, row)
    positions = np.load(shape,allow_pickle=False)
    template = diagnostics.make_mesh(row,diameter)
    if positions.shape!=template.position.shape or not np.isfinite(positions).all():
        raise ValueError('Invalid CFD input shape')
    canopy_axis = positions[:13*96].mean(0)
    tilt = np.degrees(np.arccos(np.clip(canopy_axis[2]/np.linalg.norm(canopy_axis),-1,1)))
    if abs(tilt-30.)>.01 or np.linalg.norm(positions[template.confluence[0]])>1e-9:
        raise ValueError('Input shape must have 30-degree tilt and the confluence at the origin')
    archive = source/'fields'/NAME/f"pass{int(row['pass']):02d}"
    mesh = mesh_source.resolve() if mesh_source is not None else archive/'polyMesh'
    digest = fingerprint(mesh)
    # An explicit alternative is never silently substituted for the archived mesh.
    if mesh!=archive/'polyMesh':
        print('NOTICE: explicit mesh source; confirm it belongs to the same CFD input/pass:',mesh)
    records = models()
    output.mkdir(parents=True)
    shutil.copyfile(shape,output/'input_shape.npy')
    shutil.copytree(mesh,output/'polyMesh')
    manifest = dict(source_case=str(source), source_pass=int(row['pass']), source_shape=str(shape),
                    source_mesh=str(mesh), explicit_mesh_source=mesh_source is not None,
                    mesh_sha256=digest, shape_sha256=hashlib.sha256((output/'input_shape.npy').read_bytes()).hexdigest(),
                    angle=30., diameter=diameter, rho=1.225, speed=30., nu=1.46e-5,
                    gores=96, rings=12, stride=4, cord_width_m=.0032, cord_cd=1.2,
                    reference_dp_Pa=49., reference_Un_m_s=float(speed(49.,33.2,28.6)), models=records,
                    note='Matched to the existing law at 49 Pa; not independent material calibration. CFD convergence must be assessed separately.')
    write_json(output/'manifest.json',manifest)
    law_plot(output,records)
    print(output/'manifest.json')
    for r in records:
        print(f"{r['name']}: --fabric={r['D']:.12g},{r['Cw']:.12g}")
    return manifest


def verify(work):
    """Reject a run if its shape, coefficients or exact mesh differ from the setup."""
    config = json.loads((work/'model.json').read_text(encoding='utf-8'))
    setup = Path(config['setup'])
    manifest = json.loads((setup/'manifest.json').read_text(encoding='utf-8'))
    rows = json.loads((work/'out'/f'{NAME}.json').read_text(encoding='utf-8'))
    if len(rows)!=1 or not rows[0].get('fixed_shape'):
        raise ValueError('Expected one fixed-shape CFD pass')
    row = rows[0]
    if config['name'] not in LABELS:
        raise ValueError('Unknown model name')
    params = [float(x) for x in row['fabric_coefficients'].split(',')]
    if not np.allclose(params,[config['D'],config['Cw']],rtol=1e-12,atol=1e-12):
        raise ValueError('Permeability coefficients do not match the selected model')
    if fingerprint(work/'case/constant/polyMesh')!=manifest['mesh_sha256']:
        raise ValueError('CFD mesh differs from the shared setup')
    if hashlib.sha256((setup/'input_shape.npy').read_bytes()).hexdigest()!=manifest['shape_sha256']:
        raise ValueError('Shared input shape changed after preparation')
    original = np.load(setup/'input_shape.npy',allow_pickle=False)
    for path in [work/'fields'/NAME/'pass01/input_shape.npy',work/'out'/f'{NAME}_shape.npy']:
        if not np.array_equal(original,np.load(path,allow_pickle=False)):
            raise ValueError(f'Frozen input shape changed: {path}')
    return config, manifest, row


def collect(root):
    result = []
    common = None
    for model in models():
        work = root/model['name']
        config,manifest,row = verify(work)
        signature = (manifest['mesh_sha256'],manifest['shape_sha256'])
        if config['name']!=model['name'] or (common is not None and signature!=common):
            raise ValueError('Models do not share the same input shape and mesh')
        common = signature
        result.append(dict(model=config['name'], D=config['D'], Cw=config['Cw'],
                           cx=row['cx'], cy=row['cy'], mz=row['mz'], K=row['side']/row['drag'],
                           dCp=row['dcp'], cells=row['cells'], cord_cx=row.get('cord_cx'),
                           cord_mz=row.get('cord_mz'), CFD_convergence='unconfirmed'))
    with (root/'comparison.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)
    fig, axes=plt.subplots(1,3,figsize=(11,4),constrained_layout=True)
    for key, ax in zip(['cx','cy','mz'],axes):
        ax.bar([LABELS[r['model']] for r in result],[r[key] for r in result])
        ax.set(ylabel=key,title='α = 30°, фиксированная форма')
        ax.tick_params(axis='x',labelrotation=25)
        ax.axhline(0,color='0.4',lw=.7)
    fig.savefig(root/'comparison.png',dpi=180)
    plt.close(fig)
    print(root/'comparison.csv')
    print('CFD convergence is unconfirmed; inspect solver residuals and force histories before interpreting differences.')
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_subparsers(dest='mode',required=True)
    p=modes.add_parser('prepare')
    p.add_argument('source',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--mesh-source',type=Path,help='Explicit complete mesh of the same CFD input/pass')
    p=modes.add_parser('collect')
    p.add_argument('root',type=Path)
    p=modes.add_parser('verify')
    p.add_argument('work',type=Path)
    p=modes.add_parser('laws')
    p.add_argument('output',type=Path)
    args=parser.parse_args(argv)
    try:
        if args.mode=='prepare': prepare(args.source,args.output,args.mesh_source)
        elif args.mode=='collect': collect(args.root.resolve())
        elif args.mode=='verify': verify(args.work.resolve())
        else: print(law_plot(args.output,models()))
    except (OSError,ValueError,KeyError) as error:
        parser.exit(1,f'{error}\n')


if __name__=='__main__':
    main()
