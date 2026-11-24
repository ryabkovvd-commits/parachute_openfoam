"""Prepare/compare a real rotated CFD case and draw its coordinate scheme."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from nosarev_frame import ROTATION,to_speed,to_native,FRAME_NAME
from nos6_results import read_records,write_json


def prepare(source,work,angle):
    source,work=Path(source).resolve(),Path(work).resolve()
    row=read_records(source)[-1]
    if row.get('coordinate_frame')==FRAME_NAME:
        raise ValueError('Source is already in Nosarev axes; do not rotate it twice')
    if row.get('hold') is None or not np.isclose(float(row['hold']),angle):
        raise ValueError('Coordinate control must keep the source angle unchanged')
    archive=source/'fields/nos6_held'/f"pass{int(row['pass']):02d}"
    positions=np.load(archive/'input_shape.npy',allow_pickle=False)
    if positions.ndim!=2 or positions.shape[1]!=3 or not np.isfinite(positions).all():
        raise ValueError('Invalid exact source CFD input shape')
    input_case=work/'input_mesh_case'
    if input_case.exists() or (work/'seed.npy').exists():
        raise ValueError('Preparation destination already exists')
    for name in ['points','faces','owner','neighbour','boundary']:
        if not any((archive/'polyMesh'/(name+s)).is_file() for s in ['','.gz']):
            raise ValueError('Source archive does not contain a complete matching mesh')
    (input_case/'constant').mkdir(parents=True)
    (input_case/'system').mkdir()
    (input_case/'0').mkdir()
    shutil.copytree(archive/'polyMesh',input_case/'constant/polyMesh')
    (input_case/'system/controlDict').write_text(
        'FoamFile {version 2.0; format ascii; class dictionary; object controlDict;}\n'
        'application transformPoints; startFrom startTime; startTime 0; stopAt endTime; endTime 0;\n'
        'deltaT 1; writeControl timeStep; writeInterval 1; writeFormat binary; writePrecision 12; writeCompression off;\n')
    np.save(work/'seed.npy',to_speed(positions))
    signature=json.loads((source/'out/nos6_held_config.json').read_text(encoding='utf-8'))
    result=dict(coordinate_frame=FRAME_NAME,source_run=str(source),source_pass=int(row['pass']),
                alpha_deg=float(angle),rotation_native_to_nosarev=ROTATION.tolist(),
                source_flags=signature['physics']['flags'],source_row=row,
                seed_sha256=hashlib.sha256((work/'seed.npy').read_bytes()).hexdigest(),
                expected_coefficients=dict(cx=row['cx'],cy=-row['cy'],mz=row['mz']),
                note='The copied mesh still needs transformPoints: axis (1 -1 -1), angle 120 degrees. No field is reused.')
    write_json(work/'coordinate_control.json',result)
    return result


def run_fixed(work,iterations):
    work=Path(work).resolve()
    if iterations<200 or iterations%200:
        raise ValueError('Use a positive multiple of 200 iterations')
    info=json.loads((work/'coordinate_control.json').read_text(encoding='utf-8'))
    flags=dict(info['source_flags'])
    patches=[p for p in flags.get('--patch','nos6,cords').split(',') if p]
    if 'nosarev_axes' not in patches: patches.append('nosarev_axes')
    for key in ['--mesh','--mesh-from','--attitude','--fixed-shape','--seed-frame','--mesh-frame','--all-passes']:
        flags.pop(key,None)
    flags.update({'--patch':','.join(patches),'--fixed-shape':True,'--attitude':'legacy',
                  '--seed-frame':'nosarev','--mesh-frame':'nosarev',
                  '--mesh-from':str(work/'input_mesh_case/constant/polyMesh')})
    options=[key if value is True else key+'='+str(value) for key,value in flags.items() if value is not False]
    command=[sys.executable,str(Path(__file__).with_name('cs.py')),'nos6_held','0','0',
             str(iterations),'1',str(work/'seed.npy'),*options]
    info.update(command=command,target_iterations=iterations)
    write_json(work/'coordinate_control.json',info)
    return subprocess.run(command,cwd=work).returncode


def compare(work):
    work=Path(work)
    info=json.loads((work/'coordinate_control.json').read_text(encoding='utf-8'))
    row=read_records(work)[-1]
    if row.get('coordinate_frame')!=FRAME_NAME:
        raise ValueError('Comparison target must be calculated in nosarev_speed axes')
    result=dict(expected=info['expected_coefficients'],actual={k:row[k] for k in ['cx','cy','mz']},
                difference={k:row[k]-info['expected_coefficients'][k] for k in ['cx','cy','mz']},
                same_iterations=info['source_row'].get('iters')==row.get('iters'),
                note='A proper rotation preserves Cx and pitch moment; signed Cy is reversed by the selected positive lift axis. Iterative/grid convergence is a separate check.')
    write_json(work/'out/coordinate_comparison.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return result


def diagram(work):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    work=Path(work)
    row=read_records(work)[-1]
    shape=np.load(work/'fields/nos6_held'/f"pass{int(row['pass']):02d}"/'input_shape.npy',allow_pickle=False)
    g,r=int(row['gores']),int(row['rings']);anchor=shape[(r+1)*g]
    alpha=np.radians(float(row['hold']))
    # Fixed camera: +x down-left and +y up-left, as in the source figure.
    beta=np.radians(30.)
    screen=np.array([[-np.sin(beta),-np.cos(beta),0.],[-np.cos(beta),np.sin(beta),0.]])
    project=lambda v:(np.asarray(v)-anchor)@screen.T
    fig,ax=plt.subplots(figsize=(7,7),constrained_layout=True)
    grid=shape[:(r+1)*g].reshape(r+1,g,3)
    for j in range(0,g,int(row['stride'])):
        ax.plot(*project(np.array([anchor,grid[-1,j]])).T,color='.65',lw=.55)
    profile=np.vstack([grid[::-1,0],shape[-1],grid[:,g//2]])
    ax.plot(*project(profile).T,color='#066f93',lw=2)
    length=float(row['D0'])*.4
    for vector,label in [(np.array([1.,0.,0.]),'+x'),(np.array([0.,1.,0.]),'+y')]:
        tip=(vector*length)@screen.T
        ax.annotate('',tip,(0,0),arrowprops=dict(arrowstyle='->',color='black',lw=1.3))
        ax.text(*((vector*length*1.12)@screen.T),label,ha='center',va='center')
    start=np.array([.85*length,1.5*length,0.])
    end=start+np.array([-1.1*length,0,0])
    ax.annotate('',end@screen.T,start@screen.T,arrowprops=dict(arrowstyle='->',color='#a25418',lw=2))
    ax.text(*((start+end)/2@screen.T+np.array([-.12*length,.4*length])),
            f'Набегающий поток\nU = (-{row["V"]:g}, 0, 0) м/с',color='#a25418',
            fontsize=10,ha='center',va='bottom')
    baseline=np.array([[0.,0.,0.],[-length,0.,0.]])@screen.T
    ax.plot(*baseline.T,':',color='.45',lw=.8)
    angles=np.linspace(0,alpha,48)
    arc=.7*length*np.c_[-np.cos(angles),-np.sin(angles),np.zeros_like(angles)]@screen.T
    ax.plot(*arc.T,color='.35',lw=.9)
    label=.92*length*np.array([-np.cos(alpha/2),-np.sin(alpha/2),0.])@screen.T
    ax.text(*label,'α',ha='center',va='center')
    t=np.linspace(-.3,.9,32)
    orbit=.35*length*np.c_[np.cos(t),np.sin(t),np.zeros_like(t)]@screen.T
    ax.plot(*orbit[:-1].T,color='#7b3294')
    ax.annotate('',orbit[-1],orbit[-3],arrowprops=dict(arrowstyle='->',color='#7b3294'))
    ax.text(*(orbit[len(orbit)//2]+np.array([.15*length,0])), '+Mz',color='#7b3294')
    ax.plot(0,0,'kx',ms=9)
    ax.text(.08*length,-.08*length,'O: коуш',fontsize=10)
    ax.text(-.9*length,.05*length,'⊗ +z',fontsize=12)
    bounds=np.vstack([project(shape),np.array([start,end,[length,0,0],[0,length,0]])@screen.T,
                      [[-.95*length,0]],label])
    lo,hi=bounds.min(0),bounds.max(0)
    margin=.16*max(hi-lo)
    ax.set_xlim(lo[0]-margin,hi[0]+margin)
    ax.set_ylim(lo[1]-margin,hi[1]+margin)
    ax.set_aspect('equal');ax.set_axis_off()
    ax.set_title(f'Скоростная система Носарева · α = {row["hold"]:g}°\nГеометрия, использованная CFD')
    (work/'pics').mkdir(exist_ok=True)
    output=work/'pics/coordinate_scheme.png'
    fig.savefig(output,dpi=180);plt.close(fig)
    return str(output)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='mode',required=True)
    p=sub.add_parser('prepare');p.add_argument('source',type=Path);p.add_argument('work',type=Path);p.add_argument('angle',type=float)
    p=sub.add_parser('run-fixed');p.add_argument('work',type=Path);p.add_argument('iterations',type=int)
    for name in ['compare','diagram']:
        p=sub.add_parser(name);p.add_argument('work',type=Path)
    args=parser.parse_args()
    if args.mode=='prepare': prepare(args.source,args.work,args.angle)
    elif args.mode=='run-fixed': return run_fixed(args.work,args.iterations)
    elif args.mode=='compare': compare(args.work)
    else: print(diagram(args.work))
    return 0


if __name__=='__main__': raise SystemExit(main())
