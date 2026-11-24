"""Validation and provenance for CFD/structure runs; no OpenFOAM side effects."""
import hashlib
import json
from pathlib import Path
import re

import numpy as np


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    temp.replace(path)


def geometry_quality(mesh):
    pos=np.asarray(mesh.position)
    if pos.shape!=mesh.flat.shape or not np.isfinite(pos).all():
        raise ValueError('Invalid structural coordinates')
    corners=pos[mesh.quads]
    area=(np.linalg.norm(np.cross(corners[:,1]-corners[:,0],corners[:,2]-corners[:,0]),axis=1)
          +np.linalg.norm(np.cross(corners[:,2]-corners[:,0],corners[:,3]-corners[:,0]),axis=1))/2
    tolerance=1e-14*mesh.nominal_diameter**2
    if np.any(area<=tolerance):
        raise ValueError('Collapsed or degenerate fabric panels')
    lengths=np.linalg.norm(pos[mesh.rods[:,1]]-pos[mesh.rods[:,0]],axis=1)
    if np.any(lengths<=1e-12*mesh.nominal_diameter):
        raise ValueError('Zero-length structural member')
    return dict(min_panel_area_m2=float(area.min()),surface_area_m2=float(area.sum()),
                intersection_check='not performed; radial normal sign is not an inversion test')


def source_hashes(folder,patches):
    paths=['coupled.py','cs.py','vtp.py','run_guard.py','nos6_geometry.py','fabricU.txt','cordsFv.txt']
    paths += [name+'.py' for name in patches.split(',') if name]
    paths += ['parafoil/inflation/'+name+'.py' for name in ['mesh','materials','relax','exchange','pressure','constraints']]
    return {p:hashlib.sha256((folder/p).read_bytes()).hexdigest() for p in sorted(set(paths)) if (folder/p).is_file()}


def begin_run(out,name,config,history,shape,*,sources):
    """Refuse legacy/changed histories before any CFD-case write or cleanup."""
    path=Path(out)/(name+'_config.json')
    signature=dict(physics=config,source_sha256=sources)
    if history:
        if not path.is_file():
            raise ValueError('Legacy history has no configuration fingerprint. Use a NEW run directory; keep the old results.')
        previous=json.loads(path.read_text(encoding='utf-8'))
        if previous!=signature:
            raise ValueError('Incompatible resume configuration or solver source. Use a NEW run directory.')
        if not Path(shape).is_file():
            raise ValueError('History exists but its final structural shape is missing')
        numbers=[int(row['pass']) for row in history]
        if numbers!=list(range(1,len(history)+1)):
            raise ValueError('Non-contiguous or duplicate pass history')
    else:
        if Path(shape).exists():
            raise ValueError('Partial previous run without history. Use a NEW directory or inspect the saved files.')
        if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=signature:
            raise ValueError('Incompatible partial-run configuration; use a NEW directory.')
        atomic_json(path,signature)
    return signature


def mark_status(out,name,suffix,row):
    for other in ['done','НЕСОШЁЛСЯ','fixed','failed']:
        if other!=suffix:
            (Path(out)/(name+'.'+other)).unlink(missing_ok=True)
    atomic_json(Path(out)/(name+'.'+suffix),row)


def record_failure(out,name,error):
    path=Path(out)/(name+'.json')
    history=json.loads(path.read_text(encoding='utf-8')) if path.is_file() else []
    mark_status(out,name,'failed',dict(last_completed_pass=history[-1]['pass'] if history else 0,
        error_type=type(error).__name__,error=str(error)))


def table(path):
    rows=[]
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if line.strip() and not line.lstrip().startswith('#'):
            rows.append([float(x) for x in line.replace('(',' ').replace(')',' ').split()])
    a=np.asarray(rows,dtype=float)
    if a.ndim!=2 or len(a)==0 or a.shape[1]<4 or not np.isfinite(a).all():
        raise ValueError(f'Invalid/non-finite solver history: {path}')
    return a


def check_cfd(case,archive,pressure_time,q,s0,diameter,limit=20.):
    """Reject incomplete/diverged runs. This is NOT a CFD convergence test."""
    case,archive=Path(case),Path(archive)
    text=(case/'log.simpleFoam').read_text(encoding='utf-8',errors='replace')
    if 'FOAM FATAL' in text or re.search(r'\b(?:nan|infinity)\b',text,re.I) or not re.search(r'^\s*End\s*$',text,re.M):
        raise ValueError('CFD log is incomplete, fatal or non-finite')
    for root in [case/'constant/polyMesh',archive/'polyMesh']:
        for name in ['points','faces','owner','neighbour','boundary']:
            if not any((root/(name+suffix)).is_file() and (root/(name+suffix)).stat().st_size for suffix in ['', '.gz']):
                raise ValueError(f'Incomplete archived CFD mesh: {root/name}')
    for name in ['points','faces','owner','neighbour','boundary']:
        original=next(p for s in ['', '.gz'] if (p:=case/'constant/polyMesh'/(name+s)).is_file())
        copied=archive/'polyMesh'/original.name
        if not copied.is_file() or hashlib.sha256(original.read_bytes()).digest()!=hashlib.sha256(copied.read_bytes()).digest():
            raise ValueError('Archive mesh differs from the solved case')
    for name in ['U','p','phi','nut']:
        if not any((archive/pressure_time/(name+suffix)).is_file() and (archive/pressure_time/(name+suffix)).stat().st_size for suffix in ['', '.gz']):
            raise ValueError(f'Missing archived field: {archive/pressure_time/name}')
        originals=[case/pressure_time/(name+s) for s in ['', '.gz'] if (case/pressure_time/(name+s)).is_file()]
        if len(originals)!=1 or not (archive/pressure_time/originals[0].name).is_file() or hashlib.sha256(originals[0].read_bytes()).digest()!=hashlib.sha256((archive/pressure_time/originals[0].name).read_bytes()).digest():
            raise ValueError('Archive fields differ from the solved case')
    outputs=[];sample_counts=[]
    for filename,scale in [('force.dat',q*s0),('moment.dat',q*s0*diameter)]:
        a=table(case/'postProcessing/forces/0'/filename)
        if not np.isclose(a[-1,0],float(pressure_time),rtol=0,atol=1e-7):
            raise ValueError('Force history and pressure snapshot times differ')
        recent=a[-min(10,len(a)):]
        if np.max(np.abs(recent[:,1:4]))/scale>limit:
            raise ValueError('Catastrophic CFD force/moment growth; result rejected')
        outputs.append(np.ptp(recent[:,1:4]/scale,axis=0).tolist())
        sample_counts.append(len(recent))
    return dict(CFD_convergence='unconfirmed',CFD_validation='finite complete snapshot',
                force_last_window_range=outputs[0],moment_last_window_range=outputs[1],
                force_window_samples=sample_counts[0],moment_window_samples=sample_counts[1],
                coefficient_rejection_limit=limit)
