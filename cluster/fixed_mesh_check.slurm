#!/bin/bash
#SBATCH -J fixedmesh
#SBATCH -p compute1
#SBATCH -N 1
#SBATCH -n 72
#SBATCH -t 4:00:00
#SBATCH -o %x-%j.out
set -eo pipefail
B=$SLURM_SUBMIT_DIR
PROBE=$(realpath "$1")
W=$B/run/fixed_mesh_check/$SLURM_JOB_ID
if [ -e "$W" ]; then echo "Refusing to overwrite $W" >&2; exit 2; fi
mkdir -p "$W"
P=python3
if [ -n "$NOS6_PYTHON" ]; then P=$NOS6_PYTHON; fi
ITERS=200
if [ -n "$2" ]; then ITERS=$2; fi
module load openfoam/2212
"$P" - "$B" "$PROBE" "$W" "$ITERS" <<'PY'
from pathlib import Path
import hashlib,json,subprocess,sys
import numpy as np
b,probe,w=map(Path,sys.argv[1:4])
iterations=int(sys.argv[4]) if len(sys.argv)>4 else 200
if iterations<200 or iterations%200:
    raise ValueError('Iteration count must be a positive multiple of 200 (matching field and force write intervals)')
sys.path.insert(0,str(b/'coupled'))
from nos6_geometry import closed_circular,export_closed_surface
meta=json.loads((probe/'mesh_probe.json').read_text())
counts=meta.get('afterSplit') or {}
if meta['utility_exit_code']!=0 or counts.get('canopy',0)<=0 or counts.get('difference')!=0:
    raise ValueError('Probe has no successful equal-count final mesh')
source=Path(meta['source_case']).parent
record=json.loads((source/'out/nos6_held.json').read_text(encoding='utf-8'))[-1]
signature=json.loads((source/'out/nos6_held_config.json').read_text(encoding='utf-8'))
seed=source/'out/nos6_held_shape.npy'
positions=np.load(seed,allow_pickle=False)
mesh=closed_circular(record['D0'],record['lines'],gores=record['gores'],rings=record['rings'],
                     line_stride=record['stride'],shear_ratio=record['shear'])
if positions.shape!=mesh.position.shape or not np.isfinite(positions).all():
    raise ValueError('Invalid source shape')
mesh.position=positions.copy()
export_closed_surface(mesh,w/'input_surface.stl',subdivide=12)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
if sha(w/'input_surface.stl')!=meta['copied_stl_sha256']:
    raise ValueError('Saved output shape does not reproduce the exact probed STL')
np.save(w/'input_shape.npy',positions)
flags=dict(signature['physics']['flags'])
for key in ['--mesh','--mesh-from','--all-passes','--attitude','--fixed-shape']:
    flags.pop(key,None)
flags.update({'--fixed-shape':True,'--attitude':'legacy','--mesh-from':str(probe/'constant/polyMesh')})
options=[key if value is True else key+'='+str(value) for key,value in flags.items() if value is not False]
command=[sys.executable,str(b/'coupled/cs.py'),'nos6_held','0','0',str(iterations),'1',
         str(w/'input_shape.npy'),*options]
report=dict(source_run=str(source),source_last_completed_pass=record['pass'],
    mesh_probe=str(probe),source_attitude=record.get('attitude_mode'),
    input_shape_sha256=sha(w/'input_shape.npy'),input_stl_sha256=sha(w/'input_surface.stl'),
    mesh_probe_stl_sha256=meta['copied_stl_sha256'],command=command,
    CFD_iterations=iterations,
    note=f'Fixed geometry, {iterations} SIMPLE iterations; diagnostic only, not validated ADH or coupled convergence.')
(w/'fixed_mesh_input.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('Fixed CFD probe:',w,flush=True)
done=subprocess.run(command,cwd=w)
if done.returncode==0 and sha(w/'case/constant/triSurface/canopy.stl')!=report['input_stl_sha256']:
    from run_guard import record_failure
    record_failure(w/'out','nos6_held',ValueError('CFD STL differs from the probed geometry'))
    raise ValueError('CFD STL differs from the probed geometry')
raise SystemExit(done.returncode)
PY
