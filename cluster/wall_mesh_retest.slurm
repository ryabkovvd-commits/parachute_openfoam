#!/bin/bash
#SBATCH -J wallretest
#SBATCH -p compute1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --array=0-5
#SBATCH -t 4:00:00
#SBATCH -o %x-%A_%a.out
# Usage from project root:
# sbatch cluster/wall_mesh_retest.slurm run/nos6wall_refine/9105
# Geometry-only test: no CFD, no relaxation, no modifications of source cases.
set -euo pipefail
B=${SLURM_SUBMIT_DIR:?Submit from the project root}
SERIES=$(realpath "${1:?Supply a series directory containing a47 and a50}")
P=${NOS6_PROBE_PYTHON:-/home/016/apps/dashboard/anaconda3/bin/python}
test -x "$P"
TASK=${SLURM_ARRAY_TASK_ID:?}
FACTORS=(1.25 1.5 1.75)
ANGLES=(47 50)
if (( TASK < 0 || TASK > 5 )); then echo 'Expected array index 0-5' >&2; exit 2; fi
A=${ANGLES[$((TASK / 3))]}
FACTOR=${FACTORS[$((TASK % 3))]}
SOURCE="$SERIES/$(printf 'a%02d' "$A")"
W="$B/run/wall_mesh_retest/${SLURM_ARRAY_JOB_ID}/a${A}_m$FACTOR"
module load "${NOS6_OPENFOAM_MODULE:-openfoam/2212}"
"$P" - "$SOURCE" "$W" "$FACTOR" <<'PY'
from pathlib import Path
import hashlib,json,re,shutil,sys
source,work=map(Path,sys.argv[1:3]);factor=float(sys.argv[3])
case=source/'case'
failure=json.loads((source/'out/nos6_held.failed').read_text(encoding='utf-8'))
if 'Unpaired fabric faces' not in failure.get('error',''):
    raise ValueError('Expected a saved unpaired-face failure; do not probe an active case')
failed_pass=int(failure['last_completed_pass'])+1
needed=[case/'system/blockMeshDict',case/'system/snappyHexMeshDict',case/'constant/triSurface/canopy.stl']
if not all(p.is_file() for p in needed):raise ValueError('Incomplete source dictionaries or STL')
if factor not in (1.25,1.5,1.75):raise ValueError('Unexpected refinement factor')
if work.resolve()==source.resolve() or source.resolve().is_relative_to(work.resolve()):
    raise ValueError('Probe output must not contain the source case')
if work.exists():raise FileExistsError('Keep existing probes; submit with a new job ID')
work.mkdir(parents=True)
shutil.copytree(case/'system',work/'system')
shutil.copytree(case/'constant/triSurface',work/'constant/triSurface')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
block=work/'system/blockMeshDict';text=block.read_text(encoding='utf-8')
pattern=r'(\bhex\s*\([^)]*\)\s*)\((\d+)\s+(\d+)\s+(\d+)\)'
hits=list(re.finditer(pattern,text))
if len(hits)!=1:raise ValueError('Expected one background hex block')
original=[int(hits[0][i]) for i in (2,3,4)]
# Absolute factors relative to the project's original 40x40x64 background.
# Do not multiply an already refined 50x50x80 source by 1.5 a second time.
base=[40,40,64]
if not any(original==[round(v*f) for v in base] for f in (1.,1.25,1.5,1.75,2.)):
    raise ValueError('Unexpected source background; require a known uniform project factor')
counts=[round(v*factor) for v in base]
text=re.sub(pattern,lambda m:m[1]+'('+' '.join(map(str,counts))+')',text)
block.write_text(text,encoding='utf-8')
# No physical fields/functions are copied or evaluated in the mesh probe.
(work/'system/controlDict').write_text(
    'FoamFile {version 2.0; format ascii; class dictionary; object controlDict;}\n'
    'application snappyHexMesh; startFrom startTime; startTime 0;\n'
    'stopAt endTime; endTime 1; deltaT 1; writeControl timeStep; writeInterval 1;\n'
    'purgeWrite 0; writeFormat binary; writePrecision 8; writeCompression off;\n'
    'timeFormat general; timePrecision 6; runTimeModifiable false; functions {}\n',encoding='utf-8')
shape=source/'fields/nos6_held'/f'pass{failed_pass:02d}'/'input_shape.npy'
if shape.is_file():shutil.copyfile(shape,work/'input_shape.npy')
record=dict(source_run=str(source),source_case=str(case),failed_pass=failed_pass,factor=factor,
    source_block_counts=original,probe_block_counts=counts,
    source_stl_sha256=sha(needed[2]),copied_stl_sha256=sha(work/'constant/triSurface/canopy.stl'),
    source_snappy_sha256=sha(needed[1]),copied_snappy_sha256=sha(work/'system/snappyHexMeshDict'),
    source_shape_sha256=sha(shape) if shape.is_file() else None,
    note='Fixed failed-pass STL; ABSOLUTE background factors relative to 40x40x64. No CFD or structural solve.')
assert record['source_stl_sha256']==record['copied_stl_sha256']
assert record['source_snappy_sha256']==record['copied_snappy_sha256']
(work/'probe_input.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(f'Mesh-only probe: factor {factor}; source pass {failed_pass}; output {work}',flush=True)
PY
cd "$W"
surface_ret=0
if surfaceCheck -help 2>&1 | grep -q -- '-checkSelfIntersection'; then
    surfaceCheck -checkSelfIntersection constant/triSurface/canopy.stl > log.surfaceCheck 2>&1 || surface_ret=$?
else
    surfaceCheck constant/triSurface/canopy.stl > log.surfaceCheck 2>&1 || surface_ret=$?
    echo 'Self-intersection option unavailable; inspect geometry separately.' >> log.surfaceCheck
fi
mesh_ret=0
check_ret=-1
echo 'Running blockMesh'
blockMesh > log.blockMesh 2>&1 || mesh_ret=$?
if (( mesh_ret == 0 )); then
    echo 'Running snappyHexMesh'
    snappyHexMesh -overwrite > log.snappyHexMesh 2>&1 || mesh_ret=$?
fi
if (( mesh_ret == 0 )); then
    cp constant/polyMesh/boundary boundary.beforeSplit
    echo 'Running splitMeshRegions'
    splitMeshRegions -largestOnly -overwrite > log.split 2>&1 || mesh_ret=$?
    if (( mesh_ret == 0 )); then cp constant/polyMesh/boundary boundary.afterSplit; fi
fi
if (( mesh_ret == 0 )); then
    echo 'Running checkMesh'
    check_ret=0
    checkMesh -allTopology -allGeometry > log.checkMesh 2>&1 || check_ret=$?
fi
"$P" - "$W" "$mesh_ret" "$check_ret" "$surface_ret" <<'PY'
from pathlib import Path
import json,re,sys
work=Path(sys.argv[1]);mesh_ret,check_ret,surface_ret=map(int,sys.argv[2:5])
record=json.loads((work/'probe_input.json').read_text(encoding='utf-8'))
for stage in ['beforeSplit','afterSplit']:
    path=work/('boundary.'+stage)
    counts=None
    if path.is_file():
        text=path.read_text(encoding='utf-8',errors='replace');counts={}
        for patch in ['canopy','canopy_slave']:
            match=re.search(r'\b'+patch+r'\s*\{([^{}]*)\}',text,re.S)
            faces=re.search(r'\bnFaces\s+(\d+)\s*;',match[1]) if match else None
            if not faces:raise ValueError('Cannot read patch '+patch+' in '+str(path))
            counts[patch]=int(faces[1])
        counts['difference']=counts['canopy']-counts['canopy_slave']
    record[stage]=counts
log=work/'log.split'
regions=[]
if log.is_file():
    text=log.read_text(encoding='utf-8',errors='replace')
    part=re.search(r'Region\s+Cells\s*\n[-\s]+\n(.*?)(?:\n\s*\n|\Z)',text,re.S)
    if part:
        regions=[int(row[1]) for row in re.findall(r'^\s*(\d+)\s+(\d+)\s*$',part[1],re.M)]
    selected=re.search(r'Subsetting region\s+(\d+)\s+of size\s+(\d+)',text)
    record['selected_region']=int(selected[1]) if selected else None
record['region_cell_counts_before_split']=regions
record['mesh_utility_exit_code']=mesh_ret
record['checkMesh_exit_code']=check_ret
record['surfaceCheck_exit_code']=surface_ret
after=record['afterSplit'] or {}
record['equal_count_quality_candidate']=bool(mesh_ret==0 and check_ret==0 and surface_ret==0 and len(regions)==1 and after.get('canopy',0)>0 and after.get('difference')==0 and after==record['beforeSplit'])
record['scope']='Single connected region, equal patch counts and successful utilities are necessary checks only. Inspect surfaceCheck for self-intersection; reciprocal pair geometry and CFD still require validation. No ADH calculated.'
(work/'mesh_probe.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
summary=[
    'SOURCE: '+record['source_case'],
    'ABSOLUTE FACTOR / BACKGROUND: '+str(record['factor'])+' / '+str(record['probe_block_counts']),
    'BEFORE: '+str(record['beforeSplit']), 'AFTER: '+str(record['afterSplit']),
    'REGION CELL COUNTS: '+str(regions),
    'MESH / CHECK / SURFACE EXIT: '+str([mesh_ret,check_ret,surface_ret]),
    'CANDIDATE ONLY: '+str(record['equal_count_quality_candidate']),
    record['scope']]
surface_log=work/'log.surfaceCheck'
if surface_log.is_file():summary+=['SURFACE CHECK TAIL:']+surface_log.read_text(encoding='utf-8',errors='replace').splitlines()[-25:]
(work/'summary.txt').write_text('\n'.join(summary)+'\n',encoding='utf-8')
print('SOURCE:',record['source_case'])
print('FACTOR:',record['factor'],'BACKGROUND:',record['probe_block_counts'])
print('BEFORE:',record['beforeSplit'])
print('AFTER: ',record['afterSplit'])
print('REGION CELL COUNTS:',regions)
print('MESH/CHECK EXIT:',mesh_ret,check_ret)
print('CANDIDATE (not yet a geometrically validated surface):',record['equal_count_quality_candidate'])
print(work/'mesh_probe.json',flush=True)
raise SystemExit(0 if record['equal_count_quality_candidate'] else 3)
PY
