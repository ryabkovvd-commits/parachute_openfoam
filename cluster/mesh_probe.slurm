#!/bin/bash
#SBATCH -J meshprobe
#SBATCH -p compute1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --array=0-3%2
#SBATCH -t 4:00:00
#SBATCH -o %x-%A_%a.out
# Read-only probe of each failed case. No CFD, no structural relaxation.
set -eo pipefail
B=$SLURM_SUBMIT_DIR
test -n "$1"
SOURCE=$(realpath "$1")
MODE=legacy
if (( SLURM_ARRAY_TASK_ID >= 2 )); then MODE=inner; fi
FACTOR=1
if (( SLURM_ARRAY_TASK_ID % 2 == 1 )); then FACTOR=1.25; fi
S=$SOURCE/$MODE/a30/case
test -s "$S/constant/triSurface/canopy.stl"
W=$B/run/mesh_probe/$SLURM_ARRAY_JOB_ID/$MODE/m$FACTOR
if [ -e "$W" ]; then echo "Probe directory already exists: $W" >&2; exit 2; fi
mkdir -p "$W/constant"
cp -r "$S/system" "$W/system"
cp -r "$S/constant/triSurface" "$W/constant/triSurface"
P=python3
if [ -n "$NOS6_PYTHON" ]; then P=$NOS6_PYTHON; fi
"$P" - "$S" "$W" "$FACTOR" <<'PY'
from pathlib import Path
import hashlib,json,re,sys
s,w=map(Path,sys.argv[1:3]);factor=float(sys.argv[3])
p=w/'system/blockMeshDict';text=p.read_text()
pattern=r'(\bhex\s*\([^)]*\)\s*)\((\d+)\s+(\d+)\s+(\d+)\)'
hits=list(re.finditer(pattern,text))
if len(hits)!=1: raise ValueError('Probe expects one hex block')
original=[int(hits[0][i]) for i in (2,3,4)]
counts=[round(v*factor) for v in original]
text=re.sub(pattern,lambda m:m[1]+'('+ ' '.join(map(str,counts))+')',text)
p.write_text(text,encoding='utf-8')
# No physical fields are copied: splitMeshRegions maps only mesh-generated fields.
(w/'system/controlDict').write_text(
    'FoamFile {version 2.0; format ascii; class dictionary; object controlDict;}\n'
    'application snappyHexMesh; startFrom startTime; startTime 0;\n'
    'stopAt endTime; endTime 1; deltaT 1; writeControl timeStep; writeInterval 1;\n'
    'purgeWrite 0; writeFormat binary; writePrecision 8; writeCompression off;\n'
    'timeFormat general; timePrecision 6; runTimeModifiable false; functions {}\n')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(w/'probe_input.json').write_text(json.dumps(dict(source_case=str(s),factor=factor,
    source_block_counts=original,probe_block_counts=counts,
    source_stl_sha256=sha(s/'constant/triSurface/canopy.stl'),
    copied_stl_sha256=sha(w/'constant/triSurface/canopy.stl'),
    source_snappy_sha256=sha(s/'system/snappyHexMeshDict'),
    copied_snappy_sha256=sha(w/'system/snappyHexMeshDict'),
    note='Fixed source STL; only block counts change. No CFD or structural calculation.'),indent=2))
PY
module load openfoam/2212
cd "$W"
ret=0
blockMesh > log.blockMesh 2>&1 || ret=$?
if [ "$ret" = 0 ]; then snappyHexMesh -overwrite > log.snappyHexMesh 2>&1 || ret=$?; fi
if [ "$ret" = 0 ]; then
    cp constant/polyMesh/boundary boundary.beforeSplit
    splitMeshRegions -largestOnly -overwrite > log.split 2>&1 || ret=$?
    if [ "$ret" = 0 ]; then cp constant/polyMesh/boundary boundary.afterSplit; fi
fi
if [ "$ret" = 0 ]; then checkMesh -allTopology > log.checkMesh 2>&1 || ret=$?; fi
printf '%s\n' "$ret" > mesh_exit_code.txt
"$P" - "$W" <<'PY'
from pathlib import Path
import json,re,sys
w=Path(sys.argv[1]);result=json.loads((w/'probe_input.json').read_text())
for stage in ['beforeSplit','afterSplit']:
    p=w/('boundary.'+stage)
    if not p.is_file(): result[stage]=None;continue
    text=p.read_text();counts={}
    for name in ['canopy','canopy_slave']:
        match=re.search(r'\b'+name+r'\s*\{([^{}]*)\}',text,re.S)
        faces=re.search(r'\bnFaces\s+(\d+)\s*;',match[1]) if match else None
        if not faces: raise ValueError('Missing patch count: '+name)
        counts[name]=int(faces[1])
    counts['difference']=counts['canopy']-counts['canopy_slave']
    result[stage]=counts
result['utility_exit_code']=int((w/'mesh_exit_code.txt').read_text())
result['note']='Equal counts are necessary, not sufficient, for valid geometric pairing.'
(w/'mesh_probe.json').write_text(json.dumps(result,indent=2))
print('SOURCE',result['source_case'],'FACTOR',result['factor'])
for stage in ['beforeSplit','afterSplit']: print(stage,result[stage])
print('UTILITY_EXIT',result['utility_exit_code'])
print(w/'mesh_probe.json')
PY
exit "$ret"
