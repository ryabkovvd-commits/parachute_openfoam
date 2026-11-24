#!/bin/bash
# Prepare all 21 states, submit CFD and schedule collection after every task ends.
set -euo pipefail
B=$(pwd -P)
test -f "$B/coupled/fixed_shape_study.py"
P=${NOS6_PYTHON:-python3}
SOURCE=${1:-run/nos6wall/8928}
TAG=${2:-8928_full_$(date +%Y%m%d_%H%M%S)}
PARALLEL=${NOS6_MAX_PARALLEL:-2}
if [[ ! $TAG =~ ^[a-zA-Z0-9_.-]+$ || ! $PARALLEL =~ ^[1-9][0-9]*$ ]]; then
    echo 'Invalid setup tag or parallel task count' >&2
    exit 2
fi
export NOS6_PYTHON="$P"
SETUP=$B/run/fixed_shape_study/setups/$TAG
"$P" "$B/coupled/fixed_shape_study.py" prepare "$SOURCE" "$SETUP" \
    --iterations="${NOS6_CFD_ITERS:-2000}" --mesh-factor="${NOS6_MESH:-1.25}"
J=$(sbatch --parsable --array="0-20%$PARALLEL" "$B/cluster/fixed_shape_study.slurm" "$SETUP/manifest.json")
J=${J%%;*}
if [[ ! $J =~ ^[0-9]+$ ]]; then echo "Invalid CFD job ID: $J" >&2; exit 2; fi
ROOT=$B/run/fixed_shape_study/$J
"$P" - "$SETUP" "$J" "$ROOT" <<'PY'
import json,sys
from pathlib import Path
Path(sys.argv[1],'submission.json').write_text(json.dumps(
    dict(CFD_array_job=sys.argv[2],result_root=sys.argv[3]),indent=2),encoding='utf-8')
PY
C=$(sbatch --parsable --dependency="afterany:$J" "$B/cluster/collect_fixed_shape_study.slurm" "$ROOT" "$SETUP/manifest.json")
C=${C%%;*}
"$P" - "$SETUP" "$C" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1],'submission.json');data=json.loads(p.read_text(encoding='utf-8'))
data['collection_job']=sys.argv[2];p.write_text(json.dumps(data,indent=2),encoding='utf-8')
PY
printf 'CFD array: %s\nCollection job: %s\nResults: %s\nSetup: %s\n' "$J" "$C" "$ROOT" "$SETUP"
