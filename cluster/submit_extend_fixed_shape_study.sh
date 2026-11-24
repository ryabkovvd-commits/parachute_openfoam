#!/bin/bash
# Continue the existing 21 cases; do not rerun meshing/structural relaxation.
set -euo pipefail
B=$(pwd -P)
P=${NOS6_PYTHON:-python3}
export NOS6_PYTHON="$P"
ROOT=$(realpath "${1:-run/fixed_shape_study/9016}")
OLD=$(basename "$ROOT")
PARALLEL=${NOS6_MAX_PARALLEL:-2}
[[ $PARALLEL =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid parallel count' >&2; exit 2; }
SETUP=$B/run/fixed_shape_study/setups/${OLD}_to2000_$(date +%Y%m%d_%H%M%S)_$$
DEPENDENCY=()
# Wait for remaining original tasks before touching their case directories.
ACTIVE=$(squeue -h -u "${USER:?}" -o '%F')
if [[ $OLD =~ ^[0-9]+$ ]] && printf '%s\n' "$ACTIVE" | grep -Fxq "$OLD"; then
    DEPENDENCY=(--dependency="afterany:$OLD")
fi
"$P" "$B/cluster/extend_fixed_shape_study.py" prepare "$ROOT" "$SETUP"
J=$(sbatch --parsable --array="0-20%$PARALLEL" "${DEPENDENCY[@]}" \
    "$B/cluster/extend_fixed_shape_study.slurm" "$SETUP/manifest.json" "$ROOT")
J=${J%%;*}
[[ $J =~ ^[0-9]+$ ]] || { echo 'Invalid continuation job ID' >&2; exit 2; }
C=$(sbatch --parsable --dependency="afterany:$J" \
    "$B/cluster/collect_fixed_shape_study.slurm" "$ROOT" "$SETUP/manifest.json")
C=${C%%;*}
"$P" - "$SETUP" "$ROOT" "$J" "$C" <<'PY'
import json,sys
from pathlib import Path
Path(sys.argv[1],'submission.json').write_text(json.dumps(dict(
    result_root=sys.argv[2],CFD_array_job=sys.argv[3],collection_job=sys.argv[4],
    target_iteration=2000),indent=2),encoding='utf-8')
PY
printf 'Continuation array: %s\nCollection job: %s\nResults: %s\nSetup: %s\n' "$J" "$C" "$ROOT" "$SETUP"
