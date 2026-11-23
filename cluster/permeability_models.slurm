#!/bin/bash
#SBATCH -J perm_models
#SBATCH -p compute1
#SBATCH -N 1
#SBATCH -n 72
#SBATCH --array=0-2%2
#SBATCH -t 2-00:00:00
#SBATCH -o %x-%A_%a.out
set -euo pipefail
B=${SLURM_SUBMIT_DIR:?Submit from the repository root}
module load "${NOS6_OPENFOAM_MODULE:-openfoam/2212}"
P=${NOS6_PYTHON:-python3}
M=$(realpath "${1:?Pass setup30/manifest.json as the first argument}")
SETUP=$(dirname "$M")
mapfile -t PARAMS < <("$P" - "$M" "${SLURM_ARRAY_TASK_ID:?}" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
data = json.loads(path.read_text(encoding='utf-8'))
model = data['models'][int(sys.argv[2])]
print(model['name'])
print(f"{model['D']:.17g},{model['Cw']:.17g}")
PY
)
if (( ${#PARAMS[@]} != 2 )); then echo 'Invalid model manifest' >&2; exit 2; fi
W=$B/run/permeability_models/$SLURM_ARRAY_JOB_ID/${PARAMS[0]}
mkdir -p "$W"
if [ -e "$W/model.json" ] || [ -d "$W/out" ]; then
    echo "Refusing to overwrite an existing probe: $W" >&2; exit 2
fi
"$P" - "$M" "$SLURM_ARRAY_TASK_ID" "$W/model.json" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
model = json.loads(path.read_text(encoding='utf-8'))['models'][int(sys.argv[2])]
model['setup'] = str(path.parent)
Path(sys.argv[3]).write_text(json.dumps(model, indent=2), encoding='utf-8')
PY
cd "$W"
"$P" "$B/coupled/cs.py" nos6_held 0 0 2000 1 "$SETUP/input_shape.npy" \
    --patch=nos6,cords --cord-mm=3.2 --cord-cd=1.2 \
    --угол=30 --полотнищ=96 --шаг=4 --сдвиг=0.02 \
    --V=30 --nu=1.46e-5 --fabric="${PARAMS[1]}" --fabric-relax=0.2 \
    --fixed-shape --mesh-from="$SETUP/polyMesh"
"$P" "$B/coupled/permeability_models.py" verify "$W"
"$P" "$B/coupled/nos6_results.py" case "$W" --speed=30 --no-summary --wake=0
