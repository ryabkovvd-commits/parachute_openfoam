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
GRADIENT=${2:-default}
BASELINE=
if [[ $GRADIENT != default && $GRADIENT != gauss ]]; then
    echo 'Second argument must be default or gauss' >&2; exit 2
fi
if [ -d "$M" ]; then
    BASELINE=$M
    "$P" "$B/coupled/permeability_models.py" verify "$BASELINE"
    M=$("$P" - "$BASELINE/model.json" <<'PY'
import json, sys
from pathlib import Path
config = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
print(Path(config['setup'])/'manifest.json')
PY
)
fi
gradient_opts=(--pressure-gradient="$GRADIENT")
if [[ $GRADIENT == gauss ]]; then
    if ! grep -Fq 'PRESSURE_GRADIENT = FLAG.get' "$B/coupled/cs.py"; then
        echo 'Update coupled/cs.py: pressure-gradient override is missing' >&2; exit 2
    fi
    if [ ! -x "$FOAM_USER_APPBIN/nos6MomentumAudit" ]; then
        echo 'Build the momentum-audit utility first using cluster/momentum_balance.slurm' >&2
        exit 2
    fi
fi
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
"$P" - "$M" "$SLURM_ARRAY_TASK_ID" "$W/model.json" "$GRADIENT" "$BASELINE" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
model = json.loads(path.read_text(encoding='utf-8'))['models'][int(sys.argv[2])]
model['setup'] = str(path.parent)
model['pressure_gradient'] = sys.argv[4]
if sys.argv[5]: model['baseline_case'] = sys.argv[5]
Path(sys.argv[3]).write_text(json.dumps(model, indent=2), encoding='utf-8')
PY
cd "$W"
"$P" "$B/coupled/cs.py" nos6_held 0 0 2000 1 "$SETUP/input_shape.npy" \
    --patch=nos6,cords --cord-mm=3.2 --cord-cd=1.2 \
    --угол=30 --полотнищ=96 --шаг=4 --сдвиг=0.02 \
    --V=30 --nu=1.46e-5 --fabric="${PARAMS[1]}" --fabric-relax=0.2 \
    --fixed-shape --mesh-from="$SETUP/polyMesh" "${gradient_opts[@]}"
"$P" "$B/coupled/permeability_models.py" verify "$W"
if [[ $GRADIENT == gauss ]]; then
    "$P" - "$W" "$BASELINE" <<'PY'
import json, sys
from pathlib import Path
work = Path(sys.argv[1])
row = json.loads((work/'out/nos6_held.json').read_text(encoding='utf-8'))[-1]
actual = (work/'case/system/fvSchemes').read_text(encoding='utf-8')
old = 'gradSchemes { default cellLimited Gauss linear 1; }'
new = 'gradSchemes { default cellLimited Gauss linear 1; grad(p) Gauss linear; }'
if row.get('pressure_gradient')!='gauss' or new not in actual:
    raise SystemExit('Requested grad(p) Gauss linear was not applied')
if sys.argv[2]:
    baseline = Path(sys.argv[2])
    original = (baseline/'case/system/fvSchemes').read_text(encoding='utf-8')
    if original.count(old)!=1 or actual!=original.replace(old,new):
        raise SystemExit('Numerical settings differ from the baseline beyond the pressure-gradient override')
    for relative in ['system/fvSolution','constant/transportProperties','constant/turbulenceProperties',
                     'constant/cords','system/fvOptions']:
        if (work/'case'/relative).read_text(encoding='utf-8')!=(baseline/'case'/relative).read_text(encoding='utf-8'):
            raise SystemExit(f'Additional settings differ from baseline: {relative}')
    reference = json.loads((baseline/'out/nos6_held.json').read_text(encoding='utf-8'))[-1]
    for key in ['D0','S0','V','rho','nu','hold','gores','rings','stride','lines','shear',
                'cord_width_m','cord_cd','iters']:
        if row.get(key)!=reference.get(key):
            raise SystemExit(f'Physical settings differ from baseline: {key}')
    if [float(x) for x in row['fabric_coefficients'].split(',')]!=[float(x) for x in reference['fabric_coefficients'].split(',')]:
        raise SystemExit('Fabric coefficients differ from baseline')
print('Checked: only grad(p) changed in fvSchemes',flush=True)
PY
    "$P" "$B/coupled/momentum_balance.py" "$(dirname "$W")" --model="${PARAMS[0]}" \
        --binary="$FOAM_USER_APPBIN/nos6MomentumAudit"
fi
"$P" "$B/coupled/nos6_results.py" case "$W" --speed=30 --no-summary --wake=0
