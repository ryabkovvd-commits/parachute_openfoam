#!/bin/bash
#SBATCH -J wall45to50
#SBATCH -p compute1
#SBATCH -N 1
#SBATCH -n 36
#SBATCH --array=45-50%2
#SBATCH -t 2-00:00:00
#SBATCH -o %x-%A_%a.out
set -euo pipefail
B=${SLURM_SUBMIT_DIR:?Submit from porousDisk_bundle root}
P=${NOS6_PYTHON:-/home/016/apps/dashboard/anaconda3/bin/python}
test -f "$B/coupled/cs.py"
test -x "$P"
module load "${NOS6_OPENFOAM_MODULE:-openfoam/2212}"
A=${SLURM_ARRAY_TASK_ID:?}
if (( A < 45 || A > 50 )); then echo 'Expected an angle from 45 to 50' >&2; exit 2; fi
W="$B/run/nos6wall_refine/${SLURM_ARRAY_JOB_ID}/$(printf 'a%02d' "$A")"
mkdir -p "$(dirname "$W")"
# A new job ID gets a separate series. Never replace an existing case.
mkdir "$W" || { echo "Case already exists: $W" >&2; exit 2; }
cp -a "$B/coupled" "$W/code"
mkdir "$W/pics"
cd "$W"
"$P" - <<'PY'
import hashlib,json,pathlib
p=pathlib.Path('code')
files={str(f.relative_to(p)):hashlib.sha256(f.read_bytes()).hexdigest()
       for f in sorted(p.rglob('*')) if f.is_file() and f.suffix in {'.py','.cpp','.h','.txt'}}
pathlib.Path('source_sha256.json').write_text(json.dumps(files,indent=2),encoding='utf-8')
src=(p/'cs.py').read_text(encoding='utf-8')
for flag in ('--all-passes','--attitude','--shape-tol'):
    if flag not in src: raise SystemExit('Update coupled/ from the project first; missing '+flag)
PY
# Same physical profile as 8928: sealed wall, 24 cords, legacy attitude,
# limited pressure gradient, 96x12 structure, V=30. No --fabric or inner mode.
# Twelve full outer passes retain a longer moment history. 45/50 are controls.
ret=0
"$P" "$W/code/cs.py" nos6_held 0 0 2000 12 \
    --patch=nos6,cords --cord-mm=3.2 --cord-cd=1.2 \
    --угол="$A" --полотнищ=96 --шаг=4 --сдвиг=0.02 \
    --attitude=legacy --shape-tol=0.03 --cx-tol=0.003 --all-passes \
    --V=30 --nu=1.46e-5 2>&1 | tee solve.log || ret=$?
if [[ ! -s out/nos6_held.json ]]; then echo 'No completed passes' >&2; exit 1; fi
post_ret=0
"$P" "$W/code/nos6_results.py" case "$W" --speed=30 --wake=1 --gif=off || post_ret=$?
echo "Saved case: $W; solver status=$ret; postprocessing status=$post_ret"
if (( ret != 0 )); then exit "$ret"; fi
exit "$post_ret"
