#!/bin/bash
#SBATCH -J nos6method
#SBATCH -p compute1
#SBATCH -N 1
#SBATCH -n 72
#SBATCH --array=0-1%2
#SBATCH -t 2-00:00:00
#SBATCH -o %x-%A_%a.out
set -eo pipefail
B=$SLURM_SUBMIT_DIR
test -f "$B/coupled/cs.py"
module load openfoam/2212
P=python3
if [ -n "$NOS6_PYTHON" ]; then P=$NOS6_PYTHON; fi
A=30
if [ -n "$1" ]; then A=$1; fi
MODE=legacy
if [ "$SLURM_ARRAY_TASK_ID" = 1 ]; then MODE=inner; fi
ROOT=$B/run/method_check/$SLURM_ARRAY_JOB_ID
W=$ROOT/$MODE/$(printf 'a%02d' "$A")
mkdir -p "$W/pics"
cd "$W"
set --
if [ "$NOS6_ALL_PASSES" = 1 ]; then set -- --all-passes; fi
ret=0
"$P" "$B/coupled/cs.py" nos6_held 0 0 2000 12 \
    --patch=nos6,cords --cord-mm=3.2 --cord-cd=1.2 \
    --угол="$A" --полотнищ=96 --шаг=4 --сдвиг=0.02 \
    --V=30 --nu=1.46e-5 --fabric=33.2,28.6 --fabric-relax=0.2 \
    --shape-tol=0.03 --cx-tol=0.003 --attitude="$MODE" \
    --pressure-gradient=gauss --turbulence-length=1.24 "$@" || ret=$?
if [ ! -s out/nos6_held.json ]; then
    echo "No completed passes in $W; inspect the job output and case/log.*" >&2
    exit 1
fi
post=0
"$P" "$B/coupled/nos6_results.py" case "$W" --speed=30 --wake=1 --gif=off --no-summary || post=$?
"$P" "$B/coupled/method_compare.py" "$ROOT" || post=$?
if [ "$ret" != 0 ]; then exit "$ret"; fi
exit "$post"
