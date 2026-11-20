#!/bin/bash
#SBATCH -J nos6wall
#SBATCH -p compute1
#SBATCH -N 1
#SBATCH -n 36
#SBATCH --array=0-9%2
#SBATCH -t 2-00:00:00
#SBATCH -o %x-%A_%a.out
set -e
B=${SLURM_SUBMIT_DIR:?Submit this job from the repository root}
if [ ! -f "$B/coupled/cs.py" ]; then
    echo 'Submit this job from the repository root' >&2
    exit 2
fi
module load "${NOS6_OPENFOAM_MODULE:-openfoam/2212}"
P=${NOS6_PYTHON:-python3}
C=${2:-24}
if [[ ! $C =~ ^[1-9][0-9]*$ ]]; then
    echo 'Cord count must be a positive integer' >&2
    exit 2
fi
S=$(( (96 + C - 1) / C ))
if (( S < 2 )); then S=2; fi
G=$(( C * S ))
A=$((5 * SLURM_ARRAY_TASK_ID))
W=$B/run/nos6wall/${NOS6_SERIES_ID:-$SLURM_ARRAY_JOB_ID}/$(printf 'a%02d' "$A")
mkdir -p "$W/pics"
cd "$W"
opts=(--patch=nos6)
if [ -n "${1:-}" ]; then opts=(--patch=nos6,cords --cord-mm="$1"); fi
ret=0
"$P" "$B/coupled/cs.py" nos6_held 0 0 2000 12 \
    "${opts[@]}" --угол="$A" --полотнищ="$G" --шаг="$S" --сдвиг=0.02 \
    --V=30 --nu=1.46e-5 || ret=$?
if [ ! -s out/nos6_held.json ]; then
    echo "No completed CFD passes in $W. See the job output and case/log.*" >&2
    exit 1
fi
post_opts=(--speed=30 --wake="${NOS6_WAKE:-1}" --gif="${NOS6_GIF:-off}" --jobs="${SLURM_CPUS_PER_TASK:-1}")
post_ret=0
"$P" "$B/coupled/nos6_results.py" case "$W" "${post_opts[@]}" || post_ret=$?
if (( ret != 0 )); then exit "$ret"; fi
exit "$post_ret"
