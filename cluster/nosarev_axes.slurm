#!/bin/bash
#SBATCH -J nosaxes
#SBATCH -p compute1
#SBATCH -N 1
#SBATCH -n 72
#SBATCH -t 2-00:00:00
#SBATCH -o %x-%j.out
set -eo pipefail
B=$SLURM_SUBMIT_DIR
P=python3
if [ -n "$NOS6_PYTHON" ]; then P=$NOS6_PYTHON; fi
A=30
if [ -n "$1" ]; then A=$1; fi
ITERS=2000
if [ -n "$NOS6_CFD_ITERS" ]; then ITERS=$NOS6_CFD_ITERS; fi
MESH=1.25
if [ -n "$NOS6_MESH" ]; then MESH=$NOS6_MESH; fi
ATTITUDE=inner
if [ -n "$NOS6_ATTITUDE" ]; then ATTITUDE=$NOS6_ATTITUDE; fi
module load openfoam/2212
W=$B/run/nosarev_axes/$SLURM_JOB_ID/$ATTITUDE/$(printf 'a%s' "$A")
if [ -e "$W" ]; then echo "Refusing to overwrite $W" >&2; exit 2; fi
mkdir -p "$W"
ret=0
if [ -n "$2" ]; then
    SOURCE=$(realpath "$2")
    "$P" "$B/coupled/nosarev_project.py" prepare "$SOURCE" "$W" "$A"
    transformPoints -case "$W/input_mesh_case" -rotate-angle '((1 -1 -1) 120)' > "$W/log.rotateMesh" 2>&1
    "$P" "$B/coupled/nosarev_project.py" run-fixed "$W" "$ITERS" || ret=$?
else
    cd "$W"
    "$P" "$B/coupled/cs.py" nos6_held 0 0 "$ITERS" 12 \
        --patch=nos6,cords,nosarev_axes --угол="$A" --полотнищ=96 --шаг=4 \
        --сдвиг=0.02 --V=30 --nu=1.46e-5 --fabric=33.2,28.6 --fabric-relax=0.2 \
        --cord-mm=3.2 --cord-cd=1.2 --shape-tol=0.03 --cx-tol=0.003 \
        --pressure-gradient=gauss --turbulence-length=1.24 --attitude="$ATTITUDE" --mesh="$MESH" || ret=$?
fi
if [ -s "$W/out/nos6_held.json" ]; then
    "$P" "$B/coupled/nos6_results.py" case "$W" --speed=30 --wake=1 --gif=off --no-summary || { code=$?; if [ "$ret" -eq 0 ]; then ret=$code; fi; }
    "$P" "$B/coupled/nosarev_project.py" diagram "$W" || { code=$?; if [ "$ret" -eq 0 ]; then ret=$code; fi; }
    if [ -f "$W/coordinate_control.json" ]; then
        "$P" "$B/coupled/nosarev_project.py" compare "$W" || { code=$?; if [ "$ret" -eq 0 ]; then ret=$code; fi; }
    fi
fi
exit "$ret"
