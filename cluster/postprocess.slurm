#!/bin/bash
#SBATCH -J nos6post
#SBATCH -p compute1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH -t 1-00:00:00
#SBATCH -o postprocess-%j.out
set -euo pipefail
B=${SLURM_SUBMIT_DIR:?Submit from the repository root}
ROOT=${1:?Pass the completed array directory, e.g. run/nos6wall/JOB}
shift
module load "${NOS6_OPENFOAM_MODULE:-openfoam/2212}"
P=${NOS6_PYTHON:-python3}
if [ -n "${SLURM_TMPDIR:-}" ]; then export TMPDIR="$SLURM_TMPDIR"; fi
cd "$B"
"$P" "$B/coupled/nos6_results.py" batch "$ROOT" \
    --speed=30 --wake="${NOS6_WAKE:-1}" --gif="${NOS6_GIF:-off}" \
    --jobs="${SLURM_CPUS_PER_TASK:-1}" "$@"
