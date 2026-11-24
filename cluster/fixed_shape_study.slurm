#!/bin/bash
#SBATCH -J wall_shape
#SBATCH -p compute1
#SBATCH -N 1
#SBATCH -n 72
#SBATCH --array=0-20%2
#SBATCH -t 2-00:00:00
#SBATCH -o %x-%A_%a.out
set -euo pipefail
B=${SLURM_SUBMIT_DIR:?Submit from repository root}
test -f "$B/coupled/fixed_shape_study.py"
module load "${NOS6_OPENFOAM_MODULE:-openfoam/2212}"
P=${NOS6_PYTHON:-python3}
M=$(realpath "${1:?Pass setup/manifest.json}")
ROOT=$B/run/fixed_shape_study/${SLURM_ARRAY_JOB_ID:?}
"$P" "$B/coupled/fixed_shape_study.py" run "$M" "${SLURM_ARRAY_TASK_ID:?}" "$ROOT"
