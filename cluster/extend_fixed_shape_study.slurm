#!/bin/bash
#SBATCH -J wall_to2000
#SBATCH -p compute1
#SBATCH -N 1
#SBATCH -n 72
#SBATCH --array=0-20%2
#SBATCH -t 2-00:00:00
#SBATCH -o %x-%A_%a.out
set -euo pipefail
B=${SLURM_SUBMIT_DIR:?Submit from repository root}
module load "${NOS6_OPENFOAM_MODULE:-openfoam/2212}"
P=${NOS6_PYTHON:-python3}
"$P" "$B/cluster/extend_fixed_shape_study.py" run \
    "${1:?Pass extension manifest}" "${SLURM_ARRAY_TASK_ID:?}" "${2:?Pass existing result root}"
