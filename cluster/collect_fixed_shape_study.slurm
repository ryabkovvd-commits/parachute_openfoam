#!/bin/bash
#SBATCH -J wall_shape_summary
#SBATCH -p compute1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=2
#SBATCH -t 1:00:00
#SBATCH -o %x-%j.out
set -euo pipefail
B=${SLURM_SUBMIT_DIR:?Submit from repository root}
P=${NOS6_PYTHON:-python3}
"$P" "$B/coupled/fixed_shape_study.py" collect \
    "${1:?Pass calculation root}" --manifest "${2:?Pass setup manifest}"
