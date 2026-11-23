#!/bin/bash
#SBATCH -J nos6balance
#SBATCH -p compute1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -t 02:00:00
#SBATCH -o momentum-balance-%j.out
set -eo pipefail
B=${SLURM_SUBMIT_DIR:?Submit from the repository root}
ROOT=${1:?Pass run/permeability_models/8996}
module load "${NOS6_OPENFOAM_MODULE:-openfoam/2212}"
P=${NOS6_PYTHON:-python3}
cd "$B"
mkdir -p run
wmake "$B/tools/momentumAudit" > "run/momentum-build-${SLURM_JOB_ID}.log" 2>&1 || {
    tail -n 30 "run/momentum-build-${SLURM_JOB_ID}.log"
    exit 1
}
if [ -n "${SLURM_TMPDIR:-}" ]; then export TMPDIR="$SLURM_TMPDIR"; fi
"$P" "$B/coupled/momentum_balance.py" "$ROOT" --binary "$FOAM_USER_APPBIN/nos6MomentumAudit"
