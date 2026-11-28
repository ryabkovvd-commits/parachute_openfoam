#!/bin/bash
#SBATCH -J nos6opt
#SBATCH -p compute1
#SBATCH --nodes=2
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=36
#SBATCH --threads-per-core=1
#SBATCH -t 2-00:00:00
#SBATCH -o %x-%j.out
set -e
root=${SLURM_SUBMIT_DIR:?Submit this job from the repository root}
cd "$root"
test -f coupled/cs.py
if [[ "${SLURM_JOB_NUM_NODES:?}" != 2 || "${SLURM_NTASKS:?}" != 4 || "${SLURM_CPUS_PER_TASK:?}" != 36 ]]; then
    echo 'Expected two nodes with two 36-CPU slots per node' >&2
    exit 2
fi
module load "${NOS6_OPENFOAM_MODULE:-openfoam/2212}"
python=${NOS6_PYTHON:-python3}
work="$root/run/sector_optimize/${SLURM_JOB_ID}"
exec "$python" -B -u cluster/sector_optimize.py --work-root "$work" --slurm-steps "$@"
