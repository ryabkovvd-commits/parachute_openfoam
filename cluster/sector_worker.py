"""Enter the local MPI driver within one exclusive Slurm CPU mask."""
import os
import socket
import sys


if __name__ == "__main__":
    cpus = sorted(os.sched_getaffinity(0))
    if os.environ.get("SLURM_CPUS_PER_TASK") != "36" or len(cpus) != 36:
        sys.exit(f"Expected exactly 36 usable CPUs on one node, got {cpus}")
    print(f"Slurm step {os.environ.get('SLURM_STEP_ID')} node {socket.gethostname()} CPUs {cpus}",
          flush=True)
    # Slurm owns the worker mask. Keep the nested MPI launcher local.
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("SLURM_", "PMI_", "PMIX_"))}
    env.update(NOS6_CASE_RANKS="36", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    os.execve(sys.executable, [sys.executable, "-B", "-u", *sys.argv[1:]], env)
