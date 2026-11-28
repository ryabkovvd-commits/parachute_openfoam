"""Use the Slurm worker's inherited CPU mask without launching onto other nodes."""
line("CORES = ", 'CORES = int(os.environ["NOS6_CASE_RANKS"])')
sub("mpirun -np {CORES} simpleFoam",
    "mpirun --host localhost:{CORES} --bind-to none -np {CORES} simpleFoam", count=2)
