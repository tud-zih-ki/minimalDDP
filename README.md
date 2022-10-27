## Minimal template of torch.distributed run with slurm and scorep

1. configure your DDP scripts using the template in minimalDDP.py
2. If you only want to train and do inferencing use slurmTorch.py as it is. If you want to enable scorep-tracing, enable in `slurmTorch.py`:
```
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
```
3a. sbatch minimalrun.sh without scorep

__OR__

3b. sbatch minimalrun_scorep.sh with scorep

