## Minimal template of torch.distributed run with slurm and scorep

1. configure your DDP scripts using the template in minimalDDP.py
2. If you only want to train and do inferencing use slurmTorch.py as it is. If you want to enable *scorep-tracing*, enable in `slurmTorch.py`:
```
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
```
> **scorep-tracing** refers to profiling, event tracing, and online analysis of your code with Score-P [1,2] 

3a. `sbatch minimalrun.sh` without scorep

__OR__

3b. `sbatch minimalrun_scorep.sh` with scorep


[1] https://www.vi-hps.org/projects/score-p \
[2] https://doc.zih.tu-dresden.de/software/scorep/
