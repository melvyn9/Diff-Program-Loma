# dparallel_rev.py
from mpi4py import MPI
from loma import In, reverse_diff
from math import sin

# Define your function and reverse-mode version
def mysin(x: In[float]) -> float:
    return sin(x)

d_mysin = reverse_diff(mysin)

# Inputs to differentiate
x_all = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

chunk = len(x_all) // size
start = rank * chunk
end = len(x_all) if rank == size - 1 else (rank + 1) * chunk

local_grads = []
for i in range(start, end):
    dy = d_mysin(x_all[i])
    print(f"Rank {rank}: Calling d_mysin(x={x_all[i]:.2f}) => dy = {dy:.6f}")
    local_grads.append(dy)

# Reduce gradient sum (or collect all gradients depending on your goal)
local_sum = sum(local_grads)
global_sum = comm.reduce(local_sum, op=MPI.SUM, root=0)

if rank == 0:
    print(f"Global gradient sum: {global_sum:.6f}")
