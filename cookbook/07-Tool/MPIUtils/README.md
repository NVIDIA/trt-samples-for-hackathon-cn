# MPI Utils

+ Complete MPI utility example based on `tensorrt_cookbook` wrappers.
+ Covers rank/size, broadcast, allgather, and point-to-point send/recv for both ndarray and Python object.
+ Includes a multi-node style example (`multi_node.py`) with host topology collection, local-rank GPU binding, and host-leader collaboration.
+ Includes a hierarchical aggregation example (`hierarchical.py`) for node-local reduction + cross-node leader aggregation.

+ Steps to run.

```bash
# Single process fallback mode (works without mpirun)
python3 main.py

# Multi-process mode
mpirun -n 2 python3 main.py

# Multi-node / multi-GPU style workflow
mpirun -n 8 python3 multi_node.py

# Hierarchical aggregation workflow
mpirun -n 8 python3 hierarchical.py

# If your cluster uses hostfile
mpirun -n 8 --hostfile hostfile.txt python3 multi_node.py

# Optional: force disable MPI and use fallback path
TRT_COOKBOOK_DISABLE_MPI=1 python3 main.py
TRT_COOKBOOK_DISABLE_MPI=1 python3 multi_node.py
TRT_COOKBOOK_DISABLE_MPI=1 python3 hierarchical.py
```
