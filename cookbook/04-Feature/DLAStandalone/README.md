# DLA Standalone

+ Demonstrates TensorRT DLA standalone build configuration.

+ Steps to run.

```bash
python3 main.py
```

## `DLAWorkspaceAllocationStrategy` (new in TensorRT 11.2/11.3)

`trt.DLAWorkspaceAllocationStrategy` and `IRuntime.dla_workspace_allocation_strategy` exist on this
build (11.3.0.99, present = True) and are absent from the 11.0.0.114 and 11.1.0.106 API
snapshots in [`../../07-Tool/ListAPIs/output/`](../../07-Tool/ListAPIs/output/):

| Value | Meaning |
| --- | --- |
| `DEFAULT` | each DLA module allocates its own workspace |
| `SHARED_STATIC` | modules on the same runtime and the same DLA core share one static pool |

`trtexec` gained the matching `--dlaWorkspaceAllocationStrategy`.

**There is deliberately no case for it here**, and that is not an oversight. `SHARED_STATIC` carries
an explicit warning in the TensorRT documentation - *concurrent execution of engines that share DLA
workspace results in undefined behavior* - so the only example worth writing is one that
demonstrates the sharing and the hazard, and this machine reports `runtime.num_DLA_cores == 0`.
Every DLA path here returns early at "no DLA core available", so such an example could be written
but never verified, and publishing an unverifiable example is worse than publishing none. Revisit on
Orin / DRIVE / Jetson.
