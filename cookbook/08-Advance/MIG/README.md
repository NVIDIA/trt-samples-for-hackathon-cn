# MIG (Multi-Instance GPU)

+ Why MIG gets a note rather than an example: from inside the process a slice is just a smaller GPU.

**This directory is a note, not a runnable example, and that is on purpose.**

MIG is host-side configuration, not a TensorRT feature: from inside the process a MIG slice looks
exactly like a smaller GPU, and not one TensorRT API behaves differently because of it. An example
that shows `export CUDA_VISIBLE_DEVICES=MIG-<UUID>` would only be copying the `nvidia-smi` manual.

There is exactly one MIG consequence that belongs in a TensorRT cookbook, and it is written down
below. The measurement that would back it up needs a machine with MIG enabled, which this one is
not — so it is marked as untested rather than dressed up as a result.

## The part that is actually about TensorRT

**Build the engine on the MIG profile you are going to deploy on.**

TensorRT picks its tactics at build time against the GPU it can see *then*: tile sizes, split-k
decompositions and occupancy assumptions all follow from the SM count, and the usable workspace
follows from the memory size. On this machine (**B200 180 GB**, re-measured 2026-09-06) the range is:

| Profile | SM | Memory | Instances |
| ------- | -- | ------ | --------- |
| `7g.180gb` (whole GPU) | 148 | 178.50 GiB | 1 |
| `4g.90gb` | 72 | 89.00 GiB | 1 |
| `3g.90gb` | 70 | 89.00 GiB | 2 |
| `2g.45gb` | 36 | 44.25 GiB | 3 |
| `1g.45gb` | 30 | 44.25 GiB | 4 |
| `1g.23gb` | 18 | 20.50 GiB | 7 |

(from `nvidia-smi mig -lgip`, so these are this GPU's real profiles, not a generic table. The
earlier H100 PCIe 80 GB table had 114 SM / `7g.80gb` down to 14 SM / `1g.10gb`.)

An engine built on the whole GPU and then served on a `1g.23gb` slice was tuned for **148 SM** and
is being run on **18** — the ratio got *worse* on B200, 8.2x against H100's 8.1x, on a bigger
absolute gap. It still loads, still runs, still produces the right numbers — it is only
the tactic choice that is now blind, which is the worst kind of performance bug because nothing
reports it. The same applies to `--memPoolSize=workspace`: a size that is generous on 178 GiB may
not leave room on 20.5 GiB, and tactics that need that workspace quietly drop out of the search.

**Open measurement** (needs a MIG-enabled machine): build one ONNX twice, once on `7g.180gb` and
once on `1g.23gb`, run both on `1g.23gb`, and compare throughput. If the gap is real this note
becomes an example; if the gap is inside the noise, that is equally worth knowing and the note
stays a note.

## What you probably came here for

"How do I run several models on one GPU" is usually the real question, and MIG is only one of the
answers — the one that needs no TensorRT-side change at all:

+ [`../MultiContext/`](../MultiContext/README.md) — several execution contexts on one engine.
+ [`../MultiStream/`](../MultiStream/README.md) — overlapping work on one GPU.
+ [`../MultiOptimizationProfile/`](../MultiOptimizationProfile/README.md) — one engine serving
  several shape ranges.
+ [`../../07-Tool/TritonServerDeploy/`](../../07-Tool/TritonServerDeploy/README.md) — instance
  groups and dynamic batching, i.e. the serving-side answer.

## Setting MIG up, for reference

All of it happens on the **host, as root, with no process running on the GPU** — a container
cannot do any of it. Verified here: `nvidia-smi -i 0 -mig 1` from inside this container fails with
`Insufficient Permissions`.

```bash
sudo nvidia-smi -i 0 -mig 1              # Enable MIG mode (persists across container restarts)
sudo nvidia-smi mig -i 0 -cgi 9,9 -C     # Two 3g.40gb instances, each with a compute instance
nvidia-smi -L                            # Collect the MIG-<UUID> of each instance
```

The slice is then handed to a container **at start time**:

```bash
docker run --gpus '"device=0:0"' ...                 # by index
docker run -e NVIDIA_VISIBLE_DEVICES=MIG-<UUID> ...  # by UUID
```

Two consequences worth knowing:

+ **Partitioning cannot change at run time.** A container can select among the MIG devices it was
  given (`CUDA_VISIBLE_DEVICES=MIG-<UUID>`), but creating, resizing or destroying instances is a
  host operation, and a running process cannot move to a different profile.
+ **There is no P2P between slices** (`P2P: No` in the profile listing), so multi-GPU techniques
  that rely on NVLink or NCCL between devices do not apply between MIG instances of one GPU.
