# 00-Data

+ Prepare dataset and models needed in this cookbook

## Get MNIST dataset

+ [Baidu-Netdisk](https://pan.baidu.com/s/14HNCFbySLXndumicFPD-Ww?pwd=gpq2)
+ [HuggingFace](https://huggingface.co/datasets/ylecun/mnist)
+ [Kaggle](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)
+ [LeCun](http://yann.lecun.com/exdb/mnist/) (invalid)
+ [GoogleAPIs](https://storage.googleapis.com/cvdf-datasets/mnist/) (invalid)

### From Baidu-Netdisk

+ Download the dataset (4 .gz files) and save them as `<PathToCookbook>/00-Data/data-gz/*.gz`

```bash
cd <PathToCookbook>/00-Data
python3 extract-data-gz.py
python3 get-data.py
```

### From HuggingFace

```bash
cd <PathToCookbook>/00-Data/data-hf
git clone https://huggingface.co/datasets/ylecun/mnist
cd ..
python3 extract-data-hf.py
python3 get-data.py
```

### From Kaggle

+ Download the dataset (1 .zip files) and save it as `<PathToCookbook>/00-Data/data-kg/archive.zip`

```bash
cd <PathToCookbook>/00-Data
python3 extract-data-kg.py
python3 get-data.py
```

### Output

+ By default, 6000 train images and 500 test images are generated and used for cookbook's data.
+ `data/test/*.jpg`: data for test
+ `data/train/*.jpg`: data for training
+ `data/CalibrationData.npy`: example calibration input data (deprecated)
+ `data/InferenceData.npy`: example inference input data
+ `data/TestData.npz`: example test data for training
+ `data/TrainData.npz`: example train data for training

## Build models

```bash
python3 get-model-part1.py  # models created by pytorch
python3 get-model-part2.py  # models created by ONNX
```

`model-trained-sparsity.onnx` and `model-trained-int8-qat.onnx` are produced by
`case_sparsity_modelopt` and `case_int8qat_modelopt`, both using NVIDIA ModelOptimizer
(`nvidia-modelopt[onnx]`, already in `requirements.txt`). Without it those two models print `[SKIP]`
and the script carries on; it no longer aborts, which used to leave every model after it silently
stale.

Each file has a **deprecated** producer kept alongside it, defined but never called:

| Deprecated | Replaced by | Why |
| :--------- | :---------- | :-- |
| `case_sparsity_pytorch_apex` | `case_sparsity_modelopt` | `apex.contrib.sparsity` (ASP) ships only inside the NGC container and reports version `0.1`, so it cannot be reproduced from `requirements.txt`. It also needs a monkey patch: apex's `exhaustive_search.py` calls `time.*` four times without importing `time`. |
| `case_int8qat_pytorch_quantization` | `case_int8qat_modelopt` | `pytorch_quantization` is retired upstream, and its current release ships only an sdist whose build fails with `RuntimeError: Bad params`. |

They are kept because the contrast is instructive — ModelOptimizer replaces a page of manual
calibrator or optimizer plumbing with one call — and because a claimed equivalence is easier to
trust when the thing being replaced is still there to compare against.

Verified on 2026-09-09 that each pair produces the same graph, and that TensorRT 11.0.0.114 builds
both: the QAT files are node-for-node identical (28 nodes, 8 Q/DQ pairs), and the sparsity files
match layer for layer (12 nodes; `conv2.weight` and `gemm1.weight` at exactly 50% zeros with every
group of four holding at most two non-zeros, `conv1` skipped for having a single input channel and
`gemm2` left dense). To run the deprecated QAT one, the version pin is required:

```bash
pip install pytorch-quantization==2.1.3
```

Every file below has at least one consumer; the "used by" column is what keeps this directory from
accumulating models nobody opens. If you add a model here, add its consumer too — and if you delete
the last consumer, delete the model.

+ Output of `get-model-part1.py` (MNIST classifier trained with pyTorch):

| File                                        | Content                                                         | Used by                                                                                                   |
| :------------------------------------------ | :-------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------- |
| `model/model-trained.pth`                   | the trained network as a pyTorch checkpoint                     | `06-DLFrameworkTRT/Torch-TensorRT`                                                                        |
| `model/model-trained.onnx`                  | the trained network exported to ONNX — the cookbook's workhorse | ~56 examples                                                                                              |
| `model/model-trained.npz`                   | external weight file, as a numpy archive                        | `03-Workflow/pyTorch-TensorRT` (python and C++), `04-Feature/Refit`, `tensorrt_cookbook/utils_network.py` |
| `model/model-untrained.onnx`                | the same graph with random weights                              | `04-Feature/Refit` (the engine that gets refitted)                                                        |
| `model/model-trained-no-weight.onnx`        | `model-trained.onnx` with the weights stripped out              | `04-Feature/Refit`, `07-Tool/OnnxWeightProcess`                                                           |
| `model/model-trained-no-weight.onnx.weight` | its external weight file, in ONNX format                        | (loaded automatically with the file above)                                                                |
| `model/model-trained-sparsity.onnx`         | the same network trained with 2:4 structured sparsity           | `04-Feature/Sparsity`, `07-Tool/Polygraphy/Inspect`                                                       |
| `model/model-trained-int8-qat.onnx`         | the same network trained with INT8 QAT (carries Q/DQ pairs)     | `03-Workflow/pyTorch-ONNX-TensorRT`, `07-Tool/Polygraphy/More/06-Int8IsNowExplicit`, `07-Tool/trex`       |
| `model/model-for.onnx`                      | a `Loop` whose body contains an `If` — i.e. nested subgraphs    | `02-API/ONNXParser` (`case_subgraph`)                                                                     |

+ Output of `get-model-part2.py` (small hand-built graphs, created directly with ONNX / onnx-graphsurgeon):

| File                          | Content                                         | Used by                                                                                  |
| :---------------------------- | :---------------------------------------------- | :--------------------------------------------------------------------------------------- |
| `model/model-addscalar.onnx`  | a single custom `AddScalar` operator            | `05-Plugin/ONNXParserWithPlugin`, `07-Tool/trtexec`, three `07-Tool/Polygraphy` examples |
| `model/model-half-mnist.onnx` | the first half of `model-trained.onnx`          | `07-Tool/Polygraphy/Convert`                                                             |
| `model/model-invalid.onnx`    | contains a divide-by-zero                       | `07-Tool/Polygraphy/Run`                                                                 |
| `model/model-labeled.onnx`    | carries named (labelled) dimensions             | `04-Feature/LabeledDimension`                                                            |
| `model/model-loop.onnx`       | a `Loop` and nothing else - no `If` in the body | `07-Tool/EnginePrinter` (the control-flow case `export_engine_as_onnx` rejects)           |
| `model/model-redundant.onnx`  | redundant shape operators, for constant folding | `07-Tool/Polygraphy/Surgeon`, `07-Tool/Polygraphy/API`                                   |
| `model/model-reshape.onnx`    | a custom `MyReshape` operator                   | `05-Plugin/ShapeInputTensor`                                                             |
| `model/model-unknown.onnx`    | an operator TensorRT does not know              | `02-API/ONNXParser`, four `07-Tool/Polygraphy` examples                                  |

## Download models from HuggingFace

**No example downloads anything while it runs.** Every model an example opens is already a local
file under `00-Data/`, and the downloads are separate one-off steps, listed here. An example that
fetches half a gigabyte on first use cannot be run offline, cannot be run reproducibly, and turns a
network outage into a confusing failure in the middle of a case.

Two models need a download beyond the ones above:

```bash
cd <PathToCookbook>/00-Data
python3 get-model-gpt2.py     # ~528 MB -> model/gpt2/

# ~1.6 GB -> model/model-large.onnx, a model big enough to make build time and
# weight streaming visible. Any large ONNX file will do; this is just the one measured.
wget -O model/model-large.onnx \
  https://huggingface.co/openai-community/gpt2-medium/resolve/main/onnx/decoder_model.onnx
```

Note the URL says `resolve`, not `blob` — `blob` serves the HTML preview page, so `wget` would
happily write a few kilobytes of HTML into a file named `.onnx` and every consumer would then fail
with a confusing parse error.

+ Output: `model/gpt2/` — `config.json`, `model.safetensors`, and the tokenizer files.
+ Used by [`03-Workflow/pyTorch-KVCache-ONNX-TensorRT`](../03-Workflow/pyTorch-KVCache-ONNX-TensorRT/README.md),
  which prints this command and skips if the directory is absent.
+ Only the PyTorch weights are fetched. The upstream repository also carries TensorFlow, Flax, Rust
  and ONNX copies of the same parameters, which would roughly quadruple the download for nothing.
+ Behind a proxy that breaks HuggingFace's Xet transfer protocol (a `404` on `.../xet-read-token/...`),
  set `HF_HUB_DISABLE_XET=1` to fall back to plain HTTPS.
