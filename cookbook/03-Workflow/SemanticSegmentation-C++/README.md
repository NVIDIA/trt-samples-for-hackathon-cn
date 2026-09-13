# Semantic Segmentation (C++ runtime)

+ A vision runtime in C++: read an image, run the engine, write a class map — with the pre- and post-processing that the tensor-in/tensor-out examples leave out.

+ Steps to run.

```bash
make build
python3 export.py   # writes model-segmentation.onnx, data-input.ppm and the reference
./main.exe          # writes data-output.ppm and data-output.raw
python3 check.py    # compares against the PyTorch reference
```

The cookbook's other C++ examples ([`../../01-SimpleDemo/`](../../01-SimpleDemo/README.md),
[`../pyTorch-ONNX-TensorRT/C++/`](../pyTorch-ONNX-TensorRT/README.md)) feed a tensor in and print a
tensor out. A deployed vision service does neither, and the interesting work is on both sides of
`enqueueV3`.

Re-expressed from `quickstart/SemanticSegmentation/tutorial-runtime.cpp` in TensorRT-OSS, which does
this for FCN-ResNet101. That file is already written against the modern API (`setTensorAddress` +
`enqueueV3`), so there is nothing to modernise; what it needs is `samples/common`, a ~200 MB
torchvision download and its own PPM helpers. This version is standalone and, more usefully,
**checks its answer** instead of asking you to look at a picture.

Measured on B200, TensorRT 11.1.0.106, 224x224 RGB, 4 classes.

```txt
Input image: 224x224, 150528 bytes interleaved RGB
Engine built, 2 IO tensors
  input  input (1, 3, 224, 224)
  output output (1, 4, 224, 224)
Pixels per class: [0]=12544 [1]=12544 [2]=12544 [3]=12544
Pixels agreeing    : 50176 / 50176 (100.000%)
```

## The two layout mistakes this example exists to show

Both are silent. The engine runs, the output is the right size, and the answers are wrong.

### Pre-processing: HWC uint8 to CHW float

A PPM (and every camera, and OpenCV) gives interleaved RGB: the three channels of one pixel are
**adjacent**. TensorRT wants planar NCHW: the three channels of one pixel are **one full plane
apart**. The loop that gets this right is the one that iterates the channel outermost:

```cpp
for (int32_t c = 0; c < 3; ++c)
    for (size_t i = 0; i < pixelCount; ++i)
        inputHost[c * pixelCount + i] = (imagePixel[i * 3 + c] / 255.0F - kMean[c]) / kStd[c];
```

Copying straight across produces a picture-shaped tensor whose channels are interleaved garbage.

### Post-processing: argmax across the channel *stride*

The engine emits `[1, C, H, W]` logits, so the `C` values belonging to one pixel are `H * W` floats
apart, not adjacent. Reading them as adjacent is the mirror image of the same mistake.

### And the normalisation constants live in neither file

`mean = [0.485, 0.456, 0.406]`, `std = [0.229, 0.224, 0.225]` are in the training script and in the
C++, and **nowhere in the ONNX**. Nothing checks that the two copies agree; `check.py` is what
turns that into a test.

## Why the model is not FCN-ResNet101

The cookbook does not download at run time, so `export.py` builds a model of the same *shape*:
strided encoder, 1x1 classifier, `F.interpolate` back to full resolution — which is what puts
`Convolution` and `Resize` layers in the engine.

One deliberate choice worth explaining: the classifier head is **not** randomly initialised. An FCN
with random weights argmaxes every pixel to the same class, which makes for a demo that looks
broken. Nearest-reference-colour classification is exactly a 1x1 convolution, because

```txt
argmin_k |x - c_k|^2  ==  argmax_k (2 c_k . x - |c_k|^2)
```

so the head is given weight `2 c_k` and bias `-|c_k|^2` for the four quadrant colours, while the
random encoder/decoder branch is scaled by 0.1 and perturbs that decision rather than replacing it.
The result is a class map a human can check at a glance and a test can compare exactly.

## Not ported: the EfficientDet notebook

`demo/EfficientDet/notebooks/EfficientDet-TensorRT8.ipynb` was the other half of this candidate. It
is **TensorRT-8 era**, its README requires building the TensorRT-OSS docker container and launching
Jupyter inside it, and the model comes from an external download. A notebook that cannot run in this
container, against this TensorRT, is exactly the "unverifiable example" the cookbook declines to
ship. The reusable part of it — ONNX-GS surgery around a detection head — is covered by
[`../../07-Tool/OnnxGraphSurgeon/14_fold_exporter_subgraphs.py`](../../07-Tool/OnnxGraphSurgeon/README.md)
and [`../../04-Feature/OutputAllocator/`](../../04-Feature/OutputAllocator/README.md).

## Related

+ [`../../01-SimpleDemo/`](../../01-SimpleDemo/README.md) — the minimal C++ runtime, no image I/O.
+ [`../../04-Feature/OutputAllocator/`](../../04-Feature/OutputAllocator/README.md) — the detection
  counterpart, where the output size is not known in advance.
+ [`../../04-Feature/DataFormat/`](../../04-Feature/DataFormat/README.md) — what TensorRT's tensor
  formats mean beyond plain NCHW.
