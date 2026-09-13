# Validating on a dataset

+ Check a model against labelled data, and see why `Comparator` is the wrong tool for it.

+ Steps to run.

```bash
python3 main.py
```

Dataset: `00-Data/data/TestData.npz`, 500 MNIST images with one-hot labels.

## The reason that has nothing to do with size

The Polygraphy docs say `Comparator` "is not well suited for validating a single
runner with a real dataset ... especially if the dataset is large". The size
argument is real, but it is the smaller half.

**`Comparator` compares runners to each other. It has no concept of a label.**

```
accuracy: 194/200 = 97.0%
```

That number comes from a plain loop over a `TrtRunner`, with the labels never
touching Polygraphy — because there is nowhere to put them. "Do TensorRT and
onnxruntime agree" and "is the model right" are different questions;
[`../02-ComparingBackends/`](../02-ComparingBackends/README.md) answers the first,
this one answers the second.

## The size argument, measured

```
Comparator.run kept 200 IterationResults, 9600 bytes total (48 B per iteration)
a direct runner loop keeps nothing at all
```

48 bytes per iteration is 10 float32 logits plus one int64 class. `RunResults`
holds **every** iteration's outputs — that is what makes it useful for comparing
runners and unusable as a dataset loop. The cost is linear in dataset size times
output size, which is invisible here and is not always:

```
at 10000 iterations, MNIST logits (1, 10)              :        0.4 MB retained
at 10000 iterations, segmentation (1, 3, 1024, 1024)   :   120000.0 MB retained
```

120 GB. The mechanism is the same in both rows; only the output size changed.

## The other reason: you control the batching

```
batch 1  : 194/200 correct,   135.9 ms total
batch 50 : 194/200 correct,    11.6 ms total
```

Same accuracy, **11.7x** the throughput. A hand-written loop decides the batch
size; `Comparator`'s data loader does not make that natural.

Note the engine needs an optimization profile wide enough for the batch — with
the default profile it accepts batch 1 only and raises
`Received incompatible shape`. See
[`../07-ProfilesAndDynamicShapes/`](../07-ProfilesAndDynamicShapes/README.md).

## Rule

| question | tool |
| --- | --- |
| do two backends agree? | `Comparator` + `CompareFunc` |
| is the model accurate? | a loop over one runner |
| does it still rank classes the same after quantization? | `PostprocessFunc.top_k` + `CompareFunc.indices` |
