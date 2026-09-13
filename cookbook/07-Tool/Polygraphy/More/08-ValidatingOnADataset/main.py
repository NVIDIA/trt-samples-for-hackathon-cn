# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Validating a model against a labelled dataset, and why `Comparator` is the wrong tool.

The Polygraphy docs say `Comparator` "is not well suited for validating a single
runner with a real dataset ... especially if the dataset is large". Both halves
of that are checked here rather than repeated.

The bigger half is not about size at all: **`Comparator` compares runners to each
other, it has no concept of a label.** "Do TensorRT and onnxruntime agree" and
"is the model right" are different questions, and only the second one needs a
dataset. The size argument is real too, just linear and easy to under-estimate on
a small model -- `case_what_comparator_retains` measures exactly what is kept.

Dataset: `00-Data/data/TestData.npz`, 500 MNIST images with one-hot labels.
"""

import numpy as np
from polygraphy.backend.trt import CreateConfig, EngineFromNetwork, NetworkFromOnnxPath, Profile, TrtRunner
from polygraphy.comparator import Comparator
from polygraphy.logger import G_LOGGER

from tensorrt_cookbook import case_mark, cookbook_path

G_LOGGER.module_severity = G_LOGGER.ERROR

onnx_file = str(cookbook_path("00-Data", "model", "model-trained.onnx"))
N_SAMPLE = 200
BATCH_MAX = 50

dataset = np.load(cookbook_path("00-Data", "data", "TestData.npz"))
image = dataset["data"].astype(np.float32)
# Labels are stored one-hot; the model's `z` output is already a class index.
label = dataset["label"].argmax(axis=1)

engine = None

def feed(index: int) -> dict:
    """One sample as a feed dict."""
    return {"x": image[index:index + 1]}

@case_mark
def case_build_once() -> None:
    """Build the engine once and reuse it for every case below."""
    global engine
    # A profile wide enough for `case_batching_is_the_other_reason`; with the
    # default profile the engine only accepts batch 1 and batching raises
    # `Received incompatible shape`. See `../07-ProfilesAndDynamicShapes/`.
    profile = Profile().add("x", min=(1, 1, 28, 28), opt=(1, 1, 28, 28), max=(BATCH_MAX, 1, 28, 28))
    engine = EngineFromNetwork(NetworkFromOnnxPath(onnx_file), config=CreateConfig(profiles=[profile]))()
    print(f"    dataset: {image.shape} images, labels as class indices, first 8 = {label[:8]}")
    print(f"    using the first {N_SAMPLE} samples")
    return

@case_mark
def case_validate_against_labels() -> None:
    """The thing `Comparator` cannot express: accuracy against ground truth.

    A plain loop over a runner. Nothing is accumulated, the metric is whatever
    the task needs, and the labels never touch Polygraphy -- which is the point,
    because Polygraphy has no notion of them.
    """
    n_correct = 0
    with TrtRunner(engine) as runner:
        for index in range(N_SAMPLE):
            predicted = np.asarray(runner.infer(feed(index))["z"])[0]
            n_correct += int(predicted == label[index])
    print(f"    accuracy: {n_correct}/{N_SAMPLE} = {n_correct / N_SAMPLE:.1%}")
    print("    `Comparator` has no place to put `label` -- it only ever compares runners")
    return

@case_mark
def case_what_comparator_retains() -> None:
    """The size argument, measured instead of asserted.

    `Comparator.run` returns a `RunResults` holding **every** iteration's outputs.
    That is exactly what makes it useful for comparing runners and exactly what
    makes it unsuitable as a dataset loop: the cost is linear in dataset size
    times output size, and on a small model it is invisible.
    """
    results = Comparator.run(
        [TrtRunner(engine, name="trt")],
        data_loader=(feed(index) for index in range(N_SAMPLE)),
    )
    retained = sum(np.asarray(value).nbytes for _, iteration_list in results.items() for iteration in iteration_list for _, value in iteration.items())
    print(f"    Comparator.run kept {len(results['trt'])} IterationResults, {retained} bytes total ({retained // N_SAMPLE} B per iteration)")
    print("    a direct runner loop keeps nothing at all")

    print("    what that means at 10000 iterations:")
    for tag, shape in [("MNIST logits (1, 10)", (1, 10)), ("segmentation (1, 3, 1024, 1024)", (1, 3, 1024, 1024))]:
        megabyte = int(np.prod(shape)) * 4 * 10000 / 1024 ** 2
        print(f"      {tag:<34}: {megabyte:>10.1f} MB retained")
    print("    linear and easy to under-estimate: harmless here, 120 GB on a segmentation model")
    return

@case_mark
def case_batching_is_the_other_reason() -> None:
    """A dataset loop can batch; `Comparator`'s data loader shape is fixed per run.

    Feeding the same 200 samples as 200 batches of 1 against 4 batches of 50
    changes nothing about the answer and a lot about the wall clock -- another
    thing a hand-written loop controls and `Comparator` does not make natural.
    """
    import time

    for batch in [1, 50]:
        n_correct, t0 = 0, time.perf_counter()
        with TrtRunner(engine) as runner:
            for start in range(0, N_SAMPLE, batch):
                stop = min(start + batch, N_SAMPLE)
                predicted = np.asarray(runner.infer({"x": image[start:stop]})["z"])
                n_correct += int((predicted == label[start:stop]).sum())
        second = time.perf_counter() - t0
        print(f"    batch {batch:<3}: {n_correct}/{N_SAMPLE} correct, {second * 1000:7.1f} ms total")
    print("    same accuracy, different throughput -- the loop is yours to tune")
    return

if __name__ == "__main__":
    case_build_once()
    case_validate_against_labels()
    case_what_comparator_retains()
    case_batching_is_the_other_reason()

    print("\nFinish")
