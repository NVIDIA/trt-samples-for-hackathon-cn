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
"""One max-abs-diff is not an accuracy report. Plot every output element instead.

The usual reduced-precision check produces a single number -- `max |a - b| = 0.031` -- and
that number cannot distinguish the two situations you actually care about:

+ the error is **spread** over every element, i.e. the whole tensor is slightly noisier.
  That is what reduced precision is supposed to look like, and more precision is the only
  fix.
+ the error is **driven by a handful of outliers** while the rest of the tensor is exact.
  That is a different bug -- one saturating value, one badly scaled channel, one operation
  that should not have been demoted -- and it is fixable without giving up the speed.

Plotting the absolute error of every element against the value it belongs to separates
them at a glance, and the summary statistics below say the same thing without the picture.

This is a from-scratch reimplementation of the idea behind an internal, proprietary
`accuracy_scatterplot.py`; no code was taken from it.

Reads nothing but the two output sets, so it works with any pair of runs -- here the pair
comes from `polygraphy run --trt` against `--onnxrt`, which is also how the cookbook's
`Run/` example produces its comparison.
"""

import subprocess
from collections import OrderedDict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")  # Save to file; there is no display in a container
import matplotlib.pyplot as plt  # noqa: E402

from polygraphy.json import load_json  # noqa: E402

from tensorrt_cookbook import case_mark, cookbook_path  # noqa: E402

np.random.seed(31193)

output_path = Path(__file__).parent
onnx_file = cookbook_path("00-Data", "model", "model-trained.onnx")
golden_file = output_path / "data-golden.json"
test_file = output_path / "data-test.json"
figure_file = output_path / "result-accuracy_scatterplot.png"

# The tensor is small, so a deliberately harsh comparison is needed to produce any error at
# all: the golden run is ONNX-Runtime in float32, the test run is TensorRT.
INPUT_SHAPE = "x:[8,1,28,28]"

result = OrderedDict()

def run_polygraphy(argument_list: list, save_file: Path) -> None:
    """Run one backend and save its outputs."""
    command = ["polygraphy", "run", str(onnx_file), *argument_list, "--input-shapes", INPUT_SHAPE, "--save-outputs", str(save_file), "--seed", "31193"]
    process = subprocess.run(command, capture_output=True, text=True)
    assert process.returncode == 0, f"{' '.join(command)}\n{process.stderr[-2000:]}"
    return

def load_outputs(path: Path) -> OrderedDict:
    """`--save-outputs` writes a list of iterations; take the first."""
    data = load_json(str(path))
    entry = data[0]
    if not isinstance(entry, dict):  # RunResults is a list of (runner_name, [iteration, ...])
        entry = entry[1][0]
    return OrderedDict((name, np.asarray(value)) for name, value in entry.items())

def summarize(golden: np.ndarray, test: np.ndarray) -> dict:
    """The statistics that tell 'spread' from 'a few outliers' apart."""
    error = np.abs(golden.astype(np.float64) - test.astype(np.float64)).reshape(-1)
    n_nonzero = int(np.count_nonzero(error))
    total = float(error.sum())
    order = np.sort(error)[::-1]
    # What fraction of the *total* error lives in the worst 1% of elements? For evenly
    # spread error this tends to 1%; for outlier-driven error it approaches 100%.
    top_count = max(1, error.size // 100)
    top_share = float(order[:top_count].sum() / total) if total > 0 else 0.0
    return {
        "n_element": int(error.size),
        "n_nonzero": n_nonzero,
        "fraction_wrong": n_nonzero / error.size,
        "max": float(error.max()),
        "mean": float(error.mean()),
        "median": float(np.median(error)),
        "top1pct_share": top_share,
    }

@case_mark
def case_collect() -> None:
    """Produce the two runs to compare."""
    run_polygraphy(["--onnxrt"], golden_file)
    run_polygraphy(["--trt"], test_file)
    golden = load_outputs(golden_file)
    test = load_outputs(test_file)
    common = [name for name in golden if name in test]
    print(f"    golden (onnxruntime) outputs: {[(k, v.shape, v.dtype) for k, v in golden.items()]}")
    print(f"    test   (tensorrt)   outputs: {[(k, v.shape, v.dtype) for k, v in test.items()]}")
    assert common, f"No output name in common: {list(golden)} vs {list(test)}"
    result["golden"], result["test"], result["common"] = golden, test, common
    return

@case_mark
def case_statistics() -> None:
    """The numbers a scatter plot would show, for a log file that has no pictures."""
    print(f"    {'output':<12}{'elements':>10}{'wrong':>10}{'max err':>12}{'mean err':>12}{'top-1% share':>14}  verdict")
    print("    " + "-" * 88)
    for name in result["common"]:
        statistic = summarize(result["golden"][name], result["test"][name])
        # A single element cannot be "spread", so guard the verdict on element count
        verdict = "identical" if statistic["max"] == 0.0 else ("outlier-driven" if statistic["top1pct_share"] > 0.5 and statistic["n_element"] >= 100 else "spread")
        print(f"    {name:<12}{statistic['n_element']:>10}{statistic['n_nonzero']:>10}{statistic['max']:>12.3e}{statistic['mean']:>12.3e}{statistic['top1pct_share']:>13.1%}  {verdict}")
        result.setdefault("statistic", OrderedDict())[name] = statistic
    return

@case_mark
def case_same_max_different_diagnosis() -> None:
    """The demonstration that makes the tool worth having.

    Two synthetic error patterns are built on top of the golden output and given the
    **exact same max absolute error**. A report that quotes only max-abs-diff calls them
    identical. They are not: one is noise everywhere, the other is a single broken element.
    """
    golden = result["golden"][result["common"][0]].astype(np.float64)
    peak = 0.05

    spread = golden + np.random.uniform(-peak, peak, golden.shape)
    spread.reshape(-1)[0] = golden.reshape(-1)[0] + peak  # Pin the max so both agree exactly

    outlier = golden.copy()
    outlier.reshape(-1)[0] = golden.reshape(-1)[0] + peak  # One element, same magnitude

    for name, test in [("spread", spread), ("outlier", outlier)]:
        statistic = summarize(golden, test)
        result.setdefault("synthetic", OrderedDict())[name] = (test, statistic)
        print(f"    {name:<10} max={statistic['max']:.6f}, mean={statistic['mean']:.3e}, "
              f"wrong={statistic['n_nonzero']}/{statistic['n_element']}, top-1% share={statistic['top1pct_share']:.1%}")

    spread_statistic = result["synthetic"]["spread"][1]
    outlier_statistic = result["synthetic"]["outlier"][1]
    assert abs(spread_statistic["max"] - outlier_statistic["max"]) < 1e-12, "The two patterns must share a max, that is the point"
    print(f"    -> identical max ({spread_statistic['max']:.6f}), but the mean differs by "
          f"{spread_statistic['mean'] / max(outlier_statistic['mean'], 1e-30):.0f}x and the top-1% share by "
          f"{outlier_statistic['top1pct_share'] / max(spread_statistic['top1pct_share'], 1e-30):.0f}x")
    assert outlier_statistic["top1pct_share"] > 0.9 > spread_statistic["top1pct_share"], "The share statistic should separate the two"
    return

@case_mark
def case_plot() -> None:
    """Absolute error against the golden value, one point per element."""
    panel_list = [(name, result["golden"][name], result["test"][name]) for name in result["common"]]
    reference = result["golden"][result["common"][0]]
    panel_list += [(f"synthetic {name}", reference, test) for name, (test, _) in result["synthetic"].items()]

    figure, axes_list = plt.subplots(1, len(panel_list), figsize=(5 * len(panel_list), 4.5), squeeze=False)
    for axes, (name, golden_array, test_array) in zip(axes_list[0], panel_list):
        golden = golden_array.reshape(-1).astype(np.float64)
        error = np.abs(golden - test_array.reshape(-1).astype(np.float64))
        axes.scatter(golden, error, s=8, alpha=0.5)
        axes.set_xlabel(f"golden value ({name})")
        axes.set_ylabel("absolute error")
        axes.set_yscale("symlog", linthresh=1e-12)
        axes.set_title(f"{name}\nmax {error.max():.3e}, mean {error.mean():.3e}")
        axes.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(figure_file, dpi=110)
    plt.close(figure)
    print(f"    Wrote {figure_file.name}")
    print("    A horizontal band of points = spread error; a few points far above the rest = outliers.")
    return

def main() -> None:
    case_collect()
    case_statistics()
    case_same_max_different_diagnosis()
    case_plot()
    return

if __name__ == "__main__":
    main()
    print("\nFinish")
