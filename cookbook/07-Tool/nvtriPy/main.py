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
"""nvtripy: an eager-mode Python frontend that compiles to TensorRT.

Tripy (package `nvtripy`, from NVIDIA/TensorRT-Incubator) lets you write a model as a `tp.Module`,
run it **eagerly** while debugging, then `tp.compile` it into a TensorRT `Executable`.

This file is only the driver. It exists because **`pip install nvtripy` into the cookbook's
environment breaks the cookbook**: nvtripy 0.1.7 depends on `tensorrt-cu12 10.x` and
`mlir-tensorrt ... cuda12.trt109`, which shadow the system TensorRT 11 and downgrade NumPy.
Verified the hard way, see `README.md`. So the real example, `tripy_cases.py`, runs inside a
private virtual environment that this script creates.

+ Steps to run.

```bash
python3 main.py
```
"""

import subprocess
import sys
from pathlib import Path

import tensorrt as trt

current_path = Path(__file__).parent
venv_path = current_path / ".venv"
venv_python = venv_path / "bin" / "python"
PACKAGE_INDEX = "https://nvidia.github.io/TensorRT-Incubator/packages.html"

# Pinned on purpose: an unpinned `pip install nvtripy` would silently change what this example
# demonstrates the day a new release lands. `case_version_matrix` reports when a newer one exists.
NVTRIPY_VERSION = "0.1.7"
# What the pinned version resolved to when this example was last verified. Drift is reported, not
# fatal - the point is that a surprise is visible rather than silent.
TESTED_STACK = {"tensorrt-cu12": "10.16.1.11", "numpy": "1.26.0", "mlir-tensorrt-compiler": "0.1.43+cuda12.trt109"}

def prepare_venv() -> bool:
    """Create the private environment and install nvtripy into it. False if that is not possible."""
    if venv_python.exists():
        print(f"    reusing {venv_path.name}/")
        return True

    print(f"    creating {venv_path.name}/ and installing nvtripy (first run only, needs network)")
    try:
        subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True, capture_output=True)
        subprocess.run([str(venv_path / "bin" / "pip"), "install", "-q", f"nvtripy=={NVTRIPY_VERSION}", "-f", PACKAGE_INDEX], check=True, capture_output=True, timeout=1800)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"    could not install nvtripy ({type(e).__name__}); this example needs network access")
        return False
    return True

def venv_package_version(name: str) -> str:
    """Version of one distribution inside the venv, or '-' if it is not installed there."""
    command = f"import importlib.metadata as m; print(m.version({name!r}))"
    process = subprocess.run([str(venv_python), "-c", command], capture_output=True, text=True)
    return process.stdout.strip() if process.returncode == 0 else "-"

def report_isolation():
    """The point of the venv: two TensorRT versions in one container, neither disturbing the other."""
    cookbook_version = trt.__version__
    command = "import tensorrt, nvtripy; print(tensorrt.__version__, nvtripy.__version__, tensorrt.__file__)"
    process = subprocess.run([str(venv_python), "-c", command], capture_output=True, text=True)
    venv_version, tripy_version, venv_tensorrt_path = process.stdout.split()

    print(f"    cookbook interpreter : TensorRT {cookbook_version}")
    print(f"    {venv_path.name}                 : TensorRT {venv_version} (nvtripy {tripy_version}), from {venv_tensorrt_path}")
    assert venv_version != cookbook_version, "the venv is not actually isolated"
    print("    -> nvtripy brings its own TensorRT; installing it next to the cookbook's would replace TensorRT 11")
    return

def report_version_matrix():
    """Why the venv is not a workaround to be removed later: upstream declares the incompatibility.

    Reading `nvtripy`'s own metadata is the point. It is not that the cookbook happens to have a
    TensorRT nvtripy dislikes - nvtripy states the bound itself, and the package index has no
    artifact that would satisfy this container.
    """
    requirement_list = subprocess.run(
        [str(venv_python), "-c", "import importlib.metadata as m; print('\\n'.join(r for r in m.requires('nvtripy') if 'extra ==' not in r))"],
        capture_output=True,
        text=True,
    ).stdout.split()

    print(f"    nvtripy {NVTRIPY_VERSION} runtime requirements, as declared upstream:")
    for requirement in requirement_list:
        print(f"        {requirement}")

    tensorrt_requirement = next((r for r in requirement_list if r.startswith("tensorrt-cu12")), "")
    print(f"\n    The load-bearing one is `{tensorrt_requirement}`:")
    print(f"        `<11` is upstream refusing TensorRT 11 by declaration, not a resolver accident.")
    print(f"        `cu12` is a CUDA 12 build, in a CUDA {'.'.join(str(x) for x in _cuda_toolkit_version())} container.")
    assert "<11" in tensorrt_requirement, ("nvtripy no longer declares `tensorrt-cu12<11`. If upstream now supports TensorRT 11, this whole venv may be unnecessary - re-check whether nvtripy can be installed alongside the cookbook.")

    print(f"\n    {'component':26s} {'cookbook':>18s} {'.venv':>22s}")
    row_list = [
        ("TensorRT", trt.__version__, venv_package_version("tensorrt-cu12")),
        ("NumPy", _host_version("numpy"), venv_package_version("numpy")),
        ("mlir-tensorrt-compiler", "-", venv_package_version("mlir-tensorrt-compiler")),
        ("nvidia-cuda-runtime", "-", venv_package_version("nvidia-cuda-runtime-cu12")),
    ]
    for name, host, venv in row_list:
        print(f"    {name:26s} {host:>18s} {venv:>22s}")

    drift_list = [(name, expected, venv_package_version(name)) for name, expected in TESTED_STACK.items()]
    drift_list = [row for row in drift_list if row[1] != row[2]]
    for name, expected, actual in drift_list:
        print(f"    [drift] {name}: last verified with {expected}, resolved to {actual}")
    if not drift_list:
        print(f"    the resolved stack matches what this example was last verified against")

    latest_version = _latest_available_version()
    if latest_version is None:
        print(f"\n    could not reach the package index, skipping the \"is the pin stale?\" check")
    elif latest_version == NVTRIPY_VERSION:
        print(f"\n    pinned {NVTRIPY_VERSION}, and that is still the newest release upstream")
    else:
        print(f"\n    [stale pin] pinned {NVTRIPY_VERSION}, but upstream now publishes {latest_version}.")
        print(f"    Bump NVTRIPY_VERSION and re-verify: on a pre-1.0 frontend a minor release can")
        print(f"    move the API, and it is also where a TensorRT 11 build would first appear.")

    print(f"\n    Why a CUDA 12 stack runs here at all: the wheels ship their own CUDA runtime")
    print(f"    (nvidia-cuda-runtime-cu12 above), and the driver is backward compatible, so the")
    print(f"    container's CUDA toolkit is never used by the venv. The GPU driver is shared.")
    print(f"    The package index publishes only `cuda12` builds, up to `trt109` -- there is no")
    print(f"    cuda13 or TensorRT 11 artifact to install, so this is the only possible arrangement.")
    return

def _latest_available_version():
    """Newest `nvtripy` on the package index, or None if it cannot be reached.

    A pin is only honest if something notices when it goes stale, so this is the check that keeps
    `NVTRIPY_VERSION` from quietly rotting. Best-effort by design: no network must not fail a run
    whose venv is already installed.
    """
    try:
        process = subprocess.run(
            [str(venv_path / "bin" / "pip"), "index", "versions", "nvtripy", "-f", PACKAGE_INDEX],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return None
    if process.returncode != 0:
        return None
    # `pip index versions` prints "nvtripy (0.1.7)" first, newest first in the list below it.
    for line in process.stdout.splitlines():
        if line.startswith("Available versions:"):
            return line.split(":", 1)[1].split(",")[0].strip()
    return None

def _host_version(name: str) -> str:
    import importlib.metadata

    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "-"

def _cuda_toolkit_version() -> tuple:
    """CUDA toolkit version of the container, from `nvcc`, or ('?',) if nvcc is absent."""
    process = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
    if process.returncode != 0:
        return ("?", )
    for token in process.stdout.split():
        if token.startswith("V") and token[1:2].isdigit():
            return tuple(token[1:].split(".")[:2])
    return ("?", )

if __name__ == "__main__":
    print(f"{'=' * 30} Start [prepare]")
    if not prepare_venv():
        print("Skipped")
        sys.exit(0)
    report_isolation()
    print(f"{'=' * 30} End   [prepare]")

    print(f"\n{'=' * 30} Start [version_matrix]")
    report_version_matrix()
    print(f"{'=' * 30} End   [version_matrix]")

    # `nvtripy` logs one "WARNING The logger passed into createInferRuntime differs ..." line per
    # engine it builds, which buries the output; drop those lines and keep everything else.
    process = subprocess.run([str(venv_python), str(current_path / "tripy_cases.py")], capture_output=True, text=True, cwd=current_path)
    print(process.stdout, end="")

    error_line_list = [line for line in process.stderr.splitlines() if not line.startswith("WARNING The logger passed into")]
    if error_line_list:
        # Worth showing: the out-of-range shape in `case_dynamic_shapes` is rejected by TensorRT
        # itself, and the message names the profile it failed against.
        print("\nWhat TensorRT wrote to stderr underneath Tripy:")
        for line in error_line_list:
            print(f"    {line}")
    assert process.returncode == 0, "the Tripy cases failed"
