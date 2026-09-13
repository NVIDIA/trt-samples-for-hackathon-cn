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
"""A compiled module that tracks weight changes instead of going stale.

`torch_tensorrt.compile` snapshots the weights into an engine. Change the source
model afterwards and the compiled module keeps returning the **old** answer, with
nothing to indicate it -- `case_plain_module_goes_stale` measures the drift.

`MutableTorchTensorRTModule` wraps the model instead of replacing it: it compiles
lazily on the first call, and a later `load_state_dict` refits the existing engine
rather than rebuilding it. That is what makes swapping fine-tuned or LoRA weights
practical.

Only the self-contained half of the upstream example is covered here; the Stable
Diffusion / LoRA pipeline needs `diffusers` and gated downloads.
"""

import tempfile
import time
from pathlib import Path

import torch
import torch_tensorrt as torch_trt
import torchvision.models as models

from tensorrt_cookbook import case_mark

torch.manual_seed(31193)

SHAPE = (1, 3, 224, 224)
work_path = Path(tempfile.mkdtemp(prefix="torch-trt-mutable-"))
data = (torch.rand(*SHAPE).cuda(), )

def build_model(b_randomize: bool = False):
    """A randomly initialised ResNet18; the cache and refit paths ignore weights."""
    model = models.resnet18(weights=None).cuda().eval()
    if b_randomize:
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.copy_(torch.randn_like(parameter) * 0.02)
    return model

@case_mark
def case_plain_module_goes_stale() -> None:
    """The problem: a compiled module does not follow its source model.

    The engine holds a copy of the weights taken at compile time. Mutating the
    original `nn.Module` afterwards changes nothing about the engine, and no
    exception or warning marks the divergence -- the compiled module simply keeps
    answering with weights nobody is using any more.
    """
    model = build_model()
    compiled = torch_trt.compile(model, ir="dynamo", arg_inputs=data, min_block_size=1)
    with torch.no_grad():
        before = compiled(*data)

    with torch.no_grad():  # Replace every weight in the *source* model
        for parameter in model.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.02)
        after, eager = compiled(*data), model(*data)

    print(f"    compiled output changed at all : {not torch.allclose(before, after)}")
    print(f"    still agrees with eager        : {torch.allclose(after, eager, atol=1e-2)}, "
          f"max |eager - TensorRT| = {(after - eager).abs().max().item():.3f}")
    print("    the engine kept the weights it was built with, silently")
    return

@case_mark
def case_mutable_refits() -> None:
    """`MutableTorchTensorRTModule` refits on `load_state_dict`.

    Compilation is lazy: nothing is built until the first call. Afterwards a new
    state dict triggers a refit of the existing engine, which is far cheaper than
    a rebuild and, unlike the case above, actually lands.
    """
    mutable = torch_trt.MutableTorchTensorRTModule(build_model(), immutable_weights=False, min_block_size=1)

    t0 = time.time()
    with torch.no_grad():
        mutable(*data)
    compile_second = time.time() - t0
    print(f"    first call compiles      : {compile_second:6.2f} s")

    other = build_model(b_randomize=True)
    t0 = time.time()
    mutable.load_state_dict(other.state_dict())
    with torch.no_grad():
        output, reference = mutable(*data), other(*data)
    refit_second = time.time() - t0

    difference = (output - reference).abs().max().item()
    print(f"    load_state_dict + refit  : {refit_second:6.2f} s ({compile_second / refit_second:.1f}x cheaper than compiling)")
    print(f"    max |eager - TensorRT|   : {difference:.2e}")
    assert difference < 1e-5, "the refit did not take effect"
    return

@case_mark
def case_save_and_load() -> None:
    """The whole thing pickles, engine included.

    `MutableTorchTensorRTModule.save` / `.load` is a different mechanism from
    `torch_tensorrt.save` (see `../SaveLoad/`): it keeps the mutable wrapper, so
    the reloaded object can still be refitted. The price is size -- the pickle
    carries the engine *and* the PyTorch weights it needs to refit from.
    """
    mutable = torch_trt.MutableTorchTensorRTModule(build_model(b_randomize=True), immutable_weights=False, min_block_size=1)
    with torch.no_grad():
        mutable(*data)
        expected = mutable(*data)

    save_file = work_path / "mutable.pkl"
    torch_trt.MutableTorchTensorRTModule.save(mutable, str(save_file))
    t0 = time.time()
    reloaded = torch_trt.MutableTorchTensorRTModule.load(str(save_file))
    load_second = time.time() - t0

    with torch.no_grad():
        difference = (reloaded(*data) - expected).abs().max().item()
    print(f"    pickle {save_file.stat().st_size / 1024 ** 2:6.1f} MB, loaded in {load_second:.2f} s")
    print(f"    max |before save - after load| = {difference:.2e}")
    assert difference == 0.0, "the reloaded module is not the same engine"
    return

if __name__ == "__main__":
    case_plain_module_goes_stale()
    case_mutable_refits()
    case_save_and_load()

    print("\nFinish")
