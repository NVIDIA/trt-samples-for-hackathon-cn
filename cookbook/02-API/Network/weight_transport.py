# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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
"""The `.wts` convention: move trained weights into a hand-built TensorRT network without ONNX.

The format comes from wang-xinyu/tensorrtx. It is the canonical answer to "I have a state_dict and
I want to call add_convolution_nd myself", and it is worth knowing even if you never use the file
format, because the failure modes it exposes are the ones every hand-built network hits.
"""

import struct
import time
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch
import torch.nn as nn

from tensorrt_cookbook import TRTWrapperV1, case_mark

WTS_FILE = Path("model.wts")
NPZ_FILE = Path("model.npz")
SHAPE = (4, 1, 28, 28)

class Net(nn.Module):
    """A small CNN, and one buffer that is not a parameter."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.fc = nn.Linear(16 * 7 * 7, 10, bias=False)

    def forward(self, x):
        x = torch.max_pool2d(torch.relu(self.conv1(x)), 2)
        x = torch.max_pool2d(torch.relu(self.conv2(x)), 2)
        return self.fc(x.reshape(x.shape[0], -1))

def make_model() -> nn.Module:
    torch.manual_seed(31193)
    model = Net().eval()
    return model

# ------------------------------------------------------------------------------------------------
# The format itself
# ------------------------------------------------------------------------------------------------

def write_wts(state_dict, path: Path) -> None:
    """Line 1 is the blob count. Every other line is `name count hex0 hex1 ...`.

    Each float is packed big-endian and hexlified, so the file is pure ASCII and byte-order
    explicit -- there is no endianness to get wrong when it moves between machines.
    """
    with open(path, "w") as f:
        f.write(f"{len(state_dict)}\n")
        for name, tensor in state_dict.items():
            flat = tensor.reshape(-1).cpu().numpy()
            f.write(f"{name} {len(flat)}")
            for value in flat:
                f.write(" " + struct.pack(">f", float(value)).hex())
            f.write("\n")
    return

def read_wts(path: Path) -> dict:
    """Parse it back. Note the returned arrays are flat: `.wts` carries no shape at all."""
    lines = [line.strip() for line in open(path)]
    count = int(lines[0])
    assert count == len(lines) - 1, f"header says {count} blobs, file has {len(lines) - 1}"
    weight_map = {}
    for line in lines[1:]:
        parts = line.split(" ")
        name, n_value = parts[0], int(parts[1])
        assert n_value + 2 == len(parts), f"{name}: header says {n_value} values, line has {len(parts) - 2}"
        weight_map[name] = np.array([struct.unpack(">f", bytes.fromhex(x))[0] for x in parts[2:]], dtype=np.float32)
    return weight_map

@case_mark
def case_round_trip():
    """Write, read back, and check every bit."""
    model = make_model()
    state_dict = model.state_dict()
    write_wts(state_dict, WTS_FILE)
    weight_map = read_wts(WTS_FILE)

    print(f"    {len(state_dict)} blobs, file is {WTS_FILE.stat().st_size:,} bytes of ASCII")
    print("    name              torch shape          values   flat in .wts   bit-exact")
    print("    " + "-" * 76)
    for name, tensor in state_dict.items():
        reference = tensor.reshape(-1).cpu().numpy()
        restored = weight_map[name]
        exact = np.array_equal(reference, restored)
        print(f"    {name:<17} {str(tuple(tensor.shape)):<20} {reference.size:>6}   {restored.shape[0]:>10}   {exact}")
        assert exact, f"{name} did not survive the round trip"
    print("\n    Big-endian IEEE-754 hex is a lossless representation of float32, so this is not")
    print("    'close enough' -- every value comes back bit for bit. What does NOT come back is")
    print("    the shape: `.wts` stores a flat count only, so the consumer has to know the")
    print("    layout of every blob. That is fine when the consumer is the network you are")
    print("    hand-writing anyway, and a trap the moment anyone else reads the file.")
    return

@case_mark
def case_cost_of_ascii():
    """What the text format costs, against the obvious binary alternative."""
    model = make_model()
    state_dict = model.state_dict()
    n_element = sum(t.numel() for t in state_dict.values())

    write_wts(state_dict, WTS_FILE)
    np.savez(NPZ_FILE, **{k: v.cpu().numpy() for k, v in state_dict.items()})

    t0 = time.perf_counter()
    for _ in range(5):
        read_wts(WTS_FILE)
    t_wts = (time.perf_counter() - t0) / 5
    t0 = time.perf_counter()
    for _ in range(5):
        dict(np.load(NPZ_FILE))
    t_npz = (time.perf_counter() - t0) / 5

    raw = n_element * 4
    print(f"    {n_element:,} float32 values = {raw:,} bytes raw")
    print("    format        size        vs raw   load time   keeps shape")
    print("    " + "-" * 66)
    print(f"    .wts (ASCII)  {WTS_FILE.stat().st_size:>9,}   {WTS_FILE.stat().st_size/raw:>5.2f}x   {t_wts*1e3:>7.1f} ms   no")
    print(f"    .npz          {NPZ_FILE.stat().st_size:>9,}   {NPZ_FILE.stat().st_size/raw:>5.2f}x   {t_npz*1e3:>7.1f} ms   yes")
    print(f"\n    The text format costs {WTS_FILE.stat().st_size/NPZ_FILE.stat().st_size:.1f}x the size and {t_wts/t_npz:.0f}x the load time, and loses the shapes.")
    print("    It buys one thing: the file is diffable, greppable and survives being pasted into")
    print("    a bug report. For a 100 MB model that trade is a bad one -- reach for .npz or")
    print("    safetensors and keep only the idea: ship weights beside the plan, named, so a")
    print("    hand-built network can look them up.")
    return

# ------------------------------------------------------------------------------------------------
# Buffers that are not parameters
# ------------------------------------------------------------------------------------------------

@case_mark
def case_register_buffer():
    """Fold a derived tensor into the state_dict so it travels with the weights.

    This is the trick `yolov5/gen_wts.py` uses for the anchor grid: a tensor that is *computed*
    from other tensors at export time, but that the hand-built network wants precomputed.
    """
    model = make_model()
    print(f"    state_dict before: {list(model.state_dict().keys())}")

    # A quantity derived from the weights, that the TensorRT network would rather not recompute.
    channel_scale = model.conv1.weight.detach().reshape(8, -1).norm(dim=1)
    model.register_buffer("conv1.channel_scale".replace(".", "_"), channel_scale)
    print(f"    state_dict after : {list(model.state_dict().keys())}")

    write_wts(model.state_dict(), WTS_FILE)
    weight_map = read_wts(WTS_FILE)
    assert "conv1_channel_scale" in weight_map
    assert np.allclose(weight_map["conv1_channel_scale"], channel_scale.numpy())
    print(f"\n    conv1_channel_scale = {np.array2string(weight_map['conv1_channel_scale'], precision=4)}")
    print("    register_buffer is the whole mechanism: state_dict() serializes buffers as well as")
    print("    parameters, so anything registered as a buffer is exported for free. Use it when a")
    print("    value is cheap to compute in Python and awkward to express as TensorRT layers.")
    return

# ------------------------------------------------------------------------------------------------
# Building the network from the file
# ------------------------------------------------------------------------------------------------

def build_by_hand(weight_map, data, *, tf32: bool):
    """A TensorRT network built by hand, from named weights, with no ONNX anywhere."""
    tw = TRTWrapperV1()
    network = tw.network
    if not tf32:
        tw.builder_config.clear_flag(trt.BuilderFlag.TF32)
    input_tensor = network.add_input("x", trt.float32, SHAPE)

    # `.wts` has no shapes, so every blob is reshaped by hand here. Keep a reference to each
    # array: trt.Weights wraps the buffer, it does not own or copy it.
    keep_alive = []

    def weights(name, shape):
        array = np.ascontiguousarray(weight_map[name].reshape(shape))
        keep_alive.append(array)
        return trt.Weights(array)

    conv1 = network.add_convolution_nd(input_tensor, 8, [3, 3], weights("conv1.weight", (8, 1, 3, 3)), weights("conv1.bias", (8, )))
    conv1.padding_nd = [1, 1]
    relu1 = network.add_activation(conv1.get_output(0), trt.ActivationType.RELU)
    pool1 = network.add_pooling_nd(relu1.get_output(0), trt.PoolingType.MAX, [2, 2])
    pool1.stride_nd = [2, 2]
    conv2 = network.add_convolution_nd(pool1.get_output(0), 16, [3, 3], weights("conv2.weight", (16, 8, 3, 3)), weights("conv2.bias", (16, )))
    conv2.padding_nd = [1, 1]
    relu2 = network.add_activation(conv2.get_output(0), trt.ActivationType.RELU)
    pool2 = network.add_pooling_nd(relu2.get_output(0), trt.PoolingType.MAX, [2, 2])
    pool2.stride_nd = [2, 2]
    shuffle = network.add_shuffle(pool2.get_output(0))
    shuffle.reshape_dims = [SHAPE[0], 16 * 7 * 7]
    # torch.nn.Linear stores (out_features, in_features), so the matmul transposes the second operand.
    fc_weight = network.add_constant([10, 16 * 7 * 7], weights("fc.weight", (10, 16 * 7 * 7)))
    matmul = network.add_matrix_multiply(shuffle.get_output(0), trt.MatrixOperation.NONE, fc_weight.get_output(0), trt.MatrixOperation.TRANSPOSE)

    tw.build([matmul.get_output(0)])
    tw.setup({"x": data})
    tw.infer(b_print_io=False)
    return network.num_layers, tw.buffer[network.get_output(0).name][0]

@case_mark
def case_build_from_weights():
    """The payoff, and the reason a correct hand-built network still does not match torch."""
    model = make_model()
    write_wts(model.state_dict(), WTS_FILE)
    weight_map = read_wts(WTS_FILE)

    data = torch.randn(SHAPE, dtype=torch.float32)
    with torch.no_grad():
        reference = model(data).numpy()

    result = {}
    for tf32 in [True, False]:
        n_layer, got = build_by_hand(weight_map, data.numpy(), tf32=tf32)
        result[tf32] = np.abs(got - reference).max()
        print(f"    TF32 {str(tf32):<5} -> max |TensorRT - torch| = {result[tf32]:.3e}    ({n_layer} layers)")

    assert result[False] < 1e-6 < result[True], result
    print(f"\n    Same weights, same graph, {result[True]/result[False]:.0f}x difference. **The default builder config")
    print("    enables TF32**, which rounds the mantissa to 10 bits inside the MatMul. Nothing about")
    print("    the weight transport is wrong -- with TF32 cleared the hand-built network agrees with")
    print("    torch to 1e-07, which is float32 accumulation order and nothing more.")
    print("    This is the first thing to check when a hand-built network 'does not match', and it")
    print("    is easy to misread as a transposed blob or a bad reshape.")
    print("\n    Two things had to be supplied by hand that ONNX would have carried: every blob's")
    print("    shape, and the fact that nn.Linear stores its weight transposed relative to what")
    print("    add_matrix_multiply wants. Both are silent if you get them wrong in a way that")
    print("    still has the right element count.")
    return

@case_mark
def case_weights_do_not_own_their_buffer():
    """The trap that makes hand-built networks fail intermittently."""
    model = make_model()
    write_wts(model.state_dict(), WTS_FILE)
    weight_map = read_wts(WTS_FILE)
    target = np.ascontiguousarray(weight_map["conv1.bias"].astype(np.float32))
    original = target.copy()

    tw = TRTWrapperV1()
    network = tw.network
    input_tensor = network.add_input("x", trt.float32, (1, 1, 4, 4))
    kernel = np.ascontiguousarray(np.zeros((8, 1, 3, 3), dtype=np.float32))
    convolution_layer = network.add_convolution_nd(input_tensor, 8, [3, 3], trt.Weights(kernel), trt.Weights(target))
    convolution_layer.padding_nd = [1, 1]
    network.mark_output(convolution_layer.get_output(0))

    # Overwrite the array *after* handing it to TensorRT but *before* the build.
    target[:] = 999.0
    tw.build([convolution_layer.get_output(0)])
    tw.setup({"x": np.zeros((1, 1, 4, 4), dtype=np.float32)})
    tw.infer(b_print_io=False)
    got = tw.buffer[network.get_output(0).name][0]

    per_channel = got.reshape(8, -1)[:, 0]
    print(f"    bias handed to TensorRT : {np.array2string(original[:4], precision=4)}")
    print(f"    array overwritten with  : 999.0, before build()")
    print(f"    engine actually computed: {np.array2string(per_channel[:4], precision=4)}")
    followed_the_overwrite = bool(np.allclose(per_channel, 999.0))
    print(f"    engine took the overwritten values: {followed_the_overwrite}")
    assert followed_the_overwrite, "if this flips, TensorRT started copying at add_* time"
    print("\n    trt.Weights wraps a pointer. It does not copy at add_convolution_nd() time and it")
    print("    does not keep the object alive -- the values are read when the engine is built.")
    print("    Mutating or freeing the array in between silently changes the engine, and letting")
    print("    it be garbage collected is a use-after-free with no error message. Every")
    print("    hand-built network needs a keep-alive list; see `keep_alive` in the case above.")
    return

def main() -> None:
    case_round_trip()
    case_cost_of_ascii()
    case_register_buffer()
    case_build_from_weights()
    case_weights_do_not_own_their_buffer()
    return

if __name__ == "__main__":
    main()
    print("\nFinish")
