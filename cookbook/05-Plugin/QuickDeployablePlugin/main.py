# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import logging
from enum import IntEnum
from pathlib import Path
from typing import List, Tuple

import numpy as np
import numpy.typing as npt
import tensorrt as trt
import tensorrt.plugin as trtp
import torch
import triton
import triton.language as tl
from cuda import cudart

from tensorrt_cookbook import TRTWrapperDDS, TRTWrapperV1, case_mark

@case_mark
def case_add():

    @trtp.register("sample::elemwise_add_plugin")  # Customized plugin name space and plugin name
    def add_plugin_desc(inp0: trtp.TensorDesc, block_size: int) -> trtp.TensorDesc:
        return inp0.like()

    def register_autotune():

        @trtp.autotune("sample::elemwise_add_plugin")
        def add_plugin_autotune(inp0: trtp.TensorDesc, outputs: Tuple[trtp.TensorDesc]) -> List[trtp.AutoTuneCombination]:
            return [trtp.AutoTuneCombination("FP32|FP16, FP32|FP16")]

    @trtp.impl("sample::elemwise_add_plugin")
    def add_plugin_impl(inp0: trtp.Tensor, block_size: int, outputs: Tuple[trtp.Tensor], stream: int) -> None:
        n = inp0.numel()
        inp0_t = torch.as_tensor(inp0, device="cuda")
        out_t = torch.as_tensor(outputs[0], device="cuda")

        @triton.jit
        def add_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask)
            tl.store(y_ptr + offsets, x + 1, mask=mask)

        add_kernel[(triton.cdiv(n, block_size), )](inp0_t, out_t, n, BLOCK_SIZE=block_size)

    if True:  # Enable autotune
        register_autotune()

    BLOCK_SIZE = 256
    data = {"inputT0": np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)}
    trt_file = Path("model-add.trt")

    tw = TRTWrapperV1(trt_file=trt_file)

    input_tensor = tw.network.add_input("inputT0", trt.float32, [-1, -1, -1])
    tw.profile.set_shape(input_tensor.name, [1, 1, 1], [3, 4, 5], [6, 8, 10])
    tw.config.add_optimization_profile(tw.profile)

    layer = tw.network.add_plugin(trtp.op.sample.elemwise_add_plugin(input_tensor, block_size=BLOCK_SIZE))

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_inplace_add():

    @trtp.register("sample::elemwise_add_plugin_")
    def add_plugin_desc_(inp0: trtp.TensorDesc, delta: int) -> trtp.TensorDesc:
        return inp0.aliased()

    @trtp.autotune("sample::elemwise_add_plugin_")
    def add_plugin_autotune_(inp0, outputs) -> List[trtp.AutoTuneCombination]:
        return [
            trtp.AutoTuneCombination("FP32, FP32", "LINEAR*HWC"),
            trtp.AutoTuneCombination("FP32|FP16, FP32|FP16", "LINEAR"),
        ]

    @trtp.impl("sample::elemwise_add_plugin_")
    def add_plugin_impl_(inp0, delta: int, outputs, stream) -> None:
        inp0_t = torch.as_tensor(inp0, device="cuda")
        inp0_t.add_(delta)

    # Use torch APIs
    device = "cuda:0"
    data = {"inputT0": torch.arange(3 * 4 * 5, dtype=torch.float32, device=device).reshape(3, 4, 5).contiguous()}
    trt_file = Path("model-inplace_add.trt")

    tw = TRTWrapperV1(trt_file=trt_file)
    tw.config.set_preview_feature(trt.PreviewFeature.ALIASED_PLUGIN_IO_10_03, True)

    input_tensor = tw.network.add_input("inputT0", trt.float32, [-1, -1, -1])
    tw.profile.set_shape(input_tensor.name, [1, 1, 1], [3, 4, 5], [6, 8, 10])
    tw.config.add_optimization_profile(tw.profile)

    layer = tw.network.add_plugin(trtp.op.sample.elemwise_add_plugin_(input_tensor, delta=1))
    layer = tw.network.add_plugin(trtp.op.sample.elemwise_add_plugin_(layer.get_output(0), delta=1))
    layer.get_output(0).name = "outputT0"
    tw.build([layer.get_output(0)])

    tw.runtime = trt.Runtime(tw.logger)
    tw.engine = tw.runtime.deserialize_cuda_engine(tw.engine_bytes)
    tw.context = tw.engine.create_execution_context()

    tw.context.set_input_shape("inputT0", data["inputT0"].shape)
    tw.context.set_tensor_address("inputT0", data["inputT0"].data_ptr())  # Reuse the buffer for both input and output
    tw.context.set_tensor_address("outputT0", data["inputT0"].data_ptr())

    print("inputT0 before inference")
    print(data["inputT0"])

    tw.context.execute_async_v3(0)
    cudart.cudaStreamSynchronize(tw.stream)

    print("inputT0 after inference")
    print(data["inputT0"])

@case_mark
def case_non_zero():

    @trtp.register("sample::non_zero_plugin")
    def non_zero_plugin_reg(inp0: trtp.TensorDesc, ) -> Tuple[trtp.TensorDesc, trtp.TensorDesc]:
        upper_bound = inp0.shape_expr[0] * inp0.shape_expr[1]
        st = trtp.size_tensor(upper_bound // 2, upper_bound)
        return trtp.from_shape_expr((st.expr(), 2), dtype=trt.int32), st

    @trtp.autotune("sample::non_zero_plugin")
    def non_zero_plugin_autotune(inp0, outputs) -> List[trtp.AutoTuneCombination]:
        return [trtp.AutoTuneCombination("FP32|FP16, INT32, INT32")]

    @trtp.impl("sample::non_zero_plugin")
    def non_zero_plugin_impl(inp0, outputs, stream) -> None:
        inp0_t = torch.as_tensor(inp0, device="cuda")
        out_1 = torch.as_tensor(outputs[1], device="cuda").reshape((-1, ))

        out = torch.nonzero(inp0_t)

        out0 = torch.as_tensor(outputs[0].aliased(out.shape), device="cuda")
        out0.copy_(out)
        out_1.copy_(torch.Tensor([out.shape[0]]))

    data = {"inputT0": (np.random.rand(128 * 128) > 0.9).astype(np.float32).reshape(128, 128)}
    trt_file = Path("model-add.trt")

    tw = TRTWrapperDDS(trt_file=trt_file)

    input_tensor = tw.network.add_input("inputT0", trt.float32, [128, 128])

    layer = tw.network.add_plugin(trtp.op.sample.non_zero_plugin(input_tensor))
    layer.get_output(0).name = "Y"

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_pad(enable_multi_tactic=False):

    @trtp.register("sample::circ_pad_plugin")
    def circ_pad_plugin_desc(inp0: trtp.TensorDesc, pads: npt.NDArray[np.int32]) -> trtp.TensorDesc:
        ndim = inp0.ndim
        out_desc = inp0.like()

        for i in range(np.size(pads) // 2):
            out_desc.shape_expr[ndim - i - 1] += int(pads[i * 2] + pads[i * 2 + 1])

        return out_desc

    def enable_multi_tactic_circ_pad():

        class Tactic(IntEnum):
            TORCH = 1
            TRITON = 2

        @trtp.autotune("sample::circ_pad_plugin")
        def circ_pad_plugin_autotune(
            inp0: trtp.TensorDesc,
            outputs: Tuple[trtp.TensorDesc],
        ) -> List[trtp.AutoTuneCombination]:
            c = trtp.AutoTuneCombination()
            c.pos([0, 1], "FP32|FP16")
            c.tactics([int(Tactic.TORCH), int(Tactic.TRITON)])
            return [c]

        @trtp.impl("sample::circ_pad_plugin")
        def circ_pad_plugin_impl(inp0: trtp.Tensor, pads: npt.NDArray[np.int32], outputs: Tuple[trtp.Tensor], stream: int, tactic: int) -> None:

            log = logging.getLogger("QuicklyDeployablePlugins")
            log.debug(f"Executing for inp0: dtype={inp0.dtype},format={inp0.format} and output[0]: dtype={outputs[0].dtype},format={outputs[0].format}")

            inp_t = torch.as_tensor(inp0, device="cuda")
            out_t = torch.as_tensor(outputs[0], device="cuda")

            if tactic == Tactic.TORCH:
                out = torch.nn.functional.pad(inp_t, pads.tolist(), mode="circular")
                out_t.copy_(out)
            elif tactic == Tactic.TRITON:
                N = inp0.ndim
                all_pads = np.zeros((N * 2, ), dtype=np.int32)
                out_dims = trtp.Shape(tuple(inp0.shape))

                for i in range(np.size(pads) // 2):
                    out_dims[N - i - 1] += pads[i * 2] + pads[i * 2 + 1]
                    all_pads[N * 2 - 2 * i - 2] = pads[i * 2]
                    all_pads[N * 2 - 2 * i - 1] = pads[i * 2 + 1]

                all_pads = all_pads.tolist()

                block_size = 256
                num_blocks = tuple([int((np.prod(out_dims) + block_size - 1) // block_size)])

                import triton.language as tl

                @triton.jit
                def circ_pad(X, all_pads_0, all_pads_2, all_pads_4, all_pads_6, orig_dims_0, orig_dims_1, orig_dims_2, orig_dims_3, Y, Y_shape_1, Y_shape_2, Y_shape_3, X_len, Y_len, BLOCK_SIZE: tl.constexpr):
                    pid = tl.program_id(0)
                    i = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

                    mask_y = i < Y_len

                    i3 = i % Y_shape_3
                    i2 = (i // Y_shape_3) % Y_shape_2
                    i1 = (i // Y_shape_3 // Y_shape_2) % Y_shape_1
                    i0 = i // Y_shape_3 // Y_shape_2 // Y_shape_1

                    j0 = (i0 - all_pads_0 + orig_dims_0) % orig_dims_0
                    j1 = (i1 - all_pads_2 + orig_dims_1) % orig_dims_1
                    j2 = (i2 - all_pads_4 + orig_dims_2) % orig_dims_2
                    j3 = (i3 - all_pads_6 + orig_dims_3) % orig_dims_3

                    load_idx = (orig_dims_3 * orig_dims_2 * orig_dims_1 * j0 + orig_dims_3 * orig_dims_2 * j1 + orig_dims_3 * j2 + j3)
                    mask_x = load_idx < X_len

                    x = tl.load(X + load_idx, mask=mask_x)

                    tl.store(Y + i, x, mask=mask_y)

                circ_pad[num_blocks](inp_t, all_pads[0], all_pads[2], all_pads[4], all_pads[6], inp0.shape[0], inp0.shape[1], inp0.shape[2], inp0.shape[3], out_t, int(out_dims[1]), int(out_dims[2]), int(out_dims[3]), inp0.numel(), out_dims.numel(), BLOCK_SIZE=block_size)

    # Helper to define a single tactic implementation of the plugin
    def enable_single_tactic_circ_pad():

        @trtp.autotune("sample::circ_pad_plugin")
        def circ_pad_plugin_autotune(
            inp0: trtp.TensorDesc,
            outputs: Tuple[trtp.TensorDesc],
        ) -> List[trtp.AutoTuneCombination]:

            return [trtp.AutoTuneCombination("FP32|FP16, FP32|FP16")]

        @trtp.impl("sample::circ_pad_plugin")
        def circ_pad_plugin_impl(
            inp0: trtp.Tensor,
            pads: npt.NDArray[np.int32],
            outputs: Tuple[trtp.Tensor],
            stream: int,
        ) -> None:
            inp_t = torch.as_tensor(inp0, device="cuda")
            out_t = torch.as_tensor(outputs[0], device="cuda")

            out = torch.nn.functional.pad(inp_t, pads.tolist(), mode="circular")
            out_t.copy_(out)

    if enable_multi_tactic:
        enable_multi_tactic_circ_pad()
    else:
        enable_single_tactic_circ_pad()

    shape = 1, 3, 32, 32
    data = {"inputT0": np.tile(np.arange(32, dtype=np.float32).reshape(1, 1, 8, 4), [1, 3, 4, 8])}
    trt_file = Path("model-add.trt")

    tw = TRTWrapperV1(trt_file=trt_file)

    input_tensor = tw.network.add_input("inputT0", trt.float32, shape)
    pads = np.array((1, 1, 1, 1), dtype=np.int32)
    layer = tw.network.add_plugin(trtp.op.sample.circ_pad_plugin(input_tensor, pads=pads))
    layer.get_output(0).name = "outputT0"

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

    ref = np.pad(data["inputT0"], [[0, 0], [0, 0], [pads[0], pads[1]], [pads[2], pads[3]]], "wrap")
    print(f"res={np.allclose(tw.buffer['outputT0'][0], ref, atol=1e-2)}")

if __name__ == "__main__":
    case_add()
    case_inplace_add()
    case_non_zero()
    case_pad(False)
    case_pad(True)
    print("Finish")
