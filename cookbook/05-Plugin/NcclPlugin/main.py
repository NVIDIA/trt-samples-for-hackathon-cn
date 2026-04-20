# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import tensorrt as trt
from tensorrt_cookbook import TRTWrapperV1, case_mark, check_array, get_plugin

NCCL_UNIQUE_ID_BYTES = 128

def unique_id_to_int8_array(unique_id) -> np.ndarray:
    if isinstance(unique_id, (bytes, bytearray)):
        raw = bytes(unique_id)
    elif hasattr(unique_id, "internal"):
        raw = bytes(unique_id.internal)
    else:
        raw = bytes(bytearray(unique_id))

    uid_u8 = np.frombuffer(raw, dtype=np.uint8)
    if uid_u8.size != NCCL_UNIQUE_ID_BYTES:
        raise RuntimeError(f"Unexpected NCCL unique id size: {uid_u8.size}, expected {NCCL_UNIQUE_ID_BYTES}")
    return uid_u8.view(np.int8).copy()

def run_rank(rank: int, unique_id_bytes: bytes, device_id: int):
    import cupy as cp

    cp.cuda.Device(device_id).use()

    shape = [8]
    mode = 0 if rank == 0 else 1
    peer = 1 - rank
    trt_file = Path(__file__).resolve().parent / f"model-rank{rank}.trt"
    plugin_file_list = [Path(__file__).resolve().parent / "NcclSendRecvPlugin.so"]

    input_array = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + 1 if rank == 0 else np.zeros(shape, dtype=np.float32)
    input_data = {"inputT0": input_array}

    plugin_info_dict = {
        "NcclSendRecvPluginLayer": dict(
            name="NcclSendRecv",
            version="1",
            namespace="",
            argument_dict=dict(
                mode=np.array([mode], dtype=np.int32),
                rank=np.array([rank], dtype=np.int32),
                peer=np.array([peer], dtype=np.int32),
                world_size=np.array([2], dtype=np.int32),
                unique_id=unique_id_to_int8_array(unique_id_bytes),
            ),
            number_input_tensor=1,
            number_input_shape_tensor=0,
            plugin_api_version="3",
        )
    }

    tw = TRTWrapperV1(trt_file=trt_file, plugin_file_list=plugin_file_list)
    if tw.engine_bytes is None:
        input_tensor = tw.network.add_input("inputT0", trt.float32, [-1])
        tw.profile.set_shape(input_tensor.name, [1], shape, shape)
        tw.config.add_optimization_profile(tw.profile)

        layer = tw.network.add_plugin_v3([input_tensor], [], get_plugin(plugin_info_dict["NcclSendRecvPluginLayer"]))
        layer.name = "NcclSendRecvPluginLayer"
        output_tensor = layer.get_output(0)
        output_tensor.name = "outputT0"

        tw.build([output_tensor])
        tw.serialize_engine(trt_file)

    tw.setup(input_data)
    tw.infer(b_print_io=False)

    output = tw.buffer["outputT0"][0]
    if rank == 0:
        check_array(output, input_array, True, des="rank0(send) output")
        print(f"rank={rank}, send={input_array.tolist()}, output={output.tolist()}")
    else:
        expected = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + 1
        check_array(output, expected, True, des="rank1(recv) output")
        print(f"rank={rank}, recv={output.tolist()}")

@case_mark
def case_two_process_two_gpu_send_recv():
    from cupy.cuda import nccl
    import cupy as cp

    device_count = cp.cuda.runtime.getDeviceCount()
    if device_count < 2:
        raise SystemExit(f"需要至少 2 张 GPU，当前仅 {device_count} 张。")

    unique_id = nccl.get_unique_id()
    if isinstance(unique_id, tuple):
        unique_id_bytes = bytes(unique_id)
    else:
        unique_id_bytes = bytes(unique_id)

    mp.set_start_method("spawn", force=True)
    procs = [
        mp.Process(target=run_rank, args=(0, unique_id_bytes, 0)),
        mp.Process(target=run_rank, args=(1, unique_id_bytes, 1)),
    ]

    for process in procs:
        process.start()
    for process in procs:
        process.join()

    if any(process.exitcode != 0 for process in procs):
        raise SystemExit("子进程执行失败，请检查 NCCL/CUDA/TensorRT 环境。")

if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    os.system(f"rm -rf {root}/model-rank*.trt")
    case_two_process_two_gpu_send_recv()
    print("Finish")
