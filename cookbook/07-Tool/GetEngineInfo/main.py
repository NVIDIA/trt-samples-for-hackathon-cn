#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import re
import sys
from collections import OrderedDict
from pathlib import Path

import tensorrt as trt
from cuda import cudart

device_index = 0  # Print information of coresponding GPU (set it if we have more than one GPU)
trt_file = Path("model.trt")

class Pointer:

    def __init__(self, byte):
        self.byte = byte
        self.offset = 0

    def set_offset(self, offset):
        self.offset = offset

    def read_and_move(self, size: int = 1, return_number: bool = True):
        target_byte = self.byte[self.offset:self.offset + size]
        self.offset += size
        if return_number:
            return int.from_bytes(target_byte, byteorder=sys.byteorder)
        return target_byte

    def f(self, name, size):  # Print and return value
        data_number = p.read_and_move(size)
        if name != "":
            print(f"{name:<28s}:{data_number:>16d}")
        return data_number

with open(trt_file, "rb") as f:
    p = Pointer(f.read())

# ================================================================
print("=" * 64 + " Current TensorRT")  # Print current TRT environment
info = os.popen(r"cat /usr/include/x86_64-linux-gnu/NvInferVersion.h").read()
v_major = re.search(r"NV_TENSORRT_MAJOR \d+", info)
v_minor = re.search(r"NV_TENSORRT_MINOR \d+", info)
v_patch = re.search(r"NV_TENSORRT_PATCH \d+", info)
v_build = re.search(r"NV_TENSORRT_BUILD \d+", info)
v_major = "None" if v_major is None else v_major.group().split(" ")[-1]
v_minor = "None" if v_minor is None else v_minor.group().split(" ")[-1]
v_patch = "None" if v_patch is None else v_patch.group().split(" ")[-1]
v_build = "None" if v_build is None else v_build.group().split(" ")[-1]
print(f"{'Major':<28s}:{v_major:>16s}")
print(f"{'Minor':<28s}:{v_minor:>16s}")
print(f"{'Patch':<28s}:{v_patch:>16s}")
print(f"{'Build':<28s}:{v_build:>16s}")
print(f"{'TRT-Python':<28s}:{trt.__version__:>16s}")

# ================================================================
print("=" * 64 + " Engine header")
print(f"{'HeaderSize':<28s}:{32:>16d}")
p.f("MagicTag", 4)  # uint32_t -> 4
p.f("SerializationVersion", 4)
nEntry = p.f("nEntry", 8)  # uint64_t -> 8
plan_size = p.f("PlanTotalSize", 8)
trt_major = p.f("TRT.Major", 1)  # uint8_t -> 1
p.f("TRT.Minor", 1)
p.f("TRT.Patch", 1)
p.f("TRT.Build", 1)
p.f("Pad", 4)

# ================================================================
print("=" * 64 + " Engine data")

p.set_offset(32)  # Skip head, in fact `p` must be already at this location before this expression
section_list = []
for i in range(nEntry):
    type = p.read_and_move(4, False)
    pad = p.read_and_move(4)
    offset = p.read_and_move(8)
    size = p.read_and_move(8)
    section_list.append([type, pad, offset, size])
#print(section_list)

p.set_offset(section_list[0][2])  # We only print engine section (inded `0`)
p.f("MagicTag", 4)
p.f("SafeVersion", 4)
p.f("StdVersion", 4)
p.f("HashRead", 4)
p.f("SizeRead", 8)
p.f("", 4)
p.f("TRT.Major", 1)
p.f("", 4)
p.f("TRT.Minor", 1)
p.f("", 4)
p.f("TRT.Patch", 1)
p.f("", 4)
p.f("TRT.Build", 1)
p.f("", 8)
p.f("HardwareCompatLevel", 4)
p.f("", 8)
p.f("", 4)

# ================================================================
print("=" * 64 + " Device information")
print(f"{'Property name':<28s}:{'Engine':^16s} <-> {'Current':^16s}")

eci = OrderedDict()
eci["major"] = p.f("", 4)
p.f("", 4)
eci["minor"] = p.f("", 4)
p.f("", 4)
eci["maxCoreClockRate"] = p.f("", 4)
p.f("", 4)
eci["maxMemoryClockRate"] = p.f("", 4)
p.f("", 4)
eci["memoryBusWidth"] = p.f("", 4)
p.f("", 4)
eci["l2CacheSize"] = p.f("", 4)
p.f("", 8)
eci["maxPersistentL2CacheSize"] = p.f("", 4)
p.f("", 8)
eci["sharedMemPerBlock"] = p.f("", 4)
p.f("", 4)
eci["sharedMemPerMultiprocessor"] = p.f("", 4)
p.f("", 4)
eci["textureAlignment"] = p.f("", 4)
p.f("", 4)
eci["multiProcessorCount"] = p.f("", 4)
p.f("", 4)
eci["integrated"] = p.f("", 1)
p.f("", 4)
eci["maxThreadsPerBlock"] = p.f("", 4)
p.f("", 4)
eci["maxGridDimX"] = p.f("", 4)
p.f("", 4)
eci["maxGridDimY"] = p.f("", 4)
p.f("", 4)
eci["maxGridDimZ"] = p.f("", 4)
p.f("", 4)
if trt_major >= 10:
    p.f("", 8)
eci["totalGlobalMem"] = p.f("", 8)
p.f("", 4)
eci["maxTexture1DLinear"] = p.f("", 4)
p.f("", 4)

_, info = cudart.cudaGetDeviceProperties(device_index)
rci = OrderedDict()
rci["major"] = info.major
rci["minor"] = info.minor
rci["maxCoreClockRate"] = info.clockRate
rci["maxMemoryClockRate"] = info.memoryClockRate
rci["memoryBusWidth"] = info.memoryBusWidth
rci["l2CacheSize"] = info.l2CacheSize
rci["maxPersistentL2CacheSize"] = info.persistingL2CacheMaxSize
rci["sharedMemPerBlock"] = info.sharedMemPerBlock
rci["sharedMemPerMultiprocessor"] = info.sharedMemPerMultiprocessor
rci["textureAlignment"] = info.textureAlignment
rci["multiProcessorCount"] = info.multiProcessorCount
rci["integrated"] = info.integrated
rci["maxThreadsPerBlock"] = info.maxThreadsPerBlock
rci["maxGridDimX"] = info.maxGridSize[0]
rci["maxGridDimY"] = info.maxGridSize[1]
rci["maxGridDimZ"] = info.maxGridSize[2]
rci["totalGlobalMem"] = info.totalGlobalMem
rci["maxTexture1DLinear"] = info.maxTexture1DLinear

for name in eci.keys():
    print(f"{name:<28s}:{eci[name]:16d} <->{rci[name]:16d}")

print("=" * 64 + " Input / output information")

tw = TRTWrapperV1(trt_file=trt_file)
runtime = trt.Runtime(tw.logger)
engine = runtime.deserialize_cuda_engine(p.byte)
context = engine.create_execution_context()

tensor_name_list = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
n_input = sum([engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT for name in tensor_name_list])
n_output = engine.num_io_tensors - n_input

if b_print_io:
    for name in tensor_name_list:
        mode = engine.get_tensor_mode(name)
        data_type = engine.get_tensor_dtype(name)
        buildtime_shape = engine.get_tensor_shape(name)
        print(f"{'Input ' if mode == trt.TensorIOMode.INPUT else 'Output'}->{data_type}, {buildtime_shape}, {name}")

print("Finish")
