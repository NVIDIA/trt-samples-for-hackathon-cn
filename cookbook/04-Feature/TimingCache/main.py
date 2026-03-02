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
from pathlib import Path
from time import time

import tensorrt as trt

timing_cache_file = Path("model.TimingCache")
b_ignore_mismatch = False  # True allows loading cache created from a different device
shape = [8, 1, 28, 28]

from tensorrt_cookbook import TRTWrapperV1, add_mea, build_mnist_network_trt

def run(iNetwork, b_use_timing_cache):
    print("#--------------------------------------------------------------")
    tw = TRTWrapperV1()

    timing_cache_buffer = b""
    if b_use_timing_cache and timing_cache_file.exists():
        with open(timing_cache_file, "rb") as f:
            timing_cache_buffer = f.read()
        if timing_cache_buffer == None:
            print(f"Failed loading {timing_cache_file}")
            return
        print(f"Succeeded loading {timing_cache_file}")

    if b_use_timing_cache:
        timing_cache = tw.config.create_timing_cache(timing_cache_buffer)
        #timing_cache.reset()  # reset the timing cache
        tw.config.set_timing_cache(timing_cache, b_ignore_mismatch)

    input_tensor = tw.network.add_input("inputT0", trt.float32, [-1] + shape[1:])
    tw.profile.set_shape(input_tensor.name, shape, [8] + shape[1:], [16] + shape[1:])
    tw.config.add_optimization_profile(tw.profile)

    # Common part
    output_tensor_list = build_mnist_network_trt(tw.config, tw.network, tw.profile)
    tensor = output_tensor_list[0]

    # difference part
    if iNetwork == 0:
        tensor = add_mea(tw.network, tensor, [10, 512])
        tensor = add_mea(tw.network, tensor, [512, 10])
    else:
        tensor = add_mea(tw.network, tensor, [10, 768])
        tensor = add_mea(tw.network, tensor, [768, 10])
        tensor = add_mea(tw.network, tensor, [10, 2048])
        tensor = add_mea(tw.network, tensor, [2048, 10])

    layer = tw.network.add_softmax(tensor)
    layer.axes = 1 << 1
    layer = tw.network.add_topk(layer.get_output(0), trt.TopKOperation.MAX, 1, 1 << 1)

    t0 = time()
    tw.build([layer.get_output(0)])
    t1 = time()
    print(f"{iNetwork = }, {b_use_timing_cache = }, build time: {(t1 - t0) * 1000: 10.3f} ms")

    if b_use_timing_cache:
        timing_cache_new = tw.config.get_timing_cache()
        #res = timing_cache.combine(timing_cache_new, b_ignore_mismatch)  # merge timing cache from the old one (load form file) with the new one (created by this build), not required
        timing_cache = timing_cache_new
        #print("timing_cache.combine:%s" % res)

        timing_cache_buffer = timing_cache.serialize()
        with open(timing_cache_file, "wb") as f:
            f.write(timing_cache_buffer)
            print(f"Succeed saving {timing_cache_file}")

if __name__ == "__main__":
    os.system("rm -rfv model.TimingCache")

    # Case 0, Build network 0, without timing cache
    run(0, 0)

    # Case 1, Build network 0 again without no timing cache, build-time is a little bit shorter than Case 0 due to GPU warming up
    run(0, 0)

    # Case 2, Build network 1 without timing cache
    run(1, 0)

    # Case 3, Build network 1 again without timing cache, build-time is a little bit shorter than Case 2 due to GPU warming up
    run(1, 0)

    # Case 4, Build network 0 with writing timing cache, almost the same time as Case 1
    run(0, 1)
    os.system("ls -alh |grep model.TimingCache")

    # Case 5, Build network 0 again with reading timing cache, build time is much shorter than Case 4
    run(0, 1)
    os.system("ls -alh |grep model.TimingCache")

    # Case 6, Build network 1 with reading and appending timing cache, build-time is somehow shorter than Case 3
    # i.e. it earns timing cache from a similar but different network.
    # Meawhile, the size of file `model.TimingCache` increases
    run(1, 1)
    os.system("ls -alh |grep model.TimingCache")

    # Case 7, Build network 1 again with reading timing cache, build-time is much shorter than Case 6
    run(1, 1)
    os.system("ls -alh |grep model.TimingCache")

    # Case 8, Build network 0 again with reading timing cache, build-time is similar (or shorter?) as Case 5
    # i.e. timing cache of both network 0 and 1 are stored together in file `model.TimingCache`.
    run(0, 1)
    os.system("ls -alh |grep model.TimingCache")

    print("Finish")
