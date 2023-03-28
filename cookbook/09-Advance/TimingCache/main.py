#
# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
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
#

from cuda import cudart
import numpy as np
import os
import tensorrt as trt
from time import time

trtFile = "./model.plan"
timingCacheFile = "./model.TimingCache"
bIgnoreMismatch = False  # turn on if we allow the timing cache file using among different device
nB, nC, nH, nW = 8, 1, 28, 28
data = np.random.rand(nB, nC, nH, nW).astype(np.float32) * 2 - 1
np.random.seed(31193)

def run(bUseTimeCache):
    logger = trt.Logger(trt.Logger.ERROR)
    timingCacheString = b""
    if bUseTimeCache and os.path.isfile(timingCacheFile):
        with open(timingCacheFile, "rb") as f:
            timingCacheString = f.read()
        if timingCacheString == None:
            print("Failed getting serialized timing cache!")
            return
        print("Succeeded getting serialized timing cache!")

    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()
    config = builder.create_builder_config()
    if bUseTimeCache:
        timingCache = config.create_timing_cache(timingCacheString)
        #timingCache.reset()  # clean the timing cache, not required
        config.set_timing_cache(timingCache, bIgnoreMismatch)

    inputTensor = network.add_input("inputT0", trt.float32, [-1, nC, nH, nW])
    profile.set_shape(inputTensor.name, [2, nC, nH, nW], [4, nC, nH, nW], [8, nC, nH, nW])
    config.add_optimization_profile(profile)

    w = np.ascontiguousarray(np.random.rand(32, 1, 5, 5).astype(np.float32))
    b = np.ascontiguousarray(np.random.rand(32).astype(np.float32))
    _0 = network.add_convolution_nd(inputTensor, 32, [5, 5], w, b)
    _0.padding_nd = [2, 2]
    _1 = network.add_activation(_0.get_output(0), trt.ActivationType.RELU)
    _2 = network.add_pooling_nd(_1.get_output(0), trt.PoolingType.MAX, [2, 2])
    _2.stride_nd = [2, 2]

    w = np.ascontiguousarray(np.random.rand(64, 32, 5, 5).astype(np.float32))
    b = np.ascontiguousarray(np.random.rand(64).astype(np.float32))
    _3 = network.add_convolution_nd(_2.get_output(0), 64, [5, 5], w, b)
    _3.padding_nd = [2, 2]
    _4 = network.add_activation(_3.get_output(0), trt.ActivationType.RELU)
    _5 = network.add_pooling_nd(_4.get_output(0), trt.PoolingType.MAX, [2, 2])
    _5.stride_nd = [2, 2]

    _6 = network.add_shuffle(_5.get_output(0))
    _6.first_transpose = (0, 2, 3, 1)
    _6.reshape_dims = (-1, 64 * 7 * 7)

    w = np.ascontiguousarray(np.random.rand(64 * 7 * 7, 1024).astype(np.float32))
    b = np.ascontiguousarray(np.random.rand(1, 1024).astype(np.float32))
    _7 = network.add_constant(w.shape, trt.Weights(w))
    _8 = network.add_matrix_multiply(_6.get_output(0), trt.MatrixOperation.NONE, _7.get_output(0), trt.MatrixOperation.NONE)
    _9 = network.add_constant(b.shape, trt.Weights(b))
    _10 = network.add_elementwise(_8.get_output(0), _9.get_output(0), trt.ElementWiseOperation.SUM)
    _11 = network.add_activation(_10.get_output(0), trt.ActivationType.RELU)

    w = np.ascontiguousarray(np.random.rand(1024, 10).astype(np.float32))
    b = np.ascontiguousarray(np.random.rand(1, 10).astype(np.float32))
    _12 = network.add_constant(w.shape, trt.Weights(w))
    _13 = network.add_matrix_multiply(_11.get_output(0), trt.MatrixOperation.NONE, _12.get_output(0), trt.MatrixOperation.NONE)
    _14 = network.add_constant(b.shape, trt.Weights(b))
    _15 = network.add_elementwise(_13.get_output(0), _14.get_output(0), trt.ElementWiseOperation.SUM)

    _16 = network.add_softmax(_15.get_output(0))
    _16.axes = 1 << 1

    _17 = network.add_topk(_16.get_output(0), trt.TopKOperation.MAX, 1, 1 << 1)

    network.mark_output(_17.get_output(1))

    t0 = time()
    engineString = builder.build_serialized_network(network, config)
    t1 = time()
    print("%s timing cache, %f ms" % ("With" if bUseTimeCache else "Without", (t1 - t0) * 1000))

    if bUseTimeCache and not os.path.isfile(timingCacheFile):
        timingCacheNew = config.get_timing_cache()
        #timingCache.combine(timingCacheNew, bIgnoreMismatch)  # merge timing cache from the old one (load form file) with the new one (created by this build), not required
        timingCache = timingCacheNew
        timeCacheString = timingCache.serialize()
        with open(timingCacheFile, "wb") as f:
            f.write(timeCacheString)
            print("Succeeded saving %s" % timingCacheFile)

if __name__ == "__main__":
    os.system("rm -rf ./*.cache")
    np.set_printoptions(precision=3, linewidth=100, suppress=True)
    cudart.cudaDeviceSynchronize()

    run(0)  # build engine without timing cache
    run(0)  # build engine without timing cache again
    run(1)  # build engine with saving timing cache
    run(1)  # build engine with loading timing cache
