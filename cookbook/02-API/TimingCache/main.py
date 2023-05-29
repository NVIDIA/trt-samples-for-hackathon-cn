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

import os
from time import time

import numpy as np
import tensorrt as trt
from cuda import cudart

trtFile = "model.plan"
timingCacheFile = "model.TimingCache"
bIgnoreMismatch = False  # turn on if we allow the timing cache file using among different device
shape = [8, 1, 28, 28]

def run(iNetwork, bUseTimeCache):
    print("iNetwork=%d, bUseTimeCache=%d" % (iNetwork, bUseTimeCache))
    logger = trt.Logger(trt.Logger.ERROR)
    timingCacheString = b""
    if bUseTimeCache and os.path.isfile(timingCacheFile):
        with open(timingCacheFile, "rb") as f:
            timingCacheString = f.read()
        if timingCacheString == None:
            print("Failed loading %s" % timingCacheFile)
            return
        print("Succeeded loading %s" % timingCacheFile)

    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()
    config = builder.create_builder_config()
    if bUseTimeCache:
        timingCache = config.create_timing_cache(timingCacheString)
        #timingCache.reset()  # clean the timing cache, not required
        config.set_timing_cache(timingCache, bIgnoreMismatch)

    inputTensor = network.add_input("inputT0", trt.float32, [-1] + shape[1:])
    profile.set_shape(inputTensor.name, [2] + shape[1:], [4] + shape[1:], [8] + shape[1:])
    config.add_optimization_profile(profile)

    # Common part
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

    # Differnece part
    if iNetwork == 0:
        w = np.ascontiguousarray(np.random.rand(10, 512).astype(np.float32))
        b = np.ascontiguousarray(np.random.rand(1, 512).astype(np.float32))
        layerWeight = network.add_constant(w.shape, trt.Weights(w))
        layer = network.add_matrix_multiply(_15.get_output(0), trt.MatrixOperation.NONE, layerWeight.get_output(0), trt.MatrixOperation.NONE)
        layerBias = network.add_constant(b.shape, trt.Weights(b))
        layer = network.add_elementwise(layer.get_output(0), layerBias.get_output(0), trt.ElementWiseOperation.SUM)
        layer = network.add_activation(layer.get_output(0), trt.ActivationType.RELU)

        w = np.ascontiguousarray(np.random.rand(512, 10).astype(np.float32))
        b = np.ascontiguousarray(np.random.rand(1, 10).astype(np.float32))
        layerWeight = network.add_constant(w.shape, trt.Weights(w))
        layer = network.add_matrix_multiply(layer.get_output(0), trt.MatrixOperation.NONE, layerWeight.get_output(0), trt.MatrixOperation.NONE)
        layerBias = network.add_constant(b.shape, trt.Weights(b))
        layer = network.add_elementwise(layer.get_output(0), layerBias.get_output(0), trt.ElementWiseOperation.SUM)

    else:
        w = np.ascontiguousarray(np.random.rand(10, 768).astype(np.float32))
        b = np.ascontiguousarray(np.random.rand(1, 768).astype(np.float32))
        layerWeight = network.add_constant(w.shape, trt.Weights(w))
        layer = network.add_matrix_multiply(_15.get_output(0), trt.MatrixOperation.NONE, layerWeight.get_output(0), trt.MatrixOperation.NONE)
        layerBias = network.add_constant(b.shape, trt.Weights(b))
        layer = network.add_elementwise(layer.get_output(0), layerBias.get_output(0), trt.ElementWiseOperation.SUM)
        layer = network.add_activation(layer.get_output(0), trt.ActivationType.RELU)

        w = np.ascontiguousarray(np.random.rand(768, 10).astype(np.float32))
        b = np.ascontiguousarray(np.random.rand(1, 10).astype(np.float32))
        layerWeight = network.add_constant(w.shape, trt.Weights(w))
        layer = network.add_matrix_multiply(layer.get_output(0), trt.MatrixOperation.NONE, layerWeight.get_output(0), trt.MatrixOperation.NONE)
        layerBias = network.add_constant(b.shape, trt.Weights(b))
        layer = network.add_elementwise(layer.get_output(0), layerBias.get_output(0), trt.ElementWiseOperation.SUM)

        layer = network.add_activation(layer.get_output(0), trt.ActivationType.RELU)

        w = np.ascontiguousarray(np.random.rand(10, 2048).astype(np.float32))
        b = np.ascontiguousarray(np.random.rand(1, 2048).astype(np.float32))
        layerWeight = network.add_constant(w.shape, trt.Weights(w))
        layer = network.add_matrix_multiply(layer.get_output(0), trt.MatrixOperation.NONE, layerWeight.get_output(0), trt.MatrixOperation.NONE)
        layerBias = network.add_constant(b.shape, trt.Weights(b))
        layer = network.add_elementwise(layer.get_output(0), layerBias.get_output(0), trt.ElementWiseOperation.SUM)
        layer = network.add_activation(layer.get_output(0), trt.ActivationType.RELU)

        w = np.ascontiguousarray(np.random.rand(2048, 10).astype(np.float32))
        b = np.ascontiguousarray(np.random.rand(1, 10).astype(np.float32))
        layerWeight = network.add_constant(w.shape, trt.Weights(w))
        layer = network.add_matrix_multiply(layer.get_output(0), trt.MatrixOperation.NONE, layerWeight.get_output(0), trt.MatrixOperation.NONE)
        layerBias = network.add_constant(b.shape, trt.Weights(b))
        layer = network.add_elementwise(layer.get_output(0), layerBias.get_output(0), trt.ElementWiseOperation.SUM)

    _16 = network.add_softmax(layer.get_output(0))
    _16.axes = 1 << 1

    _17 = network.add_topk(_16.get_output(0), trt.TopKOperation.MAX, 1, 1 << 1)

    network.mark_output(_17.get_output(1))

    t0 = time()
    engineString = builder.build_serialized_network(network, config)
    t1 = time()
    print("%s timing cache, %f ms" % ("With" if bUseTimeCache else "Without", (t1 - t0) * 1000))

    if bUseTimeCache:
        timingCacheNew = config.get_timing_cache()
        #res = timingCache.combine(timingCacheNew, bIgnoreMismatch)  # merge timing cache from the old one (load form file) with the new one (created by this build), not required
        timingCache = timingCacheNew
        #print("timingCache.combine:%s" % res)

        timeCacheString = timingCache.serialize()
        with open(timingCacheFile, "wb") as f:
            f.write(timeCacheString)
            print("Succeeded saving %s" % timingCacheFile)

    print("#--------------------------------------------------------------------")

if __name__ == "__main__":
    os.system("rm -rfv model.TimingCache")
    np.set_printoptions(precision=3, linewidth=100, suppress=True)
    cudart.cudaDeviceSynchronize()

    run(0, 0)
    run(0, 0)
    run(1, 0)
    run(1, 0)

    run(0, 1)
    os.system("ls -alh |grep model.TimingCache")
    run(0, 1)
    os.system("ls -alh |grep model.TimingCache")
    run(1, 1)
    os.system("ls -alh |grep model.TimingCache")
    run(1, 1)
    os.system("ls -alh |grep model.TimingCache")
    run(0, 1)
    os.system("ls -alh |grep model.TimingCache")
