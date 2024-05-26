#
# Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
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
import sys
from pathlib import Path

import numpy as np
import tensorrt as trt
from cuda import cudart

sys.path.append("/trtcookbook/include")
from utils import TRTWrapperV1, case_mark, datatype_np_to_trt

height, width = 28, 28
data_path = Path("/trtcookbook/00-MNISTData/")
onnx_file_untrained = data_path / "model0.onnx"
weight_file_trained = data_path / "model1.npz"
onnx_file_trained = data_path / "model1.onnx"
onnx_file_trained_no_weight = data_path / "model2.onnx"
onnx_file_weight = data_path / "model2.onnx.weight"
trt_file = "./model.trt"
data = {"inputT0": np.load(data_path / "InferenceData.npy")}

@case_mark
def case_dummy_engine():
    tw = TRTWrapperV1()
    tw.config.set_flag(trt.BuilderFlag.REFIT)
    tw.config.set_flag(trt.BuilderFlag.STRIP_PLAN)  # build a weightless engine for refit later

    parser = trt.OnnxParser(tw.network, tw.logger)
    with open(onnx_file_untrained, "rb") as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            exit()

    input_tensor = tw.network.get_input(0)
    input_tensor.name = "inputT0"
    tw.profile.set_shape(input_tensor.name, [1, 1, height, width], [4, 1, height, width], [8, 1, height, width])
    tw.config.add_optimization_profile(tw.profile)
    tw.network.get_output(0).name = "outputT0"
    tw.network.get_output(1).name = "outputT1"

    tw.build()
    tw.serialize_engine(trt_file)

    tw.setup(data)
    tw.infer()

@case_mark
def case_set_weights():
    tw = TRTWrapperV1()

    with open(trt_file, "rb") as f:
        tw.engine = trt.Runtime(tw.logger).deserialize_cuda_engine(f.read())
    tw.refitter = trt.Refitter(tw.engine, tw.logger)

    w = np.load(weight_file_trained)
    # Two equivalent implementations of refitting

    # 1. Use API set_weights
    tw.refitter.set_weights("/conv1/Conv", trt.WeightsRole.KERNEL, np.ascontiguousarray(w["conv1.weight"]))  # USe np.ascontiguousarray, BLOODY lesson!

    tw.refitter.set_weights("/conv1/Conv", trt.WeightsRole.BIAS, np.ascontiguousarray(w["conv1.bias"]))
    tw.refitter.set_weights("/conv2/Conv", trt.WeightsRole.KERNEL, np.ascontiguousarray(w["conv2.weight"]))
    tw.refitter.set_weights("/conv2/Conv", trt.WeightsRole.BIAS, np.ascontiguousarray(w["conv2.bias"]))
    tw.refitter.set_weights("fc1.weight", trt.WeightsRole.CONSTANT, np.ascontiguousarray(w["fc1.weight"]))
    tw.refitter.set_weights("fc1.bias", trt.WeightsRole.CONSTANT, np.ascontiguousarray(w["fc1.bias"]))
    tw.refitter.set_weights("fc2.weight", trt.WeightsRole.CONSTANT, np.ascontiguousarray(w["fc2.weight"]))
    tw.refitter.set_weights("fc2.bias", trt.WeightsRole.CONSTANT, np.ascontiguousarray(w["fc2.bias"]))
    """
    # 2. Use API set_named_weights
    for key in w.keys():
        tw.refitter.set_named_weights(key, trt.Weights(np.ascontiguousarray(w[key])))  # key name is the same between network and weight file
    """
    tw.refitter.refit_cuda_engine()

    tw.setup(data)
    tw.infer()

@case_mark
def case_set_weights_gpu():
    tw = TRTWrapperV1()

    with open(trt_file, "rb") as f:
        tw.engine = trt.Runtime(tw.logger).deserialize_cuda_engine(f.read())
    tw.refitter = trt.Refitter(tw.engine, tw.logger)

    w = np.load(weight_file_trained)

    buffer_list = []
    for key in w.keys():  # Copy the buffer to GPU, so that showing refitting from GPU
        ww = np.ascontiguousarray(w[key])
        n_byte = ww.size * ww.itemsize
        buffer = cudart.cudaMalloc(n_byte)[1]
        buffer_list.append(buffer)
        cudart.cudaMemcpy(buffer, ww.ctypes.data, n_byte, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
        weight = trt.Weights(datatype_np_to_trt(ww.dtype), buffer, ww.size)
        tw.refitter.set_named_weights(key, weight, trt.TensorLocation.DEVICE)

    tw.refitter.refit_cuda_engine()

    for buffer in buffer_list:
        cudart.cudaFree(buffer)

    tw.setup(data)
    tw.infer()

@case_mark
def case_refit_engine():
    tw = TRTWrapperV1()

    with open(trt_file, "rb") as f:
        tw.engine = trt.Runtime(tw.logger).deserialize_cuda_engine(f.read())

    tw.refitter = trt.Refitter(tw.engine, tw.logger)
    tw.opr = trt.OnnxParserRefitter(tw.refitter, tw.logger)
    # Three equivalent implementations of refitting

    # 1. Refit from ONNX file
    tw.opr.refit_from_file(str(onnx_file_trained))
    """
    # 2. Refit from ONNX file stream
    with open(onnx_file_trained, "rb") as onnx_model:
        tw.opr.refit_from_bytes(onnx_model.read())

    # 3. Refit from ONNX file stream with external weight
    with open(onnx_file_trained, "rb") as onnx_model:
        tw.opr.refit_from_bytes(onnx_model.read(), str(onnx_file_weight))
    """
    tw.refitter.refit_cuda_engine()

    tw.setup(data)
    tw.infer()

@case_mark
def case_other_api():
    tw = TRTWrapperV1()

    with open(trt_file, "rb") as f:
        tw.engine = trt.Runtime(tw.logger).deserialize_cuda_engine(f.read())

    tw.refitter = trt.Refitter(tw.engine, tw.logger)

    tw.refitter.error_recorder = None  # Custom error recorder, default value is None
    tw.refitter.logger  # Custom logger, default value is tw.logger
    tw.refitter.max_threads = 4  # Maximum thread used by the Refitter, default value is 1
    tw.refitter.weights_validation = False  # Skip validation if we confirm the new weights is correct, default value is True

    name_list, role_list = tw.refitter.get_all()
    print("All refittable layers and corresponding roles:")
    for name, role in zip(name_list, role_list):
        print(name, role.__str__()[12:])

    print("All refittable weights:")
    for name in tw.refitter.get_all_weights():
        print(name)

    missing_layer_list, weight_role_list = tw.refitter.get_missing()  # get name and role of the missing weights
    print("Missing layer and role names")
    for layer, role in zip(missing_layer_list, weight_role_list):
        print(f"[{layer}-{role}]")

    missing_layer_list = tw.refitter.get_missing_weights()  # only get name of the refitable weights
    print("Missing layer and role names")
    for layer in missing_layer_list:
        print(f"[{layer}")

    #tw.refitter.set_dynamic_range("?")
    #tw.refitter.get_dynamic_range("?")
    #tw.refitter.get_tensors_with_dynamic_range()

    w = np.load(weight_file_trained)
    for key in w.keys():
        tw.refitter.set_named_weights(key, trt.Weights(w[key]))  # key name is the same between network and weight file

    for key in w.keys():
        tw.refitter.get_weights_location(key)
        tw.refitter.get_weights_prototype(key)
        tw.refitter.get_named_weights(key).numpy()
        tw.refitter.unset_named_weights(key)  # reverse operation of get_named_weights

    tw.refitter.refit_cuda_engine()
    tw.refitter.refit_cuda_engine_async(0)

    tw.setup(data)
    tw.infer()

if __name__ == "__main__":
    os.system("rm -rf */trt")

    case_dummy_engine()
    case_set_weights()
    case_set_weights_gpu()
    case_refit_engine()
    #case_other_api()

    print("Finish")
