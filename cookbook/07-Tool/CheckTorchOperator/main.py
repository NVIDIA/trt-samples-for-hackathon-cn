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

import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import tensorrt as trt
import torch as t
from polygraphy.backend.onnx.loader import fold_constants
from tensorrt_cookbook import TRTWrapperV1, case_mark

np.random.seed(31193)
t.manual_seed(97)
t.cuda.manual_seed_all(97)
t.backends.cudnn.deterministic = True

class CheckTorchOperator:

    def __init__(self, net):
        self.net = net().cuda()

    def check(
        self,
        onnx_file: Path = Path("model.onnx"),
        data: dict = {},
        dynamic_dim_dict: dict = {},
        b_polygraphy: bool = True,
    ):
        print(f"Input :\n{data}")
        input_names = []
        model_input = {}
        dynamic_axes = {}
        for k, v in data.items():
            input_names.append(k)
            model_input[k] = t.from_numpy(v).cuda()
            dd = {}
            if k in dynamic_dim_dict:
                for d in dynamic_dim_dict[k]:
                    dd[d] = f"{k}_{d}"
            dynamic_axes[k] = dd

        # Test the operator in pyTorch
        print(f"{'='* 16} Run in pyTorch")
        with t.no_grad():
            output_torch = self.net(**model_input)
            if isinstance(output_torch, t.Tensor):
                output_torch = [output_torch]

        output_name_list = [f"output_{i}" for i in range(len(output_torch))]
        print(f"Output:")
        for name, tensor in zip(output_name_list, output_torch):
            print(f"{name}\n", tensor.detach().cpu().numpy())

        # Try to export to ONNX
        print(f"{'='* 16} Export to ONNX")
        try:
            t.onnx.export( \
                self.net,
                model_input,
                onnx_file,
                input_names=input_names,
                output_names=output_name_list,
                do_constant_folding=True,
                verbose=False,
                keep_initializers_as_inputs=False,
                opset_version=17,
                dynamic_axes=dynamic_axes,
                )
            print(f"Succeed exporting {onnx_file}")
        except:
            print(f"Failed")
            return

        # Try polygraphy to simplify the ONNX file
        print(f"{'='* 16} Simplify by polygraphy")
        if b_polygraphy:
            try:
                onnx_file_po = Path(str(onnx_file)[:-5] + "-po.onnx")
                onnx_model = onnx.load(onnx_file)
                onnx_model = fold_constants(onnx_model, allow_onnxruntime_shape_inference=True)
                onnx.save(onnx_model, onnx_file_po)
                print(f"Succeed")
            except:
                print(f"Fail")
                onnx_file_po = onnx_file
        else:
            onnx_file_po = onnx_file

        # Try to run in Onnx-runtime
        print(f"{'='* 16} Verify in Onnx-runtime")
        try:
            print(f"Device: {onnxruntime.get_device()}")
            session = onnxruntime.InferenceSession(onnx_file, providers=["CUDAExecutionProvider"])

            for i, tensor in enumerate(session.get_inputs()):
                print(f"Input {i:2d}: {tensor.name}, {tensor.shape}, {tensor.type}")

            for i, tensor in enumerate(session.get_outputs()):
                print(f"Output{i:2d}: {tensor.name}, {tensor.shape}, {tensor.type}")

            output_list = session.run(output_name_list, data)

            print("Output:")
            for name, tensor in zip(output_name_list, output_list):
                print(name, "\n", tensor)

            print(f"Succeed")
        except:
            print(f"Fail")

        # Try to parse to TensorRT
        print(f"{'='* 16} Parse into TensorRT")
        try:
            tw = TRTWrapperV1()
            parser = trt.OnnxParser(tw.network, tw.logger)
            with open(onnx_file_po, "rb") as model:
                parser.parse(model.read())

            for i in range(tw.network.num_inputs):
                input_tensor = tw.network.get_input(i)
                shape = data[input_tensor.name].shape
                min_shape = shape
                opt_shape = shape
                max_shape = [d * 2 if i in dynamic_axes[input_tensor.name] else d for i, d in enumerate(shape)]
                tw.profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)
            tw.config.add_optimization_profile(tw.profile)

            tw.build()

            print("Succeed")
            tw.setup(data)
            tw.infer()
        except:
            print("Fail")
        return

@case_mark
def case_normal():

    class Net(t.nn.Module):

        def __init__(self):
            super(Net, self).__init__()
            self.rep = t.IntTensor([1, 2, 3]).cuda()

        def forward(self, x):
            y = x + 1  #t.repeat_interleave(x, self.rep, dim=0)
            return y

    cto = CheckTorchOperator(Net)

    shape = 3, 4
    data = {"x": np.arange(np.prod(shape), dtype=np.float32).reshape(shape)}
    dynamic_dim_dict = {}
    cto.check(Path(f"model-{sys._getframe().f_code.co_name}.onnx"), data, dynamic_dim_dict)
    return

@case_mark
def case_static_repeat_interlace():

    class Net(t.nn.Module):

        def __init__(self):
            super(Net, self).__init__()
            self.rep = t.IntTensor([1, 2, 3]).cuda()

        def forward(self, x):
            y = t.repeat_interleave(x, self.rep, dim=0)
            return y

    cto = CheckTorchOperator(Net)

    shape = 3, 4
    data = {"x": np.arange(np.prod(shape), dtype=np.float32).reshape(shape)}
    dynamic_dim_dict = {"x": [1]}
    cto.check(Path(f"model-{sys._getframe().f_code.co_name}.onnx"), data, dynamic_dim_dict)
    return

@case_mark
def case_dynamic_repeat_interlace_():

    class Net(t.nn.Module):

        def __init__(self):
            super(Net, self).__init__()

        def forward(self, x, y):
            z = t.repeat_interleave(x, y, dim=0)
            return z

    cto = CheckTorchOperator(Net)

    shape = 3, 4
    data = {"x": np.arange(np.prod(shape), dtype=np.float32).reshape(shape), "y": np.array([1, 2, 3], dtype=np.int32)}
    dynamic_dim_dict = {"x": [1], "y": [0]}
    cto.check(Path(f"model-{sys._getframe().f_code.co_name}.onnx"), data, dynamic_dim_dict)
    return

if __name__ == "__main__":
    # A supported operator
    case_normal()
    # An unsupport operator
    #case_static_repeat_interlace()
    # An unsupport operator
    case_dynamic_repeat_interlace_()

    print("Finish")
