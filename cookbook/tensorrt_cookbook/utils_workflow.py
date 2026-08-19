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

import tempfile
import traceback
from pathlib import Path

import onnx
import onnxruntime
import torch
from polygraphy.backend.onnx.loader import fold_constants

from .utils_network import parse_onnx

########################################################################################################################
# Run one Torch model through Torch -> ONNX -> ONNX-Runtime -> Polygraphy -> TensorRT and report where it breaks

def get_profile_shapes_from_dynamic(shape, dynamic_shape_spec=None, build_shape=None):
    """Build TensorRT min/opt/max shapes from runtime shape and dynamic-shape spec."""
    shape = [int(d) for d in shape]
    min_shape = shape.copy()
    opt_shape = shape.copy()
    max_shape = shape.copy()

    if isinstance(dynamic_shape_spec, (list, tuple, set)):
        for i in dynamic_shape_spec:
            i = int(i)
            min_shape[i] = 1
            opt_shape[i] = shape[i]
            max_shape[i] = max(shape[i] * 2, shape[i])
        return min_shape, opt_shape, max_shape

    if not isinstance(dynamic_shape_spec, dict):
        return min_shape, opt_shape, max_shape

    for i, dim_spec in dynamic_shape_spec.items():
        i = int(i)

        if isinstance(dim_spec, int):
            min_v = int(dim_spec)
            max_v = int(dim_spec)
        else:
            min_v = getattr(dim_spec, "min", None)
            max_v = getattr(dim_spec, "max", None)
            min_v = 1 if min_v is None else int(min_v)
            max_v = max(shape[i], min_v) if max_v is None else int(max_v)

        if max_v < min_v:
            min_v, max_v = max_v, min_v

        opt_v = min(max(shape[i], min_v), max_v)
        min_shape[i] = min_v
        opt_shape[i] = opt_v
        max_shape[i] = max_v

    if build_shape is not None:
        for i, d in enumerate(build_shape):
            d = int(d)
            if d != -1:
                min_shape[i] = d
                opt_shape[i] = d
                max_shape[i] = d

    return min_shape, opt_shape, max_shape

def check_torch_operator(
    net,
    data: dict | None = None,
    dynamic_shapes: dict | None = None,
    b_polygraphy: bool = True,
    b_onnxruntime: bool = True,
    verbose_error: bool = False,
):
    """Check Torch operator workflow support across ONNX and TensorRT."""
    data = data or {}
    dynamic_shapes = dynamic_shapes or {}

    model = net.cuda() if isinstance(net, torch.nn.Module) else net().cuda()

    status = {
        "Torch": (False, "Not run"),
        "ONNX Export": (False, "Not run"),
        "ONNX Runtime": (None, "Not run" if b_onnxruntime else "Skipped"),
        "Polygraphy sanitize": (None, "Not run" if b_polygraphy else "Skipped"),
        "TensorRT": (False, "Not run"),
    }

    def print_summary():
        print("\n" + "=" * 80)
        print(f"SUPPORT SUMMARY")
        print("=" * 80)
        for framework in ["Torch", "ONNX Export", "ONNX Runtime", "Polygraphy sanitize", "TensorRT"]:
            ok, msg = status[framework]
            mark = "SUPPORTED" if ok is True else ("NOT SUPPORTED" if ok is False else "SKIPPED")
            print(f"[{framework:<19}] {mark:<13} | {msg}")
        print("=" * 80 + "\n")

    def _short_exception(e: Exception) -> str:
        return " ".join(f"{type(e).__name__}: {e}".split())

    # Try block to cover temporary directory cleanup, even if exceptions occur
    try:
        temp_dir_obj = tempfile.TemporaryDirectory(prefix="check_torch_operator_")
        temp_onnx = tempfile.NamedTemporaryFile(dir=temp_dir_obj.name, suffix=".onnx", delete=False)
        temp_onnx.close()
        onnx_file = Path(temp_onnx.name)

        input_name_list = []
        input_tensor_list = []
        for k, v in data.items():
            input_name_list.append(k)
            input_tensor_list.append(torch.from_numpy(v).cuda())
        output_name_list = []

        # Try Torch inference
        try:
            with torch.no_grad():
                output_torch_list = model(*input_tensor_list)
                if isinstance(output_torch_list, torch.Tensor):
                    output_torch_list = [output_torch_list]
            output_name_list = [f"output_{i}" for i in range(len(output_torch_list))]
            status["Torch"] = (True, "Succeeded")
        except Exception as e:
            status["Torch"] = (False, _short_exception(e))
            if verbose_error:
                print(f"[ERROR][Torch] Fail inferring")
                print(traceback.format_exc())
            print_summary()
            return

        # Try ONNX export
        try:
            model.eval()
            torch.onnx.export(
                model,
                tuple(input_tensor_list),
                onnx_file,
                input_names=input_name_list,
                output_names=output_name_list,
                do_constant_folding=True,
                verbose=False,
                keep_initializers_as_inputs=False,
                opset_version=18,
                dynamic_shapes=dynamic_shapes,
            )
            status["ONNX Export"] = (True, f"Succeeded")
        except Exception as e:
            status["ONNX Export"] = (False, _short_exception(e))
            if verbose_error:
                print(f"[ERROR][ONNX Export] Failed exporting to ONNX")
                print(traceback.format_exc())
            print_summary()
            return

        # Try Polygraphy sanitize
        onnx_file_po = onnx_file
        if b_polygraphy:
            try:
                onnx_file_po = Path(str(onnx_file)[:-5] + "-po.onnx")
                onnx_model = onnx.load(onnx_file)
                onnx_model = fold_constants(onnx_model, allow_onnxruntime_shape_inference=True)
                onnx.save(onnx_model, onnx_file_po)
                status["Polygraphy sanitize"] = (True, f"Succeeded")
            except Exception as e:
                status["Polygraphy sanitize"] = (False, _short_exception(e))
                if verbose_error:
                    print(f"[ERROR][Polygraphy] Failed simplifying {onnx_file}")
                    print(traceback.format_exc())
                print_summary()
                return

        # Try ONNX Runtime inference
        if b_onnxruntime:
            try:
                session = onnxruntime.InferenceSession(onnx_file_po, providers=["CPUExecutionProvider"])
                output_ort_list = session.run(output_name_list, data)
                if len(output_ort_list) == len(output_name_list):
                    status["ONNX Runtime"] = (True, "Succeeded")
                else:
                    status["ONNX Runtime"] = (False, "Output number mismatch")
                    print_summary()
                    return
            except Exception as e:
                status["ONNX Runtime"] = (False, _short_exception(e))
                if verbose_error:
                    print(f"[ERROR][ONNX Runtime] Failed verifying ONNX for the operator")
                    print(traceback.format_exc())
                print_summary()
                return

        # Try TensorRT inference
        try:
            from .utils_class import TRTWrapperV2
            tw = TRTWrapperV2(logger="error")
            parse_onnx(onnx_file_po, tw=tw)

            for i in range(tw.network.num_inputs):
                input_tensor = tw.network.get_input(i)
                shape = data[input_tensor.name].shape
                dynamic_shape_spec = dynamic_shapes.get(input_tensor.name, None)
                min_shape, opt_shape, max_shape = get_profile_shapes_from_dynamic(shape, dynamic_shape_spec, input_tensor.shape)
                tw.profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)

            if not tw.build():
                status["TensorRT"] = (False, "Engine build failed")
                print_summary()
                return

            tw.setup(data, b_print_io=False)
            tw.infer(b_print_io=False)
            status["TensorRT"] = (True, "Succeeded")

        except Exception as e:
            status["TensorRT"] = (False, _short_exception(e))
            if verbose_error:
                print(f"[ERROR][TensorRT] Failed parsing the operator")
                print(traceback.format_exc())
            print_summary()
            return

        print_summary()
        return
    finally:
        if temp_dir_obj is not None:
            temp_dir_obj.cleanup()
