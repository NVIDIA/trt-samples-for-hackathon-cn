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

import hashlib
import shutil
from pathlib import Path

import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cudart
from tensorrt_cookbook import case_mark, cookbook_path, datatype_cast, parse_onnx, TRTWrapperV1, print_enumerated_members

shape = [1, 1, 28, 28]
data_path = cookbook_path("00-Data", "data")
model_path = cookbook_path("00-Data", "model")
onnx_file_untrained = model_path / "model-untrained.onnx"
weight_file_trained = model_path / "model-trained.npz"
onnx_file_trained = model_path / "model-trained.onnx"
onnx_file_trained_no_weight = model_path / "model-trained-no-weight.onnx"
onnx_file_weight = model_path / "model-trained-no-weight.onnx.weight"
trt_file = Path("model.trt")
data = {"x": np.load(data_path / "InferenceData.npy")}

@case_mark
def case_dummy_engine():
    tw = TRTWrapperV1()
    tw.builder_config.set_flag(trt.BuilderFlag.REFIT)
    # [Optional] Using STRIP_PLANSTRIP_PLAN, an engine with no weight (need refitting weight later) will be built
    # https://developer.nvidia.com/blog/maximum-performance-and-minimum-footprint-for-ai-apps-with-nvidia-tensorrt-weight-stripped-engines/
    tw.builder_config.set_flag(trt.BuilderFlag.STRIP_PLAN)
    # [Optional] Combinating STRIP_PLANSTRIP_PLAN and REFIT_IDENTICAL, an engine with no weight will be built
    # The performance of the engine is the same as normal engine if and only if refitting the identical weights as build-time, or undefined.
    # This is for a single set of weights with different inference backends, or different GPU architectures.
    tw.builder_config.set_flag(trt.BuilderFlag.REFIT_IDENTICAL)
    # [Optional] Mark some of the weights as refitable, rather than all weights [TODO]: add a example
    tw.builder_config.set_flag(trt.BuilderFlag.REFIT_INDIVIDUAL)

    parse_onnx(onnx_file_untrained, tw.logger, tw.network, tw.builder_config)

    input_tensor = tw.network.get_input(0)
    tw.profile.set_shape(input_tensor.name, shape, [2] + shape[1:], [4] + shape[1:])

    tw.build()
    tw.serialize_engine(trt_file)

    tw.setup(data)
    tw.infer()

@case_mark
def case_set_weights():
    tw = TRTWrapperV1(trt_file=trt_file)
    tw.engine = trt.Runtime(tw.logger).deserialize_cuda_engine(tw.engine_bytes)
    tw.refitter = trt.Refitter(tw.engine, tw.logger)

    w = np.load(weight_file_trained)

    print_enumerated_members(trt.WeightsRole)

    # Two equivalent implementations of refitting
    # 1. Use API set_weights
    tw.refitter.set_weights("node_conv2d", trt.WeightsRole.KERNEL, np.ascontiguousarray(w["conv1.weight"]))  # Use np.ascontiguousarray, BLOODY lesson!
    tw.refitter.set_weights("node_conv2d", trt.WeightsRole.BIAS, np.ascontiguousarray(w["conv1.bias"]))
    tw.refitter.set_weights("node_conv2d_1", trt.WeightsRole.KERNEL, np.ascontiguousarray(w["conv2.weight"]))
    tw.refitter.set_weights("node_conv2d_1", trt.WeightsRole.BIAS, np.ascontiguousarray(w["conv2.bias"]))
    tw.refitter.set_weights("node_linear", trt.WeightsRole.CONSTANT, np.ascontiguousarray(w["gemm1.weight"]))
    tw.refitter.set_weights("node_linear", trt.WeightsRole.CONSTANT, np.ascontiguousarray(w["gemm1.bias"]))
    tw.refitter.set_weights("node_linear_1", trt.WeightsRole.CONSTANT, np.ascontiguousarray(w["gemm2.weight"]))
    tw.refitter.set_weights("node_linear_1", trt.WeightsRole.CONSTANT, np.ascontiguousarray(w["gemm2.bias"]))
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
    for key in w.keys():  # We copy the buffer to GPU to show refitting directly from GPU
        ww = np.ascontiguousarray(w[key])
        n_byte = ww.size * ww.itemsize
        buffer = cudart.cudaMalloc(n_byte)[1]
        buffer_list.append(buffer)
        cudart.cudaMemcpy(buffer, ww.ctypes.data, n_byte, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
        weight = trt.Weights(datatype_cast(ww.dtype, "trt"), buffer, ww.size)
        tw.refitter.set_named_weights(key, weight, trt.TensorLocation.DEVICE)  # Declare the weights is on GPU

    tw.refitter.refit_cuda_engine()

    for buffer in buffer_list:
        cudart.cudaFree(buffer)

    tw.setup(data)
    tw.infer()

@case_mark
def case_from_onnx():
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
    ""
    ""
    # 3. Refit from ONNX file stream with external weight
    with open(onnx_file_trained, "rb") as onnx_model:
        tw.opr.refit_from_bytes(onnx_model.read(), str(onnx_file_weight))
    """
    tw.refitter.refit_cuda_engine()

    tw.setup(data)
    tw.infer()

@case_mark
def case_refit_only_what_changed():
    """Refit only the weights that actually differ, decided by hashing.

    Adapted from `get_refit_weights()` in `exporter.py` and `Engine.refit_from_dict()` in
    `utilities.py` of https://github.com/NVIDIA/Stable-Diffusion-WebUI-TensorRT (Apache-2.0).
    There the question is "the user switched to another LoRA -- which weights changed?", and the
    answer is to hash every ONNX initializer, hash the new state dict, and refit only the
    mismatches. Nothing is copied: that code is TensorRT 9 and Stable-Diffusion-shaped, this is the
    same idea on the cookbook's MNIST weights.

    The point is the host-to-device traffic. On a model of any size, pushing every weight to change
    two of them is gigabytes moved for nothing. Whether you are *allowed* to push only some depends
    on how the engine was built, which the second half of this case measures.
    """
    # An engine refittable with weights that differ from build time. Note this case builds its own
    # engine rather than reusing `trt_file`: that one carries REFIT_IDENTICAL, which promises the
    # refitted weights will be the build-time ones, and this case deliberately breaks that promise.
    tw = TRTWrapperV1()
    tw.builder_config.set_flag(trt.BuilderFlag.REFIT)
    parse_onnx(onnx_file_trained, tw.logger, tw.network, tw.builder_config)
    input_tensor = tw.network.get_input(0)
    tw.profile.set_shape(input_tensor.name, shape, [2] + shape[1:], [4] + shape[1:])
    tw.build()

    engine = trt.Runtime(tw.logger).deserialize_cuda_engine(tw.engine_bytes)
    refitter = trt.Refitter(engine, tw.logger)
    refittable_name_list = list(refitter.get_all_weights())

    # What the engine currently holds, and what we want it to hold. Here the "new checkpoint"
    # touches only the second Linear -- the shape of an adapter that fine-tunes one block.
    w_current = {k: np.ascontiguousarray(v) for k, v in np.load(weight_file_trained).items()}
    rng = np.random.default_rng(31193)
    w_new = {k: v.copy() for k, v in w_current.items()}
    for key in ("gemm2.weight", "gemm2.bias"):
        w_new[key] = np.ascontiguousarray(w_new[key] + rng.standard_normal(w_new[key].shape).astype(np.float32) * 0.01)

    def digest(array):
        # `tobytes()` on a non-contiguous array raises, and a C-order view of the same values can
        # hash differently from an F-order one, so canonicalise before hashing. This is the same
        # `np.ascontiguousarray` discipline `case_set_weights` calls a BLOODY lesson, for a
        # different reason.
        return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()

    digest_current = {k: digest(v) for k, v in w_current.items()}
    changed_name_list = [name for name in refittable_name_list if name in w_new and digest(w_new[name]) != digest_current.get(name)]

    print(f"    Refittable weights in the engine : {len(refittable_name_list)}")
    print(f"    Weights whose hash changed       : {len(changed_name_list)} -> {changed_name_list}")

    for name in changed_name_list:
        refitter.set_named_weights(name, trt.Weights(w_new[name]))

    # A plain REFIT engine already holds every weight, so nothing is "missing" and setting a
    # subset is enough. Measured, not assumed -- this is the half that makes the technique usable.
    missing_name_list = refitter.get_missing_weights()
    print(f"    Missing after setting only the changed ones: {len(missing_name_list)}")
    assert refitter.refit_cuda_engine(), "Refit failed"
    print("    refit_cuda_engine() succeeded having set only the 2 changed weights")

    # Correctness: the selectively refit engine must equal one refit with everything.
    tw.engine = engine
    tw.setup(data, b_print_io=False)
    tw.infer(b_print_io=False)
    output_selective = tw.buffer[tw.tensor_name_list[1]][0].copy()

    refitter_full = trt.Refitter(engine, tw.logger)
    for name in refittable_name_list:
        if name in w_new:
            refitter_full.set_named_weights(name, trt.Weights(w_new[name]))
    assert refitter_full.refit_cuda_engine(), "Full refit failed"
    tw.infer(b_print_io=False)
    output_full = tw.buffer[tw.tensor_name_list[1]][0]
    print(f"    Selective vs full refit, max abs diff: {np.abs(output_selective - output_full).max():.3e}")
    assert np.allclose(output_selective, output_full, atol=1e-5)

    # The same selection against a STRIP_PLAN engine, which is where the technique stops working --
    # and the way it stops working is the finding. Predicted: the refitter reports the six weights
    # we did not set as missing and refuses. Measured: it reports **none** missing and succeeds.
    tw_stripped = TRTWrapperV1()
    tw_stripped.builder_config.set_flag(trt.BuilderFlag.REFIT)
    tw_stripped.builder_config.set_flag(trt.BuilderFlag.STRIP_PLAN)
    parse_onnx(onnx_file_trained, tw_stripped.logger, tw_stripped.network, tw_stripped.builder_config)
    tw_stripped.profile.set_shape(tw_stripped.network.get_input(0).name, shape, [2] + shape[1:], [4] + shape[1:])
    tw_stripped.build()
    print(f"    STRIP_PLAN engine is {tw_stripped.engine_bytes.nbytes / 1e6:.2f} MB vs "
          f"{tw.engine_bytes.nbytes / 1e6:.2f} MB with weights")
    engine_stripped = trt.Runtime(tw_stripped.logger).deserialize_cuda_engine(tw_stripped.engine_bytes)
    refitter_stripped = trt.Refitter(engine_stripped, tw_stripped.logger)
    print(f"    get_missing_weights() before setting anything: {len(refitter_stripped.get_missing_weights())}")
    for name in changed_name_list:
        refitter_stripped.set_named_weights(name, trt.Weights(w_new[name]))
    print(f"    After setting only {len(changed_name_list)}: {len(refitter_stripped.get_missing_weights())} missing, "
          f"refit_cuda_engine() -> {refitter_stripped.refit_cuda_engine()}")

    tw_stripped.engine = engine_stripped
    tw_stripped.setup(data, b_print_io=False)
    tw_stripped.infer(b_print_io=False)
    output_stripped = tw_stripped.buffer[tw_stripped.tensor_name_list[1]][0]
    print(f"    ... and the output is wrong by {np.abs(output_stripped - output_full).max():.3e}")
    print("    So neither get_missing_weights() nor refit_cuda_engine() tells you a stripped engine")
    print("    still lacks its weights. Only comparing the output does. Cross-check the same")
    print("    silent-success behaviour recorded in 07-Tool/trtexec and 04-Feature/WeightStripping.")

    # Hash the dtype the engine stores, not the dtype on disk. Constructed so the answer cannot be
    # argued with: `w` and `w` round-tripped through fp16 are a different fp32 array but the same
    # fp16 array, so an fp32 hash reports a change that an fp16 engine could not represent.
    probe = w_current["gemm1.weight"]
    probe_fp16_roundtrip = np.ascontiguousarray(probe.astype(np.float16).astype(np.float32))
    print("    Same weights, one round-tripped through fp16:")
    print(f"        hashed as fp32 -> differs: {digest(probe) != digest(probe_fp16_roundtrip)}")
    print(f"        hashed as fp16 -> differs: {digest(probe.astype(np.float16)) != digest(probe_fp16_roundtrip.astype(np.float16))}")
    print("    An fp32 hash would refit this weight into an fp16 engine for no change at all.")

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
        tw.refitter.unset_named_weights(key)  # Reverse operation of get_named_weights

    tw.refitter.refit_cuda_engine()
    tw.refitter.refit_cuda_engine_async(0)

    tw.setup(data)
    tw.infer()

if __name__ == "__main__":
    for trt_path in Path(".").glob("*/trt"):
        if trt_path.is_dir():
            shutil.rmtree(trt_path, ignore_errors=True)
        else:
            trt_path.unlink(missing_ok=True)

    case_dummy_engine()
    case_set_weights()
    case_set_weights_gpu()
    case_from_onnx()
    case_refit_only_what_changed()
    #case_other_api()

    print("Finish")
