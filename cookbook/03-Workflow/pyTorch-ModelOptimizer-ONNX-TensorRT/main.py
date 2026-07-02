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

from pathlib import Path

import modelopt.onnx.autocast as autocast
import modelopt.onnx.quantization as moq
import modelopt.torch.quantization as mtq
import numpy as np
import onnx
import tensorrt as trt
import torch
import torch.nn.functional as F
from tensorrt_cookbook import TRTWrapperV1, case_mark, cookbook_path, parse_onnx

data_path = cookbook_path("00-Data", "data")
train_data_file = data_path / "TrainData.npz"
test_data_file = data_path / "TestData.npz"
inference_data_file = data_path / "InferenceData.npz"  # Data used for TensorRT inference

model_path = Path(__file__).parent
onnx_file_fp32 = model_path / "model-fp32.onnx"
onnx_file_fp16 = model_path / "model-fp16-autocast.onnx"
onnx_file_int8 = model_path / "model-int8-qat.onnx"
onnx_file_fp8 = model_path / "model-fp8.onnx"
trt_file_fp16 = model_path / "model-fp16-autocast.trt"
trt_file_int8 = model_path / "model-int8-qat.trt"
trt_file_fp8 = model_path / "model-fp8.trt"

batch_size, height, width = 128, 28, 28
n_epoch = 5  # Epochs for the floating-point pre-training
n_epoch_qat = 2  # Epochs for quantization-aware fine-tuning
n_calibration_batch = 10  # Mini-batches used to initialize the INT8 quantizer amax (QAT)
n_calibration = 256  # Number of samples fed to ModelOptimizer for FP8 calibration (PTQ)

class MyData(torch.utils.data.Dataset):
    """Dataset wrapper for the preprocessed MNIST `.npz` files in `00-Data/data`."""

    def __init__(self, b_train=True):
        data = np.load(train_data_file if b_train else test_data_file)
        self.data = data["data"]
        self.label = data["label"]

    def __getitem__(self, index):
        return torch.from_numpy(self.data[index]), torch.from_numpy(self.label[index])

    def __len__(self):
        return len(self.data)

class Net(torch.nn.Module):
    """Simple CNN identical to `00-Data/get-model-part1.py:Net`."""

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, (5, 5), padding=(2, 2), bias=True)
        self.conv2 = torch.nn.Conv2d(32, 64, (5, 5), padding=(2, 2), bias=True)
        self.gemm1 = torch.nn.Linear(64 * 7 * 7, 1024, bias=True)
        self.gemm2 = torch.nn.Linear(1024, 10, bias=True)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.reshape(-1, 64 * 7 * 7)
        x = F.relu(self.gemm1(x))
        y = self.gemm2(x)
        z = torch.argmax(F.softmax(y, dim=1), dim=1)
        return y, z

def train(model, train_loader, test_loader, n=n_epoch, lr=1e-3, tag="FP32"):
    """Train / fine-tune the model for `n` epochs."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    get_loss = torch.nn.CrossEntropyLoss()
    for epoch in range(n):
        model.train()
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()
            optimizer.zero_grad()
            out, _ = model(x)
            get_loss(out, torch.argmax(y, dim=1)).backward()
            optimizer.step()
        print(f"[{tag}] Epoch {epoch:2d}, test acc = {evaluate(model, test_loader):.4f}")

def evaluate(model, test_loader):
    """Evaluate top-1 accuracy on the test set."""
    model.eval()
    n_correct, n_total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()
            _, z = model(x)
            n_correct += (z == torch.argmax(y, dim=1)).sum().item()
            n_total += x.shape[0]
    return n_correct / n_total

def get_trained_model():
    """Build data loaders and pre-train a FP32 model, shared by every case."""
    train_loader = torch.utils.data.DataLoader(MyData(True), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(MyData(False), batch_size=batch_size, shuffle=False)
    model = Net().cuda()
    train(model, train_loader, test_loader, n_epoch, 1e-3, "FP32")
    return model, train_loader, test_loader

def export_onnx(model, onnx_file, opset_version=19, **kwargs):
    """Export a (possibly quantized) pyTorch model to ONNX with a dynamic batch dimension."""
    model.eval()
    dummy = torch.randn(1, 1, height, width, device="cuda")
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            str(onnx_file),
            input_names=["x"],
            output_names=["y", "z"],
            opset_version=opset_version,
            dynamic_axes={
                "x": {
                    0: "nBS"
                },
                "y": {
                    0: "nBS"
                },
                "z": {
                    0: "nBS"
                }
            },
            dynamo=False,  # ModelOptimizer fake-quant modules require the legacy exporter
            **kwargs,
        )
    print(f"Succeed exporting {onnx_file}")

def build_and_infer(onnx_file, trt_file):
    """Parse a reduced-precision ONNX, build a strongly-typed engine and run inference.

    The ONNX graph already carries its own per-tensor data types (from AutoCast Cast nodes
    or from QuantizeLinear / DequantizeLinear pairs), so we build a strongly-typed network:
    TensorRT honors the types in the graph and no `BuilderFlag.FP16/INT8` is needed.
    """
    data = {"x": np.ascontiguousarray(np.load(inference_data_file)["x"])}

    tw = TRTWrapperV1()
    tw.network = tw.builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parse_onnx(onnx_file, tw.logger, tw.network, tw.builder_config)

    input_tensor = tw.network.get_input(0)
    shape = list(data["x"].shape)
    tw.profile.set_shape(input_tensor.name, [1] + shape[1:], shape, [16] + shape[1:])

    tw.build()
    tw.serialize_engine(trt_file)
    tw.setup(data)
    tw.infer()

@case_mark
def case_autocast():
    # Train the floating-point model and export it to a FP32 ONNX file
    model, _, _ = get_trained_model()
    export_onnx(model, onnx_file_fp32)

    # Convert the FP32 ONNX to a mixed FP16/FP32 ONNX with ModelOptimizer AutoCast.
    # AutoCast inserts explicit Cast nodes, keeping numerically-sensitive nodes in FP32.
    # `keep_io_types=True` keeps the network inputs/outputs in FP32 (Cast nodes are
    # inserted right after inputs / before outputs), which is convenient for I/O.
    model_fp16 = autocast.convert_to_mixed_precision(
        onnx_path=str(onnx_file_fp32),
        low_precision_type="fp16",  # "fp16" or "bf16"
        keep_io_types=True,
    )
    onnx.save(model_fp16, str(onnx_file_fp16))
    print(f"Succeed exporting {onnx_file_fp16}")

    build_and_infer(onnx_file_fp16, trt_file_fp16)

@case_mark
def case_qat_train():
    # Pre-train the floating-point model
    model, train_loader, test_loader = get_trained_model()

    # Insert INT8 fake-quantizers and initialize their amax by a calibration pass.
    # `mtq.quantize` replaces Conv/Linear with quantized versions and runs `forward_loop`.
    def forward_loop(m):
        m.eval()
        with torch.no_grad():
            for i, (x, _) in enumerate(train_loader):
                if i >= n_calibration_batch:
                    break
                m(x.cuda())

    model = mtq.quantize(model, mtq.INT8_DEFAULT_CFG, forward_loop)
    mtq.print_quant_summary(model)
    print(f"After PTQ (before QAT) test acc = {evaluate(model, test_loader):.4f}")

    # Quantization-aware fine-tuning: keep training WITH the fake-quantizers in place
    # (a smaller learning rate) so the weights adapt to the quantization noise.
    train(model, train_loader, test_loader, n_epoch_qat, 1e-4, "QAT")

    # Export the QAT model to ONNX. ModelOptimizer emits standard INT8
    # QuantizeLinear / DequantizeLinear pairs that TensorRT reads as explicit quantization.
    export_onnx(
        model,
        onnx_file_int8,
        opset_version=17,
        do_constant_folding=True,  # Fold the calibrated amax buffers into Q/DQ scale constants
        keep_initializers_as_inputs=False,
    )

    build_and_infer(onnx_file_int8, trt_file_int8)

@case_mark
def case_onnx_post_train():
    # Train the floating-point model and export it to a FP32 ONNX file
    model, _, _ = get_trained_model()
    export_onnx(model, onnx_file_fp32)

    # Use ModelOptimizer post-training quantization to insert FP8 (E4M3) Q/DQ nodes.
    # Calibration ranges (amax) are collected by running the ONNX graph on real data.
    calibration_data = np.ascontiguousarray(np.load(train_data_file)["data"][:n_calibration])

    moq.quantize(
        onnx_path=str(onnx_file_fp32),
        quantize_mode="fp8",  # FP8 E4M3 explicit quantization
        calibration_data={"x": calibration_data},
        calibration_method="max",  # FP8 uses absolute-max calibration
        output_path=str(onnx_file_fp8),
    )
    print(f"Succeed exporting {onnx_file_fp8}")

    build_and_infer(onnx_file_fp8, trt_file_fp8)

if __name__ == "__main__":
    # ONNX AutoCast to a mixed FP16/FP32 model
    case_autocast()
    # pyTorch quantization-aware training (QAT) to an INT8 model
    case_qat_train()
    # ONNX post-training quantization (PTQ) to a FP8 model
    case_onnx_post_train()

    print("Finish")
