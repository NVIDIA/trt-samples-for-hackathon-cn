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

from datetime import datetime as dt

import numpy as np
import onnx
import torch
import torch.nn.functional as F
from tensorrt_cookbook import case_mark, cookbook_path, initialize_random_seed

initialize_random_seed()

batch_size, height, width = 128, 28, 28
n_epoch = 20
data_path = cookbook_path("00-Data", "data")
model_path = cookbook_path("00-Data", "model")
model_path.mkdir(parents=True, exist_ok=True)
train_data_file = data_path / "TrainData.npz"
test_data_file = data_path / "TestData.npz"

torch_model_file = model_path / "model-trained.pth"

onnx_file_untrained = model_path / "model-untrained.onnx"
weight_file_untrained = model_path / "model-untrained.npz"
onnx_file_trained = model_path / "model-trained.onnx"
weight_file_trained = model_path / "model-trained.npz"
onnx_file_trained_no_weight = model_path / "model-trained-no-weight.onnx"
onnx_file_weight = onnx_file_trained_no_weight.name + ".weight"
onnx_file_trained_sparsity = model_path / "model-trained-sparsity.onnx"

onnx_file_int8_qat = model_path / "model-trained-int8-qat.onnx"
onnx_file_if = model_path / "model-if.onnx"
onnx_file_for = model_path / "model-for.onnx"

class MyData(torch.utils.data.Dataset):
    """Dataset wrapper for preprocessed MNIST `.npz` files."""

    def __init__(self, b_train=True):
        data = np.load(train_data_file if b_train else test_data_file)
        self.data = data["data"]
        self.label = data["label"]
        return

    def __getitem__(self, index):
        return torch.from_numpy(self.data[index]), torch.from_numpy(self.label[index])

    def __len__(self):
        return len(self.data)

def build_data_loaders():
    """Build train and test data loaders used by all training/export cases."""
    train_data_loader = torch.utils.data.DataLoader(dataset=MyData(True), batch_size=batch_size, shuffle=False)
    test_data_loader = torch.utils.data.DataLoader(dataset=MyData(False), batch_size=batch_size, shuffle=False)
    return train_data_loader, test_data_loader

def labels_to_indices(labels):
    """Convert one-hot labels or class-index labels into class indices."""
    if labels.ndim > 1:
        return torch.argmax(labels, dim=1)
    return labels.to(torch.int64)

def train_one_epoch(model, train_data_loader, optimizer, get_loss):
    """Train the model for one epoch and return the last mini-batch loss."""
    model.train()
    loss = None
    for x_train, y_train in train_data_loader:
        optimizer.zero_grad()
        x_train, y_train = x_train.cuda(), y_train.cuda()
        y, _ = model(x_train)
        loss = get_loss(y, y_train)
        loss.backward()
        optimizer.step()
    return loss

def evaluate_accuracy(model, test_data_loader):
    """Evaluate top-1 accuracy on the test set."""
    model.eval()
    with torch.no_grad():
        acc = 0
        n = 0
        for x_test, y_test in test_data_loader:
            x_test, y_test = x_test.cuda(), y_test.cuda()
            _, z = model(x_test)
            label_index = labels_to_indices(y_test)
            acc += (z == label_index).sum().item()
            n += x_test.shape[0]
    return acc / n

class Net(torch.nn.Module):
    """Simple CNN used for baseline, sparsity, and QAT workflows."""

    def __init__(self):
        super(Net, self).__init__()
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
        z = F.softmax(y, dim=1)
        z = torch.argmax(z, dim=1)
        return y, z

def export_onnx_model(model, model_input, file_name, input_names, output_names):
    """Export model to ONNX with dynamic batch and dynamo->legacy fallback."""
    dynamic_axes = {input_names[0]: {0: "nBS"}}
    if "y" in output_names:
        dynamic_axes["y"] = {0: "nBS"}
    if "z" in output_names:
        dynamic_axes["z"] = {0: "nBS"}

    export_kwargs = dict(
        input_names=input_names,
        output_names=output_names,
        do_constant_folding=True,
        verbose=False,
        keep_initializers_as_inputs=False,
        opset_version=18,
        external_data=False,
    )

    if hasattr(model, "eval"):
        model.eval()

    if hasattr(torch, "export") and hasattr(torch.export, "Dim"):
        export_kwargs["dynamo"] = True
        export_kwargs["dynamic_shapes"] = {input_names[0]: {0: torch.export.Dim("nBS")}}
        try:
            torch.onnx.export(model, model_input, file_name, **export_kwargs)
            return
        except Exception as export_error:
            print(f"[export_onnx_model] dynamo export failed, fallback to legacy exporter: {type(export_error).__name__}: {export_error}")
    # Fallback to old export method
    export_kwargs.pop("dynamo", None)
    export_kwargs.pop("dynamic_shapes", None)
    export_kwargs["dynamo"] = False
    export_kwargs["dynamic_axes"] = dynamic_axes
    torch.onnx.export(model, model_input, file_name, **export_kwargs)

@case_mark
def case_unified(b_sparsity: bool = False, b_int8qat: bool = False):
    """Train and export baseline/sparsity/INT8-QAT model variants."""
    if b_sparsity and b_int8qat:
        raise ValueError("b_sparsity and b_int8qat cannot be True at the same time")

    train_data_loader, test_data_loader = build_data_loaders()

    if b_sparsity:
        # Get sparse model utility
        import time
        from apex.contrib.sparsity import ASP
        try:
            from apex.contrib.sparsity.permutation_search_kernels import exhaustive_search
            if not hasattr(exhaustive_search, "time"):
                exhaustive_search.time = time
        except Exception:
            pass
        model = Net().cuda()
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        ASP.prune_trained_model(model, opt)
        local_onnx_file_trained = onnx_file_trained_sparsity
    elif b_int8qat:
        # Get Int8QAT model utility
        import pytorch_quantization.calib as calib
        import pytorch_quantization.nn as qnn
        from polygraphy.backend.onnx.loader import fold_constants
        from pytorch_quantization import quant_modules
        from pytorch_quantization.tensor_quant import QuantDescriptor

        calibrator = ["max", "histogram"][0]
        percentile_list = [99.9, 99.99, 99.999, 99.9999]
        quant_desc_input = QuantDescriptor(calib_method=calibrator, axis=None)
        qnn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
        qnn.QuantConvTranspose2d.set_default_quant_desc_input(quant_desc_input)
        qnn.QuantLinear.set_default_quant_desc_input(quant_desc_input)
        quant_desc_weight = QuantDescriptor(calib_method=calibrator, axis=None)
        qnn.QuantConv2d.set_default_quant_desc_weight(quant_desc_weight)
        qnn.QuantConvTranspose2d.set_default_quant_desc_weight(quant_desc_weight)
        qnn.QuantLinear.set_default_quant_desc_weight(quant_desc_weight)

        class NetInt8QAT(torch.nn.Module):

            def __init__(self):
                super(NetInt8QAT, self).__init__()
                self.conv1 = qnn.QuantConv2d(1, 32, (5, 5), padding=(2, 2), bias=True)
                self.conv2 = qnn.QuantConv2d(32, 64, (5, 5), padding=(2, 2), bias=True)
                self.gemm1 = qnn.QuantLinear(64 * 7 * 7, 1024, bias=True)
                self.gemm2 = qnn.QuantLinear(1024, 10, bias=True)

            def forward(self, x):
                x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
                x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
                x = x.reshape(-1, 64 * 7 * 7)
                x = F.relu(self.gemm1(x))
                y = self.gemm2(x)
                z = F.softmax(y, dim=1)
                z = torch.argmax(z, dim=1)
                return y, z

        model = NetInt8QAT().cuda()
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        local_onnx_file_trained = onnx_file_int8_qat
    else:
        # For normal case, export untrained model as ONNX and weight file
        model = Net().cuda()
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        export_onnx_model(model, torch.randn(1, 1, height, width, device="cuda"), onnx_file_untrained, ["x"], ["y", "z"])
        print(f"Succeed exporting {onnx_file_untrained}")
        weight = {}
        for name, data in model.named_parameters():
            weight[name] = data.detach().cpu().numpy()
        np.savez(weight_file_untrained, **weight)
        print(f"Succeed exporting {weight_file_untrained}")
        local_onnx_file_trained = onnx_file_trained
        local_weight_file_trained = weight_file_trained

    get_loss = torch.nn.CrossEntropyLoss()

    for epoch in range(n_epoch):
        loss = train_one_epoch(model, train_data_loader, opt, get_loss)
        test_acc = evaluate_accuracy(model, test_data_loader)
        print(f"[{dt.now()}]Epoch {epoch:2d}, loss = {loss.data}, test acc = {test_acc}")

    if b_int8qat:
        # Calibrate the model
        quant_modules.initialize()
        n_calibration_batch = 100

        with torch.no_grad():
            # Turn on calibration tool
            for _, module in model.named_modules():
                if isinstance(module, qnn.TensorQuantizer):
                    if module._calibrator is not None:
                        module.disable_quant()
                        module.enable_calib()
                    else:
                        module.disable()

            for i, (x_train, _) in enumerate(train_data_loader):
                if i >= n_calibration_batch:
                    break
                model(x_train.cuda())

            # Turn off calibration tool
            for _, module in model.named_modules():
                if isinstance(module, qnn.TensorQuantizer):
                    if module._calibrator is not None:
                        module.enable_quant()
                        module.disable_calib()
                    else:
                        module.enable()

            def compute_argmax(model, **kwargs):
                for _, module in model.named_modules():
                    if isinstance(module, qnn.TensorQuantizer) and module._calibrator is not None:
                        if isinstance(module._calibrator, calib.MaxCalibrator):
                            module.load_calib_amax()
                        else:
                            module.load_calib_amax(**kwargs)

            if calibrator == "max":
                compute_argmax(model, method="max")
            else:
                for _ in percentile_list:
                    compute_argmax(model, method="percentile")
                for method in ["mse", "entropy"]:
                    compute_argmax(model, method=method)

        # Fine-tune the model, not required
        model.cuda()
        for epoch in range(n_epoch):
            loss = train_one_epoch(model, train_data_loader, opt, get_loss)
            test_acc = evaluate_accuracy(model, test_data_loader)
            print(f"[{dt.now()}]Epoch {epoch:2d}, loss = {loss.data}, test acc = {test_acc}")

        model.eval()
        qnn.TensorQuantizer.use_fb_fake_quant = True

    # Export trained model ONNX file
    export_onnx_model(model, torch.randn(1, 1, height, width, device="cuda"), local_onnx_file_trained, ["x"], ["y", "z"])
    print(f"Succeed exporting {local_onnx_file_trained}")
    if b_int8qat:
        # Use Polygraphy to fold the quantization parameters into constants
        onnx_model = onnx.load(local_onnx_file_trained)
        onnx_model = fold_constants(onnx_model, allow_onnxruntime_shape_inference=True)
        onnx.save(onnx_model, local_onnx_file_trained)
    elif b_sparsity:
        pass
    else:
        # For normal case, export trained model as ts model, file, and weight-separate ONNX file
        torch.serialization.add_safe_globals([Net])
        torch.save(model, torch_model_file)
        print(f"Succeed exporting {torch_model_file}")
        weight = {}
        for name, data in model.named_parameters():
            weight[name] = data.detach().cpu().numpy()
        np.savez(local_weight_file_trained, **weight)
        print(f"Succeed exporting {local_weight_file_trained}")
        onnx_model = onnx.load(local_onnx_file_trained, load_external_data=False)
        onnx.save(onnx_model, onnx_file_trained_no_weight, save_as_external_data=True, all_tensors_to_one_file=True, location=onnx_file_weight)
        print(f"Succeed exporting {onnx_file_trained_no_weight}")

@case_mark
def case_if():
    """Export an ONNX graph containing control-flow If logic."""

    @torch.jit.script
    def sum_if(x):
        y = torch.zeros(1, dtype=torch.int32)
        for c in x:
            if c % 2 == 0:
                y += c
        return y

    class CaseIf(torch.nn.Module):

        def __init__(self):
            super().__init__()

        def forward(self, x):
            return sum_if(x)

    export_onnx_model(CaseIf(), torch.zeros(4, dtype=torch.int32), onnx_file_if, ["x"], ["y"])
    print(f"Succeed exporting {onnx_file_if}")

@case_mark
def case_for():
    """Export an ONNX graph containing control-flow For logic."""

    @torch.jit.script
    def sum_for(x):
        y = torch.zeros_like(x, dtype=torch.int32)
        for i, c in enumerate(x):
            if c % 2 == 0:
                y[i] += c
        return y

    class CaseFor(torch.nn.Module):

        def __init__(self):
            super().__init__()

        def forward(self, x):
            return sum_for(x)

    export_onnx_model(CaseFor(), torch.zeros(4, dtype=torch.int32), onnx_file_for, ["x"], ["y"])
    print(f"Succeed exporting {onnx_file_for}")

if __name__ == "__main__":
    case_unified()  # Normal model
    case_unified(b_sparsity=True)  # Sparsity model
    case_unified(b_int8qat=True)  # Int8QAT model
    case_if()
    case_for()

    print("Finish")
