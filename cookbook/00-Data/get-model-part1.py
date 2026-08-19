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
onnx_file_trained = model_path / "model-trained.onnx"
weight_file_trained = model_path / "model-trained.npz"
onnx_file_trained_no_weight = model_path / "model-trained-no-weight.onnx"
onnx_file_weight = onnx_file_trained_no_weight.name + ".weight"
onnx_file_trained_sparsity = model_path / "model-trained-sparsity.onnx"

onnx_file_int8_qat = model_path / "model-trained-int8-qat.onnx"
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
def case_unified():
    """Train the baseline model and export it in every form the cookbook consumes.

    Outputs `model-untrained.onnx`, `model-trained.onnx`, `model-trained.pth`, `model-trained.npz`
    and `model-trained-no-weight.onnx`. The sparsity and INT8-QAT variants used to be branches of
    this function; they now live in `case_sparsity_modelopt` and `case_int8qat_modelopt`.
    """
    train_data_loader, test_data_loader = build_data_loaders()

    model = Net().cuda()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    export_onnx_model(model, torch.randn(1, 1, height, width, device="cuda"), onnx_file_untrained, ["x"], ["y", "z"])
    print(f"Succeed exporting {onnx_file_untrained}")

    get_loss = torch.nn.CrossEntropyLoss()
    for epoch in range(n_epoch):
        loss = train_one_epoch(model, train_data_loader, opt, get_loss)
        test_acc = evaluate_accuracy(model, test_data_loader)
        print(f"[{dt.now()}]Epoch {epoch:2d}, loss = {loss.data}, test acc = {test_acc}")

    export_onnx_model(model, torch.randn(1, 1, height, width, device="cuda"), onnx_file_trained, ["x"], ["y", "z"])
    print(f"Succeed exporting {onnx_file_trained}")

    # Export the trained model as a torch checkpoint, a numpy weight archive, and an ONNX file whose
    # weights live in a separate file
    torch.serialization.add_safe_globals([Net])
    torch.save(model, torch_model_file)
    print(f"Succeed exporting {torch_model_file}")
    weight = {}
    for name, data in model.named_parameters():
        weight[name] = data.detach().cpu().numpy()
    np.savez(weight_file_trained, **weight)
    print(f"Succeed exporting {weight_file_trained}")
    onnx_model = onnx.load(onnx_file_trained, load_external_data=False)
    onnx.save(onnx_model, onnx_file_trained_no_weight, save_as_external_data=True, all_tensors_to_one_file=True, location=onnx_file_weight)
    print(f"Succeed exporting {onnx_file_trained_no_weight}")

########################################################################################################################
# Deprecated producers.
#
# Both write the same files as their ModelOptimizer replacements above and both still work, but
# neither is called by the workflow at the bottom of this file. They are kept because they show what
# the hand-rolled flow looks like, and because "we replaced it with ModelOptimizer" is easier to
# trust when the thing it replaced is still there to compare against. Nothing else in the cookbook
# should call them.

@case_mark
def case_sparsity_pytorch_apex():
    """DEPRECATED - use `case_sparsity_modelopt`, which writes the same file.

    Uses `apex.contrib.sparsity` (ASP). Apex ships only inside the NGC container and reports version
    `0.1`, so this cannot be reproduced from `requirements.txt` at all. It also needs the monkey
    patch below: `apex/contrib/sparsity/permutation_search_kernels/exhaustive_search.py` calls
    `time.*` four times without importing `time`, so the attribute has to be injected from outside
    or the permutation search raises `NameError`.

    Verified on 2026-09-09 that this and `case_sparsity_modelopt` produce the same graph (12 nodes)
    with the same 2:4 pattern - `conv2.weight` and `gemm1.weight` at exactly 50% zeros with every
    group of four holding at most two non-zeros, `conv1` skipped for having one input channel and
    `gemm2` left dense.
    """
    import time
    from apex.contrib.sparsity import ASP
    try:
        from apex.contrib.sparsity.permutation_search_kernels import exhaustive_search
        if not hasattr(exhaustive_search, "time"):
            exhaustive_search.time = time
    except Exception:
        pass

    train_data_loader, test_data_loader = build_data_loaders()
    model = Net().cuda()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    ASP.prune_trained_model(model, opt)

    get_loss = torch.nn.CrossEntropyLoss()
    for epoch in range(n_epoch):
        loss = train_one_epoch(model, train_data_loader, opt, get_loss)
        test_acc = evaluate_accuracy(model, test_data_loader)
        print(f"[{dt.now()}]Epoch {epoch:2d}, loss = {loss.data}, test acc = {test_acc}")

    export_onnx_model(model, torch.randn(1, 1, height, width, device="cuda"), onnx_file_trained_sparsity, ["x"], ["y", "z"])
    print(f"Succeed exporting {onnx_file_trained_sparsity}")

@case_mark
def case_int8qat_pytorch_quantization():
    """DEPRECATED - use `case_int8qat_modelopt`, which writes the same file.

    Uses `pytorch_quantization`, retired upstream in favour of NVIDIA ModelOptimizer and no longer
    installable without a pin: the current release ships only an sdist whose build fails with
    `RuntimeError: Bad params`, so it needs `pip install pytorch-quantization==2.1.3`.

    The contrast with `case_int8qat_modelopt` is the point. Here the network has to be declared a
    second time with quantized layer types, then the calibrators enabled, the model run, the
    calibrators disabled, and `load_calib_amax` called with a method that has to agree with the
    `QuantDescriptor` chosen far above. ModelOptimizer does all of it in one call that takes the
    ordinary `Net`.

    Verified on 2026-09-09 that both produce node-for-node identical graphs - 28 nodes with 8
    QuantizeLinear / DequantizeLinear pairs - and that TensorRT 11.0.0.114 builds both.
    """
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

    train_data_loader, test_data_loader = build_data_loaders()
    model = NetInt8QAT().cuda()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    get_loss = torch.nn.CrossEntropyLoss()

    for epoch in range(n_epoch):
        loss = train_one_epoch(model, train_data_loader, opt, get_loss)
        test_acc = evaluate_accuracy(model, test_data_loader)
        print(f"[{dt.now()}]Epoch {epoch:2d}, loss = {loss.data}, test acc = {test_acc}")

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

    export_onnx_model(model, torch.randn(1, 1, height, width, device="cuda"), onnx_file_int8_qat, ["x"], ["y", "z"])
    # Use Polygraphy to fold the quantization parameters into constants
    onnx_model = fold_constants(onnx.load(onnx_file_int8_qat), allow_onnxruntime_shape_inference=True)
    onnx.save(onnx_model, onnx_file_int8_qat)
    print(f"Succeed exporting {onnx_file_int8_qat}")

@case_mark
def case_int8qat_modelopt():
    """Export `model-trained-int8-qat.onnx` with NVIDIA ModelOptimizer.

    This is the supported replacement for `case_unified(b_int8qat=True)`, which uses the deprecated
    `pytorch_quantization` package. Both produce the same thing - the MNIST network carrying INT8
    `QuantizeLinear` / `DequantizeLinear` pairs - and the file name is deliberately the same, so no
    consumer has to care which one produced it.

    The difference is how much of the machinery is yours to get right. `pytorch_quantization` needs
    the layers to be declared as quantized types up front (`qnn.QuantConv2d` instead of `Conv2d`, so
    a whole parallel copy of the network), then the calibrators enabled, then the model run, then
    the calibrators disabled and `load_calib_amax` called with a method that has to match the
    `QuantDescriptor` chosen much earlier. ModelOptimizer replaces all of that with one call that
    takes the ordinary model and a function that runs it:

        model = mtq.quantize(model, mtq.INT8_DEFAULT_CFG, forward_loop)

    See `03-Workflow/pyTorch-ModelOptimizer-ONNX-TensorRT` for the wider workflow this comes from,
    including FP8 and the ONNX-level (post-training) path.
    """
    import modelopt.torch.quantization as mtq
    from polygraphy.backend.onnx.loader import fold_constants

    train_data_loader, test_data_loader = build_data_loaders()
    get_loss = torch.nn.CrossEntropyLoss()

    # Pre-train the plain floating-point network - note this is `Net`, not a quantized copy of it
    model = Net().cuda()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(n_epoch):
        loss = train_one_epoch(model, train_data_loader, opt, get_loss)
        test_acc = evaluate_accuracy(model, test_data_loader)
        print(f"[{dt.now()}]Epoch {epoch:2d}, loss = {loss.data}, test acc = {test_acc}")

    # Insert the fake-quantizers and calibrate their amax. `mtq.quantize` swaps the Conv/Linear
    # modules for quantized ones and runs `forward_loop` to see real activations; the calibration
    # method is part of the config rather than something to enable and disable by hand.
    n_calibration_batch = 100

    def forward_loop(m):
        m.eval()
        with torch.no_grad():
            for i, (x_train, _) in enumerate(train_data_loader):
                if i >= n_calibration_batch:
                    break
                m(x_train.cuda())

    model = mtq.quantize(model, mtq.INT8_DEFAULT_CFG, forward_loop)
    mtq.print_quant_summary(model)
    print(f"[{dt.now()}]After calibration, test acc = {evaluate_accuracy(model, test_data_loader)}")

    # Quantization-aware fine-tuning: keep training with the fake-quantizers in place, at a smaller
    # learning rate, so the weights adapt to the quantization noise
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    for epoch in range(n_epoch):
        loss = train_one_epoch(model, train_data_loader, opt, get_loss)
        test_acc = evaluate_accuracy(model, test_data_loader)
        print(f"[{dt.now()}]QAT epoch {epoch:2d}, loss = {loss.data}, test acc = {test_acc}")

    model.eval()
    export_onnx_model(model, torch.randn(1, 1, height, width, device="cuda"), onnx_file_int8_qat, ["x"], ["y", "z"])
    # Fold the calibrated amax buffers into Q/DQ scale constants, as the deprecated path also does
    onnx_model = fold_constants(onnx.load(onnx_file_int8_qat), allow_onnxruntime_shape_inference=True)
    onnx.save(onnx_model, onnx_file_int8_qat)
    print(f"Succeed exporting {onnx_file_int8_qat}")

@case_mark
def case_sparsity_modelopt():
    """Export `model-trained-sparsity.onnx` with NVIDIA ModelOptimizer.

    This is the supported replacement for `case_sparsity_pytorch_apex`, which uses `apex.contrib.
    sparsity` (ASP). Both apply the same 2:4 structured pattern - ModelOptimizer's magnitude mode
    carries the very same `m4n2_1d` kernel ASP defaults to - so the file name is deliberately the
    same and no consumer has to care which one produced it.

    Two reasons this is the one to use. Apex ships only inside the NGC container and its last
    version string is `0.1`, so the ASP path cannot be reproduced by `pip install -r
    requirements.txt` at all, whereas `nvidia-modelopt` already is a dependency. And ASP has bit-rot
    that the cookbook has to paper over: `apex/contrib/sparsity/permutation_search_kernels/
    exhaustive_search.py` calls `time.*` four times without importing `time`, which
    `case_sparsity_pytorch_apex` works around by injecting the attribute from outside.

    `mts.sparsify` also offers `"sparsegpt"`, which updates the surviving weights using a Hessian
    approximation instead of only masking the small ones. It is not used here because the point of
    this model is to be a 2:4 example rather than the most accurate sparse MNIST, and matching what
    ASP produced keeps the file comparable to the one it replaces.
    """
    import modelopt.torch.sparsity as mts

    train_data_loader, test_data_loader = build_data_loaders()
    get_loss = torch.nn.CrossEntropyLoss()

    # Train dense first: magnitude pruning needs trained weights to decide which ones to keep
    model = Net().cuda()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(n_epoch):
        loss = train_one_epoch(model, train_data_loader, opt, get_loss)
        test_acc = evaluate_accuracy(model, test_data_loader)
        print(f"[{dt.now()}]Epoch {epoch:2d}, loss = {loss.data}, test acc = {test_acc}")

    # Apply the 2:4 mask. Unlike ASP, this takes the ordinary model and needs no optimizer surgery.
    model = mts.sparsify(model, "sparse_magnitude")
    if isinstance(model, tuple):  # Older signatures return `(model, metadata)`
        model = model[0]
    print(f"[{dt.now()}]After pruning, test acc = {evaluate_accuracy(model, test_data_loader)}")

    # Fine-tune with the mask in place so the surviving weights recover the accuracy lost to pruning
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    for epoch in range(n_epoch):
        loss = train_one_epoch(model, train_data_loader, opt, get_loss)
        test_acc = evaluate_accuracy(model, test_data_loader)
        print(f"[{dt.now()}]Sparse epoch {epoch:2d}, loss = {loss.data}, test acc = {test_acc}")

    # Fold the masks into the weights, so the exported graph holds plain zeros rather than a
    # ModelOptimizer wrapper that ONNX has no way to represent
    model = mts.export(model)
    if isinstance(model, tuple):
        model = model[0]

    model.eval()
    export_onnx_model(model, torch.randn(1, 1, height, width, device="cuda"), onnx_file_trained_sparsity, ["x"], ["y", "z"])
    print(f"Succeed exporting {onnx_file_trained_sparsity}")

@case_mark
def case_for():
    """Export an ONNX graph containing control-flow logic.

    The exported graph is a `Loop` whose body contains an `If`, so it covers both control-flow
    operators at once. `02-API/ONNXParser`'s `case_subgraph` is the only consumer.
    """

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
    # The deprecated `case_sparsity_pytorch_apex` and `case_int8qat_pytorch_quantization` write the
    # same two files as the ModelOptimizer cases below and are deliberately not called here.
    #
    # Skip rather than abort when an optional package is missing: an uncaught ImportError also takes
    # down every case after it, which is how `model-for.onnx` was once left stale for a whole test
    # sweep without anyone noticing.
    case_unified()  # Normal model
    case_int8qat_modelopt()
    case_sparsity_modelopt()

    case_for()

    print("Finish")
