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

import sys
from datetime import datetime as dt
from pathlib import Path

import numpy as np
import onnx
import torch as t
import torch.nn.functional as F
from torch.autograd import Variable

sys.path.append("/trtcookbook/include")
from utils import (build_custom_op_network_onnx, build_half_mnist_network_onnx, build_invalid_network_onnx, build_redundant_network_onnx, build_unknown_network_onnx, case_mark)

np.random.seed(31193)
t.manual_seed(97)
t.cuda.manual_seed_all(97)
t.backends.cudnn.deterministic = True
batch_size, height, width = 128, 28, 28
n_epoch = 20
data_path = Path("/trtcookbook/00-Data/data/")
model_path = Path("/trtcookbook/00-Data/model/")

train_data_file = data_path / "TrainData.npz"
test_data_file = data_path / "TestData.npz"

torch_model_file = model_path / "model-trained.pth"

onnx_file_untrained = model_path / "model-untrained.onnx"
weight_file_untrained = model_path / "model-untrained.npz"
onnx_file_trained = model_path / "model-trained.onnx"
weight_file_trained = model_path / "model-trained.npz"
onnx_file_trained_no_weight = model_path / "model-trained-no-weight.onnx"
onnx_file_weight = str(onnx_file_trained_no_weight).split("/")[-1] + ".weight"

onnx_file_custom_op = model_path / "model-custom-op.onnx"
onnx_file_half_mnist = model_path / "model-half-mnist.onnx"
onnx_file_int8_qat = model_path / "model-trained-int8-qat.onnx"
onnx_file_invalid = model_path / "model-invalid.onnx"
onnx_file_redundant = model_path / "model-redundant.onnx"
onnx_file_unknown = model_path / "model-unknown.onnx"

class MyData(t.utils.data.Dataset):

    def __init__(self, isTrain=True):
        data = np.load(train_data_file if isTrain else test_data_file)
        self.data = data["data"]
        self.label = data["label"]
        return

    def __getitem__(self, index):
        return t.from_numpy(self.data[index]), t.from_numpy(self.label[index])

    def __len__(self):
        return len(self.data)

train_data_loader = t.utils.data.DataLoader(dataset=MyData(True), batch_size=batch_size, shuffle=False)
test_data_loader = t.utils.data.DataLoader(dataset=MyData(False), batch_size=batch_size, shuffle=False)

class Net(t.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = t.nn.Conv2d(1, 32, (5, 5), padding=(2, 2), bias=True)
        self.conv2 = t.nn.Conv2d(32, 64, (5, 5), padding=(2, 2), bias=True)
        self.gemm1 = t.nn.Linear(64 * 7 * 7, 1024, bias=True)
        self.gemm2 = t.nn.Linear(1024, 10, bias=True)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.reshape(-1, 64 * 7 * 7)
        x = F.relu(self.gemm1(x))
        y = self.gemm2(x)
        z = F.softmax(y, dim=1)
        z = t.argmax(z, dim=1)
        return y, z

@case_mark
def case_normal():

    # Build network in pyTorch
    model = Net().cuda()

    # Export untrained model as ONNX file and weight file
    t.onnx.export( \
        model,
        t.randn(1, 1, height, width, device="cuda"),
        onnx_file_untrained,
        input_names=["x"],
        output_names=["y", "z"],
        do_constant_folding=True,
        verbose=False,
        keep_initializers_as_inputs=False,
        opset_version=17,
        dynamic_axes={"x": {0: "nBS"}, "y": {0: "nBS"}, "z": {0: "nBS"}})
    print(f"Succeed exporting {onnx_file_untrained}")

    weight = {}
    for name, data in model.named_parameters():
        weight[name] = data.detach().cpu().numpy()
    np.savez(weight_file_untrained, **weight)
    print(f"Succeed exporting {weight_file_untrained}")

    # Train the model
    ceLoss = t.nn.CrossEntropyLoss()
    opt = t.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(n_epoch):
        for xTrain, yTrain in train_data_loader:
            xTrain = Variable(xTrain).cuda()
            yTrain = Variable(yTrain).cuda()
            opt.zero_grad()
            y_, z = model(xTrain)
            loss = ceLoss(y_, yTrain)
            loss.backward()
            opt.step()

        with t.no_grad():
            acc = 0
            n = 0
            for xTest, yTest in test_data_loader:
                xTest = Variable(xTest).cuda()
                yTest = Variable(yTest).cuda()
                y_, z = model(xTest)
                acc += t.sum(z == t.matmul(yTest, t.Tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).to("cuda:0"))).cpu().numpy()
                n += xTest.shape[0]
            print("%s, epoch %2d, loss = %f, test acc = %f" % (dt.now(), epoch + 1, loss.data, acc / n))

    # Export trained model as ts model, ONNX file and weight file
    t.save(model, torch_model_file)
    print(f"Succeed exporting {torch_model_file}")

    t.onnx.export( \
        model,
        t.randn(1, 1, height, width, device="cuda"),
        onnx_file_trained,
        input_names=["x"],
        output_names=["y", "z"],
        do_constant_folding=True,
        verbose=False,
        keep_initializers_as_inputs=False,
        opset_version=17,
        dynamic_axes={"x": {0: "nBS"}, "y": {0: "nBS"}, "z": {0: "nBS"}})
    print(f"Succeed exporting {onnx_file_trained}")

    # Save a ONNX file with external weight
    onnx_model = onnx.load(onnx_file_trained, load_external_data=False)
    onnx.save(onnx_model, onnx_file_trained_no_weight, save_as_external_data=True, all_tensors_to_one_file=True, location=onnx_file_weight)
    print(f"Succeed exporting {onnx_file_trained_no_weight}")

    weight = {}
    for name, data in model.named_parameters():
        weight[name] = data.detach().cpu().numpy()
    np.savez(weight_file_trained, **weight)
    print(f"Succeed exporting {weight_file_trained}")

@case_mark
def case_int8qat():
    import pytorch_quantization.calib as calib
    import pytorch_quantization.nn as qnn
    from polygraphy.backend.onnx.loader import fold_constants
    from pytorch_quantization import quant_modules
    from pytorch_quantization.tensor_quant import QuantDescriptor

    calibrator = ["max", "histogram"][0]
    percentileList = [99.9, 99.99, 99.999, 99.9999]
    quant_desc_input = QuantDescriptor(calib_method=calibrator, axis=None)
    qnn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
    qnn.QuantConvTranspose2d.set_default_quant_desc_input(quant_desc_input)
    qnn.QuantLinear.set_default_quant_desc_input(quant_desc_input)
    quant_desc_weight = QuantDescriptor(calib_method=calibrator, axis=None)
    qnn.QuantConv2d.set_default_quant_desc_weight(quant_desc_weight)
    qnn.QuantConvTranspose2d.set_default_quant_desc_weight(quant_desc_weight)
    qnn.QuantLinear.set_default_quant_desc_weight(quant_desc_weight)

    # Build network in pyTorch
    class NetInt8QAT(t.nn.Module):

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
            z = t.argmax(z, dim=1)
            return y, z

    model = NetInt8QAT().cuda()

    # Train the model
    ceLoss = t.nn.CrossEntropyLoss()
    opt = t.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(n_epoch):
        for xTrain, yTrain in train_data_loader:
            xTrain = Variable(xTrain).cuda()
            yTrain = Variable(yTrain).cuda()
            opt.zero_grad()
            y_, z = model(xTrain)
            loss = ceLoss(y_, yTrain)
            loss.backward()
            opt.step()

        with t.no_grad():
            acc = 0
            n = 0
            for xTest, yTest in test_data_loader:
                xTest = Variable(xTest).cuda()
                yTest = Variable(yTest).cuda()
                y_, z = model(xTest)
                acc += t.sum(z == t.matmul(yTest, t.Tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).to("cuda:0"))).cpu().numpy()
                n += xTest.shape[0]
            print("%s, epoch %2d, loss = %f, test acc = %f" % (dt.now(), epoch + 1, loss.data, acc / n))

    # Calibrate the model
    quant_modules.initialize()
    n_calibration_batch = 100

    with t.no_grad():
        # Turn on calibration tool
        for name, module in model.named_modules():
            if isinstance(module, qnn.TensorQuantizer):
                if module._calibrator is not None:
                    module.disable_quant()
                    module.enable_calib()
                else:
                    module.disable()

        for i, (xTrain, yTrain) in enumerate(train_data_loader):
            if i >= n_calibration_batch:
                break
            model(Variable(xTrain).cuda())

        # Turn off calibration tool
        for name, module in model.named_modules():
            if isinstance(module, qnn.TensorQuantizer):
                if module._calibrator is not None:
                    module.enable_quant()
                    module.disable_calib()
                else:
                    module.enable()

        def computeArgMax(model, **kwargs):
            for _, module in model.named_modules():
                if isinstance(module, qnn.TensorQuantizer) and module._calibrator is not None:
                    if isinstance(module._calibrator, calib.MaxCalibrator):
                        module.load_calib_amax()
                    else:
                        module.load_calib_amax(**kwargs)

        if calibrator == "max":
            computeArgMax(model, method="max")
            #modelName = "./model-max-%d.pth" % (n_calibration_batch * train_data_loader.batch_size)

        else:
            for percentile in percentileList:
                computeArgMax(model, method="percentile")
                #modelName = "./model-percentile-%f-%d.pth" % (percentile, n_calibration_batch * train_data_loader.batch_size)

            for method in ["mse", "entropy"]:
                computeArgMax(model, method=method)
                #modelName = "./model-%s-%f.pth" % (method, percentile)

    # Fine-tune the model, not required
    model.cuda()

    for epoch in range(n_epoch):
        for xTrain, yTrain in train_data_loader:
            xTrain = Variable(xTrain).cuda()
            yTrain = Variable(yTrain).cuda()
            opt.zero_grad()
            y_, z = model(xTrain)
            loss = ceLoss(y_, yTrain)
            loss.backward()
            opt.step()

        with t.no_grad():
            acc = 0
            n = 0
            for xTest, yTest in test_data_loader:
                xTest = Variable(xTest).cuda()
                yTest = Variable(yTest).cuda()
                y_, z = model(xTest)
                acc += t.sum(z == t.matmul(yTest, t.Tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).to("cuda:0"))).cpu().numpy()
                n += xTest.shape[0]
            print("%s, epoch %2d, loss = %f, test acc = %f" % (dt.now(), epoch + 1, loss.data, acc / n))

    # Export model as ONNX file
    model.eval()
    qnn.TensorQuantizer.use_fb_fake_quant = True
    t.onnx.export( \
        model, \
        t.randn(1, 1, height, width, device="cuda"), \
        onnx_file_int8_qat, \
        input_names=["x"], \
        output_names=["y", "z"], \
        do_constant_folding=True, \
        verbose=False, \
        keep_initializers_as_inputs=False, \
        opset_version=17, \
        dynamic_axes={"x": {0: "nBS"}, "y": {0: "nBS"}, "z": {0: "nBS"}})
    print(f"Succeed exporting {onnx_file_int8_qat}")

    onnx_model = onnx.load(onnx_file_int8_qat)
    onnx_model = fold_constants(onnx_model, allow_onnxruntime_shape_inference=True)
    onnx.save(onnx_model, onnx_file_int8_qat)

@case_mark
def case_custom_op():
    build_custom_op_network_onnx(onnx_file_custom_op)
    print(f"Succeed exporting {onnx_file_custom_op}")

@case_mark
def case_half_mnist():
    build_half_mnist_network_onnx(onnx_file_half_mnist)
    print(f"Succeed exporting {onnx_file_half_mnist}")

@case_mark
def case_invalid():
    build_invalid_network_onnx(onnx_file_invalid)
    print(f"Succeed exporting {onnx_file_invalid}")

@case_mark
def case_redundant():
    build_redundant_network_onnx(onnx_file_redundant)
    print(f"Succeed exporting {onnx_file_redundant}")

@case_mark
def case_unknown():
    build_unknown_network_onnx(onnx_file_unknown)
    print(f"Succeed exporting {onnx_file_unknown}")

if __name__ == "__main__":
    case_normal()
    case_int8qat()
    case_custom_op()
    case_half_mnist()
    case_invalid()
    case_redundant()
    case_unknown()

    print("Finish")
