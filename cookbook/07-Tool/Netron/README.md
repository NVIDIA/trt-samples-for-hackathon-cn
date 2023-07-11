#

## Steps to run

+ Open model.onnx with Netron

+ A visualization tool of neural network, which supports **ONNX**, TensorFlow Lite, Caffe, Keras, Darknet, PaddlePaddle, ncnn, MNN, Core ML, RKNN, MXNet, MindSpore Lite, TNN, Barracuda, Tengine, CNTK, TensorFlow.js, Caffe2, UFF, and experimental supports for PyTorch, TensorFlow, TorchScript, OpenVINO, Torch, Vitis AI, kmodel, Arm NN, BigDL, Chainer, Deeplearning4j, MediaPipe, MegEngine, ML.NET, scikit-learn

+ Original repository: [Link](https://github.com/lutzroeder/Netron)

+ Installation: please refer to the original repository of the tool.

+ We usually use this tool to check metadata of compute graph, structure of network, node information and tensor information of the model.

+ Here is a ONNX model *model.onnx* in the subdirectory which can be opened with Netron.

+ Here is another interesting tool **onnx-modifier** [Link](https://github.com/ZhangGe6/onnx-modifier). Its appearance is similar to Netron and ONNX compute graph can be edited directly by the user interface. However, it becomes much slower during opening or editing large models, meanwhile the type of graph edition are limited.