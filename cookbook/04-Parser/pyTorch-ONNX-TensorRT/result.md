Exported graph: graph(%x : Float(*, 1, 28, 28, strides=[784, 784, 28, 1], requires_grad=0, device=cuda:0),
      %conv1.weight : Float(32, 1, 5, 5, strides=[25, 25, 5, 1], requires_grad=1, device=cuda:0),
      %conv1.bias : Float(32, strides=[1], requires_grad=1, device=cuda:0),
      %conv2.weight : Float(64, 32, 5, 5, strides=[800, 25, 5, 1], requires_grad=1, device=cuda:0),
      %conv2.bias : Float(64, strides=[1], requires_grad=1, device=cuda:0),
      %fc1.weight : Float(1024, 3136, strides=[3136, 1], requires_grad=1, device=cuda:0),
      %fc1.bias : Float(1024, strides=[1], requires_grad=1, device=cuda:0),
      %fc2.weight : Float(10, 1024, strides=[1024, 1], requires_grad=1, device=cuda:0),
      %fc2.bias : Float(10, strides=[1], requires_grad=1, device=cuda:0)):
  %onnx::Relu_9 : Float(*, 32, 28, 28, strides=[25088, 784, 28, 1], requires_grad=0, device=cuda:0) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[5, 5], pads=[2, 2, 2, 2], strides=[1, 1], onnx_name="Conv_0"](%x, %conv1.weight, %conv1.bias) # /usr/local/lib/python3.8/dist-packages/torch/nn/modules/conv.py:453:0
  %onnx::MaxPool_10 : Float(*, 32, 28, 28, strides=[25088, 784, 28, 1], requires_grad=1, device=cuda:0) = onnx::Relu[onnx_name="Relu_1"](%onnx::Relu_9) # /usr/local/lib/python3.8/dist-packages/torch/nn/functional.py:1457:0
  %input : Float(*, 32, 14, 14, strides=[6272, 196, 14, 1], requires_grad=1, device=cuda:0) = onnx::MaxPool[ceil_mode=0, kernel_shape=[2, 2], pads=[0, 0, 0, 0], strides=[2, 2], onnx_name="MaxPool_2"](%onnx::MaxPool_10) # /usr/local/lib/python3.8/dist-packages/torch/nn/functional.py:782:0
  %onnx::Relu_12 : Float(*, 64, 14, 14, strides=[12544, 196, 14, 1], requires_grad=0, device=cuda:0) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[5, 5], pads=[2, 2, 2, 2], strides=[1, 1], onnx_name="Conv_3"](%input, %conv2.weight, %conv2.bias) # /usr/local/lib/python3.8/dist-packages/torch/nn/modules/conv.py:453:0
  %onnx::MaxPool_13 : Float(*, 64, 14, 14, strides=[12544, 196, 14, 1], requires_grad=1, device=cuda:0) = onnx::Relu[onnx_name="Relu_4"](%onnx::Relu_12) # /usr/local/lib/python3.8/dist-packages/torch/nn/functional.py:1457:0
  %onnx::Reshape_14 : Float(*, 64, 7, 7, strides=[3136, 49, 7, 1], requires_grad=1, device=cuda:0) = onnx::MaxPool[ceil_mode=0, kernel_shape=[2, 2], pads=[0, 0, 0, 0], strides=[2, 2], onnx_name="MaxPool_5"](%onnx::MaxPool_13) # /usr/local/lib/python3.8/dist-packages/torch/nn/functional.py:782:0
  %onnx::Reshape_15 : Long(2, strides=[1], device=cpu) = onnx::Constant[value=   -1  3136 [ CPULongType{2} ], onnx_name="Constant_6"]() # main.py:71:0
  %onnx::Gemm_16 : Float(*, *, strides=[3136, 1], requires_grad=1, device=cuda:0) = onnx::Reshape[onnx_name="Reshape_7"](%onnx::Reshape_14, %onnx::Reshape_15) # main.py:71:0
  %onnx::Relu_17 : Float(*, 1024, strides=[1024, 1], requires_grad=1, device=cuda:0) = onnx::Gemm[alpha=1., beta=1., transB=1, onnx_name="Gemm_8"](%onnx::Gemm_16, %fc1.weight, %fc1.bias) # /usr/local/lib/python3.8/dist-packages/torch/nn/modules/linear.py:114:0
  %onnx::Gemm_18 : Float(*, 1024, strides=[1024, 1], requires_grad=1, device=cuda:0) = onnx::Relu[onnx_name="Relu_9"](%onnx::Relu_17) # /usr/local/lib/python3.8/dist-packages/torch/nn/functional.py:1457:0
  %y : Float(*, 10, strides=[10, 1], requires_grad=1, device=cuda:0) = onnx::Gemm[alpha=1., beta=1., transB=1, onnx_name="Gemm_10"](%onnx::Gemm_18, %fc2.weight, %fc2.bias) # /usr/local/lib/python3.8/dist-packages/torch/nn/modules/linear.py:114:0
  %onnx::ArgMax_20 : Float(*, 10, strides=[10, 1], requires_grad=1, device=cuda:0) = onnx::Softmax[axis=1, onnx_name="Softmax_11"](%y) # /usr/local/lib/python3.8/dist-packages/torch/nn/functional.py:1834:0
  %z : Long(*, strides=[1], requires_grad=0, device=cuda:0) = onnx::ArgMax[axis=1, keepdims=0, select_last_index=0, onnx_name="ArgMax_12"](%onnx::ArgMax_20) # main.py:75:0
  return (%y, %z)

2022-07-18 15:00:04.047221, epoch  1, loss = 1.365225, test acc = 0.488000
2022-07-18 15:00:04.238958, epoch  2, loss = 0.685623, test acc = 0.880000
2022-07-18 15:00:04.424836, epoch  3, loss = 0.203396, test acc = 0.916000
2022-07-18 15:00:04.611375, epoch  4, loss = 0.093921, test acc = 0.922000
2022-07-18 15:00:04.796362, epoch  5, loss = 0.048991, test acc = 0.940000
2022-07-18 15:00:04.979013, epoch  6, loss = 0.040391, test acc = 0.932000
2022-07-18 15:00:05.164127, epoch  7, loss = 0.063854, test acc = 0.940000
2022-07-18 15:00:05.347811, epoch  8, loss = 0.086976, test acc = 0.954000
2022-07-18 15:00:05.546136, epoch  9, loss = 0.064093, test acc = 0.950000
2022-07-18 15:00:05.774180, epoch 10, loss = 0.050108, test acc = 0.940000
Succeeded building model in pyTorch!
Succeeded converting model into onnx!
Succeeded finding onnx file!
Succeeded parsing .onnx file!
Succeeded building engine!
Binding0-> (-1, 1, 28, 28) (1, 1, 28, 28) DataType.FLOAT
Binding1-> (-1,) (1,) DataType.INT32
inputH0 : (1, 1, 28, 28)
outputH0: (1,)
[8]
Succeeded running model in TensorRT!
