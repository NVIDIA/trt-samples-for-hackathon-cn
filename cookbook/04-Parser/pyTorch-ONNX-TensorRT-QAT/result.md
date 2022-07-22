Exported graph: graph(%x : Float(*, 1, 28, 28, strides=[784, 784, 28, 1], requires_grad=0, device=cuda:0),
      %conv1.weight : Float(32, 1, 5, 5, strides=[25, 25, 5, 1], requires_grad=1, device=cuda:0),
      %conv1.bias : Float(32, strides=[1], requires_grad=1, device=cuda:0),
      %conv2.weight : Float(64, 32, 5, 5, strides=[800, 25, 5, 1], requires_grad=1, device=cuda:0),
      %conv2.bias : Float(64, strides=[1], requires_grad=1, device=cuda:0),
      %fc1.weight : Float(1024, 3136, strides=[3136, 1], requires_grad=1, device=cuda:0),
      %fc1.bias : Float(1024, strides=[1], requires_grad=1, device=cuda:0),
      %fc2.weight : Float(10, 1024, strides=[1024, 1], requires_grad=1, device=cuda:0),
      %fc2.bias : Float(10, strides=[1], requires_grad=1, device=cuda:0),
      %onnx::QuantizeLinear_78 : Char(requires_grad=0, device=cpu),
      %onnx::QuantizeLinear_79 : Float(requires_grad=0, device=cpu),
      %onnx::QuantizeLinear_81 : Float(requires_grad=0, device=cpu),
      %onnx::QuantizeLinear_83 : Float(requires_grad=0, device=cpu),
      %onnx::QuantizeLinear_85 : Float(requires_grad=0, device=cpu),
      %onnx::QuantizeLinear_87 : Float(requires_grad=0, device=cpu),
      %onnx::QuantizeLinear_89 : Float(requires_grad=0, device=cpu),
      %onnx::QuantizeLinear_91 : Float(requires_grad=0, device=cpu),
      %onnx::QuantizeLinear_93 : Float(requires_grad=0, device=cpu)):
  %onnx::QuantizeLinear_92 : Char(requires_grad=0, device=cpu) = onnx::Identity[onnx_name="Identity_0"](%onnx::QuantizeLinear_78)
  %onnx::QuantizeLinear_90 : Char(requires_grad=0, device=cpu) = onnx::Identity[onnx_name="Identity_1"](%onnx::QuantizeLinear_78)
  %onnx::QuantizeLinear_88 : Char(requires_grad=0, device=cpu) = onnx::Identity[onnx_name="Identity_2"](%onnx::QuantizeLinear_78)
  %onnx::QuantizeLinear_86 : Char(requires_grad=0, device=cpu) = onnx::Identity[onnx_name="Identity_3"](%onnx::QuantizeLinear_78)
  %onnx::QuantizeLinear_84 : Char(requires_grad=0, device=cpu) = onnx::Identity[onnx_name="Identity_4"](%onnx::QuantizeLinear_78)
  %onnx::QuantizeLinear_82 : Char(requires_grad=0, device=cpu) = onnx::Identity[onnx_name="Identity_5"](%onnx::QuantizeLinear_78)
  %onnx::QuantizeLinear_80 : Char(requires_grad=0, device=cpu) = onnx::Identity[onnx_name="Identity_6"](%onnx::QuantizeLinear_78)
  %onnx::DequantizeLinear_21 : Char(*, 1, 28, 28, device=cpu) = onnx::QuantizeLinear[onnx_name="QuantizeLinear_7"](%x, %onnx::QuantizeLinear_79, %onnx::QuantizeLinear_78) # /usr/local/lib/python3.8/dist-packages/pytorch_quantization/nn/modules/tensor_quantizer.py:284:0
  %onnx::Conv_22 : Float(*, 1, 28, 28, strides=[784, 784, 28, 1], requires_grad=0, device=cuda:0) = onnx::DequantizeLinear[onnx_name="DequantizeLinear_8"](%onnx::DequantizeLinear_21, %onnx::QuantizeLinear_79, %onnx::QuantizeLinear_78) # /usr/local/lib/python3.8/dist-packages/pytorch_quantization/nn/modules/tensor_quantizer.py:284:0
  %onnx::DequantizeLinear_27 : Char(32, 1, 5, 5, strides=[25, 25, 5, 1], device=cpu) = onnx::QuantizeLinear[onnx_name="QuantizeLinear_9"](%conv1.weight, %onnx::QuantizeLinear_81, %onnx::QuantizeLinear_80) # /usr/local/lib/python3.8/dist-packages/pytorch_quantization/nn/modules/tensor_quantizer.py:284:0
  %onnx::Conv_28 : Float(32, 1, 5, 5, strides=[25, 25, 5, 1], requires_grad=1, device=cuda:0) = onnx::DequantizeLinear[onnx_name="DequantizeLinear_10"](%onnx::DequantizeLinear_27, %onnx::QuantizeLinear_81, %onnx::QuantizeLinear_80) # /usr/local/lib/python3.8/dist-packages/pytorch_quantization/nn/modules/tensor_quantizer.py:284:0
  %onnx::Relu_29 : Float(*, 32, 28, 28, strides=[25088, 784, 28, 1], requires_grad=0, device=cuda:0) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[5, 5], pads=[2, 2, 2, 2], strides=[1, 1], onnx_name="Conv_11"](%onnx::Conv_22, %onnx::Conv_28, %conv1.bias) # /usr/local/lib/python3.8/dist-packages/pytorch_quantization/nn/modules/quant_conv.py:128:0
  %onnx::MaxPool_30 : Float(*, 32, 28, 28, strides=[25088, 784, 28, 1], requires_grad=1, device=cuda:0) = onnx::Relu[onnx_name="Relu_12"](%onnx::Relu_29) # /usr/local/lib/python3.8/dist-packages/torch/nn/functional.py:1457:0
  %inputs : Float(*, 32, 14, 14, strides=[6272, 196, 14, 1], requires_grad=1, device=cuda:0) = onnx::MaxPool[ceil_mode=0, kernel_shape=[2, 2], pads=[0, 0, 0, 0], strides=[2, 2], onnx_name="MaxPool_13"](%onnx::MaxPool_30) # /usr/local/lib/python3.8/dist-packages/torch/nn/functional.py:782:0
  %onnx::DequantizeLinear_36 : Char(*, 32, 14, 14, device=cpu) = onnx::QuantizeLinear[onnx_name="QuantizeLinear_14"](%inputs, %onnx::QuantizeLinear_83, %onnx::QuantizeLinear_82) # /usr/local/lib/python3.8/dist-packages/pytorch_quantization/nn/modules/tensor_quantizer.py:284:0
  %onnx::Conv_37 : Float(*, 32, 14, 14, strides=[6272, 196, 14, 1], requires_grad=1, device=cuda:0) = onnx::DequantizeLinear[onnx_name="DequantizeLinear_15"](%onnx::DequantizeLinear_36, %onnx::QuantizeLinear_83, %onnx::QuantizeLinear_82) # /usr/local/lib/python3.8/dist-packages/pytorch_quantization/nn/modules/tensor_quantizer.py:284:0
  %onnx::DequantizeLinear_42 : Char(64, 32, 5, 5, strides=[800, 25, 5, 1], device=cpu) = onnx::QuantizeLinear[onnx_name="QuantizeLinear_16"](%conv2.weight, %onnx::QuantizeLinear_85, %onnx::QuantizeLinear_84) # /usr/local/lib/python3.8/dist-packages/pytorch_quantization/nn/modules/tensor_quantizer.py:284:0
  %onnx::Conv_43 : Float(64, 32, 5, 5, strides=[800, 25, 5, 1], requires_grad=1, device=cuda:0) = onnx::DequantizeLinear[onnx_name="DequantizeLinear_17"](%onnx::DequantizeLinear_42, %onnx::QuantizeLinear_85, %onnx::QuantizeLinear_84) # /usr/local/lib/python3.8/dist-packages/pytorch_quantization/nn/modules/tensor_quantizer.py:284:0
  %onnx::Relu_44 : Float(*, 64, 14, 14, strides=[12544, 196, 14, 1], requires_grad=0, device=cuda:0) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[5, 5], pads=[2, 2, 2, 2], strides=[1, 1], onnx_name="Conv_18"](%onnx::Conv_37, %onnx::Conv_43, %conv2.bias) # /usr/local/lib/python3.8/dist-packages/pytorch_quantization/nn/modules/quant_conv.py:128:0
  %onnx::MaxPool_45 : Float(*, 64, 14, 14, strides=[12544, 196, 14, 1], requires_grad=1, device=cuda:0) = onnx::Relu[onnx_name="Relu_19"](%onnx::Relu_44) # /usr/local/lib/python3.8/dist-packages/torch/nn/functional.py:1457:0
  %onnx::Reshape_46 : Float(*, 64, 7, 7, strides=[3136, 49, 7, 1], requires_grad=1, device=cuda:0) = onnx::MaxPool[ceil_mode=0, kernel_shape=[2, 2], pads=[0, 0, 0, 0], strides=[2, 2], onnx_name="MaxPool_20"](%onnx::MaxPool_45) # /usr/local/lib/python3.8/dist-packages/torch/nn/functional.py:782:0
  %onnx::Reshape_47 : Long(2, strides=[1], device=cpu) = onnx::Constant[value=   -1  3136 [ CPULongType{2} ], onnx_name="Constant_21"]() # main.py:81:0
  %inputs.4 : Float(*, *, strides=[3136, 1], requires_grad=1, device=cuda:0) = onnx::Reshape[onnx_name="Reshape_22"](%onnx::Reshape_46, %onnx::Reshape_47) # main.py:81:0
  %onnx::DequantizeLinear_53 : Char(*, *, device=cpu) = onnx::QuantizeLinear[onnx_name="QuantizeLinear_23"](%inputs.4, %onnx::QuantizeLinear_87, %onnx::QuantizeLinear_86) # /usr/local/lib/python3.8/dist-packages/pytorch_quantization/nn/modules/tensor_quantizer.py:284:0
  %onnx::Gemm_54 : Float(*, *, strides=[3136, 1], requires_grad=1, device=cuda:0) = onnx::DequantizeLinear[onnx_name="DequantizeLinear_24"](%onnx::DequantizeLinear_53, %onnx::QuantizeLinear_87, %onnx::QuantizeLinear_86) # /usr/local/lib/python3.8/dist-packages/pytorch_quantization/nn/modules/tensor_quantizer.py:284:0
  %onnx::DequantizeLinear_59 : Char(1024, 3136, strides=[3136, 1], device=cpu) = onnx::QuantizeLinear[onnx_name="QuantizeLinear_25"](%fc1.weight, %onnx::QuantizeLinear_89, %onnx::QuantizeLinear_88) # /usr/local/lib/python3.8/dist-packages/pytorch_quantization/nn/modules/tensor_quantizer.py:284:0
  %onnx::Gemm_60 : Float(1024, 3136, strides=[3136, 1], requires_grad=1, device=cuda:0) = onnx::DequantizeLinear[onnx_name="DequantizeLinear_26"](%onnx::DequantizeLinear_59, %onnx::QuantizeLinear_89, %onnx::QuantizeLinear_88) # /usr/local/lib/python3.8/dist-packages/pytorch_quantization/nn/modules/tensor_quantizer.py:284:0
  %onnx::Relu_61 : Float(*, 1024, strides=[1024, 1], requires_grad=1, device=cuda:0) = onnx::Gemm[alpha=1., beta=1., transB=1, onnx_name="Gemm_27"](%onnx::Gemm_54, %onnx::Gemm_60, %fc1.bias) # /usr/local/lib/python3.8/dist-packages/pytorch_quantization/nn/modules/quant_linear.py:72:0
  %inputs.8 : Float(*, 1024, strides=[1024, 1], requires_grad=1, device=cuda:0) = onnx::Relu[onnx_name="Relu_28"](%onnx::Relu_61) # /usr/local/lib/python3.8/dist-packages/torch/nn/functional.py:1457:0
  %onnx::DequantizeLinear_67 : Char(*, 1024, device=cpu) = onnx::QuantizeLinear[onnx_name="QuantizeLinear_29"](%inputs.8, %onnx::QuantizeLinear_91, %onnx::QuantizeLinear_90) # /usr/local/lib/python3.8/dist-packages/pytorch_quantization/nn/modules/tensor_quantizer.py:284:0
  %onnx::Gemm_68 : Float(*, 1024, strides=[1024, 1], requires_grad=1, device=cuda:0) = onnx::DequantizeLinear[onnx_name="DequantizeLinear_30"](%onnx::DequantizeLinear_67, %onnx::QuantizeLinear_91, %onnx::QuantizeLinear_90) # /usr/local/lib/python3.8/dist-packages/pytorch_quantization/nn/modules/tensor_quantizer.py:284:0
  %onnx::DequantizeLinear_73 : Char(10, 1024, strides=[1024, 1], device=cpu) = onnx::QuantizeLinear[onnx_name="QuantizeLinear_31"](%fc2.weight, %onnx::QuantizeLinear_93, %onnx::QuantizeLinear_92) # /usr/local/lib/python3.8/dist-packages/pytorch_quantization/nn/modules/tensor_quantizer.py:284:0
  %onnx::Gemm_74 : Float(10, 1024, strides=[1024, 1], requires_grad=1, device=cuda:0) = onnx::DequantizeLinear[onnx_name="DequantizeLinear_32"](%onnx::DequantizeLinear_73, %onnx::QuantizeLinear_93, %onnx::QuantizeLinear_92) # /usr/local/lib/python3.8/dist-packages/pytorch_quantization/nn/modules/tensor_quantizer.py:284:0
  %y : Float(*, 10, strides=[10, 1], requires_grad=1, device=cuda:0) = onnx::Gemm[alpha=1., beta=1., transB=1, onnx_name="Gemm_33"](%onnx::Gemm_68, %onnx::Gemm_74, %fc2.bias) # /usr/local/lib/python3.8/dist-packages/pytorch_quantization/nn/modules/quant_linear.py:72:0
  %onnx::ArgMax_76 : Float(*, 10, strides=[10, 1], requires_grad=1, device=cuda:0) = onnx::Softmax[axis=1, onnx_name="Softmax_34"](%y) # /usr/local/lib/python3.8/dist-packages/torch/nn/functional.py:1834:0
  %z : Long(*, strides=[1], requires_grad=0, device=cuda:0) = onnx::ArgMax[axis=1, keepdims=0, select_last_index=0, onnx_name="ArgMax_35"](%onnx::ArgMax_76) # main.py:85:0
  return (%y, %z)

[I] Loading model: /tmp/tmp_polygraphy_6779a92fdbc3fbf8f75fc581d775182afada92e53d29577a.onnx
[I] Original Model:
    Name: torch_jit | ONNX Opset: 13
    
    ---- 1 Graph Input(s) ----
    {x [dtype=float32, shape=('nBatchSize', 1, 28, 28)]}
    
    ---- 2 Graph Output(s) ----
    {y [dtype=float32, shape=('Gemmy_dim_0', 10)],
     z [dtype=int64, shape=('Gemmy_dim_0',)]}
    
    ---- 17 Initializer(s) ----
    
    ---- 36 Node(s) ----
    
[I] Folding Constants | Pass 1
[I]     Total Nodes | Original:    36, After Folding:    20 |    16 Nodes Folded
[I] Folding Constants | Pass 2
[I]     Total Nodes | Original:    20, After Folding:    20 |     0 Nodes Folded
[I] Saving ONNX model to: ./model-p.onnx
[I] New Model:
    Name: torch_jit | ONNX Opset: 13
    
    ---- 1 Graph Input(s) ----
    {x [dtype=float32, shape=('nBatchSize', 1, 28, 28)]}
    
    ---- 2 Graph Output(s) ----
    {y [dtype=float32, shape=('Gemmy_dim_0', 10)],
     z [dtype=int64, shape=('Gemmy_dim_0',)]}
    
    ---- 17 Initializer(s) ----
    
    ---- 20 Node(s) ----
    
2022-07-19 02:05:31.352550, epoch  1, loss = 1.458799, test acc = 0.452000
2022-07-19 02:05:31.626280, epoch  2, loss = 0.850891, test acc = 0.858000
2022-07-19 02:05:31.892267, epoch  3, loss = 0.186861, test acc = 0.916000
2022-07-19 02:05:32.159117, epoch  4, loss = 0.074890, test acc = 0.934000
2022-07-19 02:05:32.427161, epoch  5, loss = 0.011453, test acc = 0.932000
2022-07-19 02:05:32.697147, epoch  6, loss = 0.022941, test acc = 0.936000
2022-07-19 02:05:32.966036, epoch  7, loss = 0.021655, test acc = 0.934000
2022-07-19 02:05:33.233703, epoch  8, loss = 0.066643, test acc = 0.932000
2022-07-19 02:05:33.500044, epoch  9, loss = 0.006845, test acc = 0.942000
2022-07-19 02:05:33.765901, epoch 10, loss = 0.006764, test acc = 0.930000
Succeeded building model in pyTorch!
Succeeded calibrating model in pyTorch!
2022-07-19 02:05:51.370880, epoch  1, loss = 0.016763, test acc = 0.936000
2022-07-19 02:05:51.624170, epoch  2, loss = 0.003400, test acc = 0.942000
2022-07-19 02:05:51.876195, epoch  3, loss = 0.000732, test acc = 0.940000
2022-07-19 02:05:52.128365, epoch  4, loss = 0.001354, test acc = 0.944000
2022-07-19 02:05:52.382404, epoch  5, loss = 0.001461, test acc = 0.950000
2022-07-19 02:05:52.634999, epoch  6, loss = 0.000490, test acc = 0.952000
2022-07-19 02:05:52.888366, epoch  7, loss = 0.000237, test acc = 0.954000
2022-07-19 02:05:53.141042, epoch  8, loss = 0.000299, test acc = 0.954000
2022-07-19 02:05:53.393638, epoch  9, loss = 0.000179, test acc = 0.954000
2022-07-19 02:05:53.646952, epoch 10, loss = 0.000275, test acc = 0.954000
Succeeded fine tuning model in pyTorch!
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
