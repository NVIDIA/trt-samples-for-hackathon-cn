# OneHotPlugin
+ Use to create one-hot embedding matrix from the input tensor
+ Not compatible for TensorRT8, need several edition before using in ensorRT8.
+ input tensor:
    - [0]: (nBatchSize, n1, n2, ...,nK)         int, K>=0
+ input parameter:
    - [0]: nEmbed                               int, 
    - [1]: isFP16                               int, 1 for True / 0 for False 
+ output tensor:
    - [0]: (nBatchSize, n1, n2, ...,nK, nEmbed) float32/float16

# Envionment：
+ nvcr.io/nvidia/tensorrt:21.06-py3 (including CUDA 11.3.1, cudnn 8.2.1, TensorRT 7.2.3.4)
+ nvcr.io/nvidia/tensorrt:21.09-py3 (including CUDA 11.4.2, cudnn 8.2.4.15, TensorRT 8.0.3)

# Quick start：
```shell
make

make test
```

# Result:
+ TRT7
```
python ./testOneHotPlugin.py
test [32] 8 Fp32
Succeeded building engine!
Check result: True
test [16, 8] 16 Fp32
Succeeded building engine!
Check result: True
test [4, 4, 4] 32 Fp32
Succeeded building engine!
Check result: True
test [4, 4, 4, 4] 1024 Fp32
Succeeded building engine!
Check result: True
test [16, 8] 2048 Fp32
Succeeded building engine!
Check result: True
test [32] 8 Fp16
Succeeded building engine!
Check result: True
test [16, 8] 16 Fp16
Succeeded building engine!
Check result: True
test [4, 4, 4] 32 Fp16
Succeeded building engine!
Check result: True
test [4, 4, 4, 4] 1024 Fp16
Succeeded building engine!
Check result: True
test [16, 8] 2048 Fp16
Succeeded building engine!
Check result: True
test finish!
```

+ TRT8
```
python ./testOneHotPlugin.py
test [1] 8 Fp32
./testOneHotPlugin.py:41: DeprecationWarning: Use build_serialized_network instead.
  return builder.build_engine(network,config)
Succeeded building engine!
Check result: True
test [2, 2] 16 Fp32
Succeeded building engine!
Check result: True
test [4, 4, 4] 32 Fp32
Succeeded building engine!
Check result: True
test [8, 8, 8, 8] 1024 Fp32
Succeeded building engine!
Check result: True
test [4, 4, 4] 2048 Fp32
Succeeded building engine!
Check result: True
test [1] 8 Fp16
Succeeded building engine!
Check result: True
test [2, 2] 16 Fp16
Succeeded building engine!
Check result: True
test [4, 4, 4] 32 Fp16
Succeeded building engine!
Check result: True
test [8, 8, 8, 8] 1024 Fp16
Succeeded building engine!
Check result: True
test [4, 4, 4] 2048 Fp16
Succeeded building engine!
Check result: True
test finish!
```

