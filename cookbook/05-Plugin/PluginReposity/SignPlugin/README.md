# RevesePlugin in MMDNN
+ Use to get the sign of the input tensor
+ Not compatible for TensorRT8, need several edition before using in ensorRT8.
+ input tensor:
    - [0]: (nBatchSize,n)   float32,
+ output:
    - [0]: (nBatchSize,n)   float32,

# Envionment：
+ nvcr.io/nvidia/tensorrt:21.06-py3 (including CUDA 11.3.1, cudnn 8.2.1, TensorRT 7.2.3.4)

# Quick start：
```shell
make

make test
```

# Result:
```
test finish!
root@dd7a3dd46989:/work/gitlab/tensorrt-plugin/SignPlugin# make test
python3 ./testSignPlugin.py
test 4 16
[TensorRT] INFO: Detected 1 inputs and 1 output network tensors.
Succeeded building engine!
check result: True

test 4 18
[TensorRT] INFO: Detected 1 inputs and 1 output network tensors.
Succeeded building engine!
check result: True

test 4 600
[TensorRT] INFO: Detected 1 inputs and 1 output network tensors.
Succeeded building engine!
check result: True

test finish!
```
