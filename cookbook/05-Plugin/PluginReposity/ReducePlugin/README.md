# ReducePlugin
+ Use to calculate reduce-sum or reduce-max on the last 2 axis of the input tensor.
+ Not compatible for TensorRT8, need several edition before using in ensorRT8.
+ input tensor:
    - [0]: (nBatchSize, a, b, ..., l, m, n) float32/float16
+ input parameter:
    - [0]: isSum                            int, calculate reduce-sum or reduce-max
+ output tensor:
    - [0]: (nBatchSize, a, b, ..., l, n) float32/float16

# Envionment：
+ nvcr.io/nvidia/tensorrt:21.06-py3 (including CUDA 11.3.1, cudnn 8.2.1, TensorRT 7.2.3.4)

# Quick start：
```shell
make

make test
```

# Result:
```
python3 ./testReducePlugin.py
test 4 [8, 2, 128] False
Succeeded building engine!
Check result: True
test 4 [8, 5, 128] False
Succeeded building engine!
Check result: True
test 4 [8, 6, 128] False
Succeeded building engine!
Check result: True
test 4 [8, 10, 128] False
Succeeded building engine!
Check result: True
test 4 [8, 15, 128] False
Succeeded building engine!
Check result: True
test 4 [8, 16, 128] False
Succeeded building engine!
Check result: True
test 4 [8, 30, 128] False
Succeeded building engine!
Check result: True
test 4 [8, 82, 128] False
Succeeded building engine!
Check result: True
test 4 [8, 30, 128] True
Succeeded building engine!
Check result: True
test finish!
```
