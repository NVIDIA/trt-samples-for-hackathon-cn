# WherePlugin
+ Use to implement tf.where
+ Not compatible for TensorRT8, need several edition before using in ensorRT8.
+ input tensor:
    - [0]: (batchSize, n [, m]) float32/float16
+ output tensor:
    - [0]: (batchSize, n [, m]) float32/float16

# Envionment：
+ nvcr.io/nvidia/tensorrt:21.06-py3 (including CUDA 11.3.1, cudnn 8.2.1, TensorRT 7.2.3.4)

# Quick start：
```shell
make

make test
```

# Result:
```
python3 testWherePlugin.py
test 4 5 4
Succeeded building engine!
Check result: True
test 4 20 9
Succeeded building engine!
Check result: True
test 4 200 10
Succeeded building engine!
Check result: True
test finish!

```
