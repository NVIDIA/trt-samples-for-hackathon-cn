# MMTPlugin in MMDNN
+ Use to implement match_matrix_tensor operation in PaddlePaddle.
+ https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/contrib/layers/nn.py#L247
+ Thanks Ryan Jeng for providing the code.
+ Not compatible for TensorRT8, need several edition before using in ensorRT8.
+ input tensor:
    - [0]: (nGroup,xWidth,h)            float16/float32
+ input parameter:
    - [0]: w                            float16/float32 [h,dim_t,h]
    - [1]: h                            int32
    - [2]: dim_t                        int32
+ output:
    - [0]: (nGroup,dim_t,xWidth,yWidth) float16/float32

# Envionment：
+ nvcr.io/nvidia/tensorrt:21.06-py3 (including CUDA 11.3.1, cudnn 8.2.1, TensorRT 7.2.3.4)

# Quick start：
```shell
make

make test
```

# Result:
```
python3 ./testMMTPlugin.py
test [4,5/6,2],dim_t=3 <class 'numpy.float32'>
Succeed building engine!
Check result: True
test [4,5/6,2],dim_t=3 <class 'numpy.float16'>
Succeed building engine!
Check result: True
test finish!

```
