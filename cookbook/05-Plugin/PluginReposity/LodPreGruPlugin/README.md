# LodPreGruPlugin in MMDNN
+ Use to preprocess the input data, to generate the input tensor for the subsequent GRU layer, as well as the mask tensor in the network.
+ The operation mainly containsemed (gather) elementwise addition and reverse calculation.
+ The name and shape of the input/output data are shown in the .cu file.
+ Not compatible for TensorRT8, need several edition before using in ensorRT8.

+ input parameter:
    - [0]: datatype     int32,

# Envionment：
+ nvcr.io/nvidia/tensorrt:21.06-py3 (including CUDA 11.3.1, cudnn 8.2.1, TensorRT 7.2.3.4)

# Quick start：
```shell
make

make test
```

# Result:
```
python ./testLodPreGruPlugin.py
test 4 [2, 4, 8, 16] <class 'numpy.float32'>
Succeeded building engine!
All bind: True
Check result 0 : True
Check result 1 : True
Check result 2 : True
Check result 3 : True
Check result 4 : True
Check result 5 : True
Check result 6 : True
Check result 7 : True
Check result 8 : True
Check result 9 : True
Check result 10 : True
Check result 11 : True
Check result 12 : True
Check result 13 : True
Check result 14 : True
Check result 15 : True
Check result 16 : True
Check result 17 : True
test 4 [2, 4, 8, 16] <class 'numpy.float16'>
Succeeded building engine!
All bind: True
Check result 0 : True
Check result 1 : True
Check result 2 : True
Check result 3 : True
Check result 4 : True
Check result 5 : True
Check result 6 : True
Check result 7 : True
Check result 8 : True
Check result 9 : True
Check result 10 : True
Check result 11 : True
Check result 12 : True
Check result 13 : True
Check result 14 : True
Check result 15 : True
Check result 16 : True
Check result 17 : True
test 40 [17 29 40 35 43 23 47 58 32 12 55 54 42 43 66  3 61 32 24 59 34 97 61  5 19 59 75 72 50 92 88 49 31 89 55 85 45 61 41  3] <class 'numpy.float32'>
Succeeded building engine!
All bind: True
Check result 0 : True
Check result 1 : True
Check result 2 : True
Check result 3 : True
Check result 4 : True
Check result 5 : True
Check result 6 : True
Check result 7 : True
Check result 8 : True
Check result 9 : True
Check result 10 : True
Check result 11 : True
Check result 12 : True
Check result 13 : True
Check result 14 : True
Check result 15 : True
Check result 16 : True
Check result 17 : True
test 40 [84 76 31 44 69 97 18 43  8 30  5 61 40 99 45 50 31 15 52 99 75 72 89 42 94 24 62 34 30 32 37  7 48 81 32 21 63 80 45 95] <class 'numpy.float16'>
Succeeded building engine!
All bind: True
Check result 0 : True
Check result 1 : True
Check result 2 : True
Check result 3 : True
Check result 4 : True
Check result 5 : True
Check result 6 : True
Check result 7 : True
Check result 8 : True
Check result 9 : True
Check result 10 : True
Check result 11 : True
Check result 12 : True
Check result 13 : True
Check result 14 : True
Check result 15 : True
Check result 16 : True
Check result 17 : True
test finish!
```

