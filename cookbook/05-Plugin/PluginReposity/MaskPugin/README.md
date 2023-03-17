# MaskPlugin in ASR (openNMT)
+ Use to create 3 mask from input tensor
+ Not compatible for TensorRT8, need several edition before using in ensorRT8.
+ input tensor:
    - [0]: (nBatchSize,nSequenceLength,nEmbedding)          float16/float32,  input tensor, providing the shape
    - [1]: (nBatchSize,)                                    int32,            count of valid element in each batch
+ input parameter:
    - None
+ output:
    - [0]: (nBatchSize,4,nSequenceLength,nSequenceLength)   float16/float32,  1/0 2D mask
    - [1]: (nBatchSize,4,nSequenceLength,nSequenceLength)   float16/float32,  0/-inf 2D mask
    - [2]: (nBatchSize,nSequenceLength,320)                 float16/float32,  1/0 mask

# Envionment：
+ nvcr.io/nvidia/tensorrt:21.06-py3 (including CUDA 11.3.1, cudnn 8.2.1, TensorRT 7.2.3.4)

# Quick start：
```shell
make

make test
```

# Result:
```
python3 testMaskPlugin.py
Succeeded building engine!
Binding all? Yes
input -> DataType.FLOAT (-1, -1, 560) (4, 8, 560)
input -> DataType.INT32 (-1,) (4,)
output-> DataType.FLOAT (-1, 4, -1, -1) (4, 4, 8, 8)
output-> DataType.FLOAT (-1, 4, -1, -1) (4, 4, 8, 8)
output-> DataType.FLOAT (-1, -1, 320) (4, 8, 320)
input0: (4, 8, 560),SumAbs=8.98300e+03,Var=0.08430,Max=0.99984,Min=0.00002,SAD=5968.28076
	 [0.8369 0.9702 0.4472 0.2846 0.7712 0.2989 0.9763 0.9292 0.2047 0.3451]
input1: (4,),SumAbs=1.00000e+01,Var=1.25000,Max=4.00000,Min=1.00000,SAD=3.00000
	 [1 2 3 4]
output0: (4, 4, 8, 8),SumAbs=1.20000e+02,Var=0.10345,Max=1.00000,Min=0.00000,SAD=79.00000
	 [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
/home/cuan/software/anaconda3/envs/trt8/lib/python3.6/site-packages/numpy/core/fromnumeric.py:87: RuntimeWarning: overflow encountered in reduce
  return ufunc.reduce(obj, axis, dtype, out, **passkwargs)
/home/cuan/software/anaconda3/envs/trt8/lib/python3.6/site-packages/numpy/core/_methods.py:192: RuntimeWarning: overflow encountered in reduce
  arrmean = umr_sum(arr, axis, dtype, keepdims=True)
output1: (4, 4, 8, 8),SumAbs=inf,Var=inf,Max=0.00000,Min=-300000000549775575777803994281145270272.00000,SAD=inf
	 [ 0.e+00 -3.e+38 -3.e+38 -3.e+38 -3.e+38 -3.e+38 -3.e+38 -3.e+38 -3.e+38 -3.e+38]
output2: (4, 8, 320),SumAbs=3.20000e+03,Var=0.21484,Max=1.00000,Min=0.00000,SAD=7.00000
	 [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
diff0: (4, 4, 8, 8),SumAbs=0.00000e+00,Var=0.00000,Max=0.00000,Min=0.00000,SAD=0.00000
	 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
diff1: (4, 4, 8, 8),SumAbs=0.00000e+00,Var=0.00000,Max=0.00000,Min=0.00000,SAD=0.00000
	 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
diff2: (4, 8, 320),SumAbs=0.00000e+00,Var=0.00000,Max=0.00000,Min=0.00000,SAD=0.00000
	 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
Test test<fp32,bs=4,sl=8> finish!
Succeeded building engine!
Binding all? Yes
input -> DataType.HALF (-1, -1, 560) (4, 8, 560)
input -> DataType.INT32 (-1,) (4,)
output-> DataType.HALF (-1, 4, -1, -1) (4, 4, 8, 8)
output-> DataType.HALF (-1, 4, -1, -1) (4, 4, 8, 8)
output-> DataType.HALF (-1, -1, 320) (4, 8, 320)
input0: (4, 8, 560),SumAbs=8.91200e+03,Var=0.08319,Max=1.00000,Min=0.00000,SAD=6020.00000
	 [0.791  0.629  0.3582 0.0696 0.9033 0.832  0.4763 0.8667 0.3672 0.972 ]
input1: (4,),SumAbs=1.00000e+01,Var=1.25000,Max=4.00000,Min=1.00000,SAD=3.00000
	 [1 2 3 4]
output0: (4, 4, 8, 8),SumAbs=1.20000e+02,Var=0.10345,Max=1.00000,Min=0.00000,SAD=79.00000
	 [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
output1: (4, 4, 8, 8),SumAbs=inf,Var=inf,Max=0.00000,Min=-60000.00000,SAD=inf
	 [     0. -60000. -60000. -60000. -60000. -60000. -60000. -60000. -60000. -60000.]
output2: (4, 8, 320),SumAbs=3.20000e+03,Var=0.21484,Max=1.00000,Min=0.00000,SAD=7.00000
	 [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
diff0: (4, 4, 8, 8),SumAbs=0.00000e+00,Var=0.00000,Max=0.00000,Min=0.00000,SAD=0.00000
	 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
diff1: (4, 4, 8, 8),SumAbs=0.00000e+00,Var=0.00000,Max=0.00000,Min=0.00000,SAD=0.00000
	 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
diff2: (4, 8, 320),SumAbs=0.00000e+00,Var=0.00000,Max=0.00000,Min=0.00000,SAD=0.00000
	 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
Test test<fp16,bs=4,sl=8> finish!

```
