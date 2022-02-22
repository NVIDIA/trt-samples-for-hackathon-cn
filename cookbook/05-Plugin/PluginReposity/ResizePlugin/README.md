# ResizePlugin
+ Use to resize
+ Not compatible for TensorRT8, need several edition before using in ensorRT8.
+ input tensor:
    - [0]: (c,h,w)          float32
+ input parameter:
    - [0]: hOut             int
    - [1]: wOut             int
+ output tensor:
    - [0]: (c,hOut,wOut)        float32

# Envionment：
+ nvcr.io/nvidia/tensorrt:21.06-py3 (including CUDA 11.3.1, cudnn 8.2.1, TensorRT 7.2.3.4)
+ pyTorch 1.8.0cuda11.1

# Quick start：
```shell
make

make test
```

# Result:
+ python pyTorchTest.py
```
input data:
tensor([[[[7., 5., 6., 4.],
          [4., 2., 5., 3.],
          [3., 9., 9., 7.]]]])
bilinear interpolate with align_corners=False:
[[[[7.    6.667 5.778 5.056 5.5   5.944 5.222 4.333 4.   ]
   [5.8   5.467 4.578 3.9   4.7   5.5   4.822 3.933 3.6  ]
   [4.    3.667 2.778 2.167 3.5   4.833 4.222 3.333 3.   ]
   [3.4   3.867 5.111 6.267 6.8   7.333 6.622 5.733 5.4  ]
   [3.    4.    6.667 9.    9.    9.    8.222 7.333 7.   ]]]]
bilinear interpolate with align_corners=True:
[[[[7.    6.25  5.5   5.125 5.5   5.875 5.5   4.75  4.   ]
   [5.5   4.75  4.    3.75  4.5   5.25  5.    4.25  3.5  ]
   [4.    3.25  2.5   2.375 3.5   4.625 4.5   3.75  3.   ]
   [3.5   4.25  5.    5.688 6.25  6.812 6.5   5.75  5.   ]
   [3.    5.25  7.5   9.    9.    9.    8.5   7.75  7.   ]]]]

```

+ make test
```
python ./testResizePlugin.py
build engine sucessfully.
input=
 [[[ 7.  5.  6.  4.]
  [ 4.  2.  5.  3.]
  [ 3.  9.  9.  7.]]

 [[ 1.  2.  3.  4.]
  [ 5.  6.  7.  8.]
  [ 9. 10. 11. 12.]]]
real output=
 [[[ 7.     6.667  5.778  5.056  5.5    5.944  5.222  4.333  4.   ]
  [ 5.8    5.467  4.578  3.9    4.7    5.5    4.822  3.933  3.6  ]
  [ 4.     3.667  2.778  2.167  3.5    4.833  4.222  3.333  3.   ]
  [ 3.4    3.867  5.111  6.267  6.8    7.333  6.622  5.733  5.4  ]
  [ 3.     4.     6.667  9.     9.     9.     8.222  7.333  7.   ]]

 [[ 1.     1.167  1.611  2.056  2.5    2.944  3.389  3.833  4.   ]
  [ 2.6    2.767  3.211  3.656  4.1    4.544  4.989  5.433  5.6  ]
  [ 5.     5.167  5.611  6.056  6.5    6.944  7.389  7.833  8.   ]
  [ 7.4    7.567  8.011  8.456  8.9    9.344  9.789 10.233 10.4  ]
  [ 9.     9.167  9.611 10.056 10.5   10.944 11.389 11.833 12.   ]]]
test finish!
```

# More information:
+ refer to layer example of resize [link](https://gitlab-master.nvidia.com/wili/tensorrt-cookbook-in-chinese/-/blob/main/LayerExample/COLLECTION.md).
