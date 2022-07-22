### TensorRT 7
```
Input 0: (3, 4, 7) 
 [[[1. 1. 1. 1. 1. 1. 1.]
  [1. 1. 1. 1. 1. 1. 1.]
  [1. 1. 1. 1. 1. 1. 1.]
  [1. 1. 1. 1. 1. 1. 1.]]

 [[1. 1. 1. 1. 1. 1. 1.]
  [1. 1. 1. 1. 1. 1. 1.]
  [1. 1. 1. 1. 1. 1. 1.]
  [1. 1. 1. 1. 1. 1. 1.]]

 [[1. 1. 1. 1. 1. 1. 1.]
  [1. 1. 1. 1. 1. 1. 1.]
  [1. 1. 1. 1. 1. 1. 1.]
  [1. 1. 1. 1. 1. 1. 1.]]]
Output 0: (4, 3, 5) 
 [[[   7.    7.    7.    7.    7.]
  [   7.    7.    7.    7.    7.]
  [   7.    7.    7.    7.    7.]]

 [[  42.   42.   42.   42.   42.]
  [  42.   42.   42.   42.   42.]
  [  42.   42.   42.   42.   42.]]

 [[ 217.  217.  217.  217.  217.]
  [ 217.  217.  217.  217.  217.]
  [ 217.  217.  217.  217.  217.]]

 [[1092. 1092. 1092. 1092. 1092.]
  [1092. 1092. 1092. 1092. 1092.]
  [1092. 1092. 1092. 1092. 1092.]]]
Output 1: (1, 3, 5) 
 [[[1092. 1092. 1092. 1092. 1092.]
  [1092. 1092. 1092. 1092. 1092.]
  [1092. 1092. 1092. 1092. 1092.]]]
```

### TensorRT 8
```
Traceback (most recent call last):
  File "testWili.py", line 25, in <module>
    rnnLayer = network.add_rnn(shuffleLayer.get_output(0), 1, nHidden, nH, trt.RNNOperation.RELU, trt.RNNInputMode.LINEAR, trt.RNNDirection.UNIDIRECTION, fakeWeight, fakeBias)
AttributeError: 'tensorrt.tensorrt.INetworkDefinition' object has no attribute 'add_rnn'
```

### 说明
+ add_rnn 在 TensorRT 7 中被标记为弃用，并在 TensorRT 8 中被移除，需要改用 add_rnn_v2（也在 TensorRT 8 中被标记为弃用）或使用 Loop 结构来实现 RNN

