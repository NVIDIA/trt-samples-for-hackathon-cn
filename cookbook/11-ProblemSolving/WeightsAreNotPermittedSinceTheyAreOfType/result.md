### kernel 使用 np.int32
```
[TRT] [E] 3: (Unnamed Layer* 0) [Convolution]: kernel weights are not permitted since they are of type Int32
[TRT] [E] 4: (Unnamed Layer* 0) [Convolution]: count of 2400 weights in kernel, but kernel dimensions (5,5) with 3 input channels, 32 output channels and 1 groups were specified. Expected Weights count is 3 * 5*5 * 32 / 1 = 2400
[TRT] [E] 4: [convolutionNode.cpp::computeOutputExtents::39] Error Code 4: Internal Error ((Unnamed Layer* 0) [Convolution]: number of kernel weights does not match tensor dimensions)
[TRT] [E] 3: (Unnamed Layer* 0) [Convolution]: kernel weights are not permitted since they are of type Int32
[TRT] [E] 4: (Unnamed Layer* 0) [Convolution]: count of 2400 weights in kernel, but kernel dimensions (5,5) with 3 input channels, 32 output channels and 1 groups were specified. Expected Weights count is 3 * 5*5 * 32 / 1 = 2400
[TRT] [E] 4: [network.cpp::validate::2871] Error Code 4: Internal Error (Layer (Unnamed Layer* 0) [Convolution] failed validation)
[TRT] [E] 2: [builder.cpp::buildSerializedNetwork::609] Error Code 2: Internal Error (Assertion enginePtr != nullptr failed. )
```

### bias 使用 np.int32
```
[TRT] [E] 3: (Unnamed Layer* 0) [Convolution]: bias weights are not permitted since they are of type Int32
[TRT] [E] 4: (Unnamed Layer* 0) [Convolution]: count of 32 weights in bias, but number of output maps is 32
[TRT] [E] 4: [convolutionNode.cpp::computeOutputExtents::39] Error Code 4: Internal Error ((Unnamed Layer* 0) [Convolution]: number of kernel weights does not match tensor dimensions)
[TRT] [E] 3: (Unnamed Layer* 0) [Convolution]: bias weights are not permitted since they are of type Int32
[TRT] [E] 4: (Unnamed Layer* 0) [Convolution]: count of 32 weights in bias, but number of output maps is 32
[TRT] [E] 4: [network.cpp::validate::2871] Error Code 4: Internal Error (Layer (Unnamed Layer* 0) [Convolution] failed validation)
[TRT] [E] 2: [builder.cpp::buildSerializedNetwork::609] Error Code 2: Internal Error (Assertion enginePtr != nullptr failed. )
```
