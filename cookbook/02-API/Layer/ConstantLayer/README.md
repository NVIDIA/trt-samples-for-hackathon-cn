# Constant Layer

+ **np.ascontiguousarray()** must be used while converting np.array to trt.Weights, or the output of the constant layer might be incorrect.

## Case Simple

+ A simple case of using a constant layer.

## Case Weight shape

+ Modify the weight and shape after adding the layer.

## Case Datatype INT4

+ Using constant layer with INT4 data type.
