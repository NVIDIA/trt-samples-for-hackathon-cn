# TensorFlow-TensorRT

+ Train a TF2 CNN model on MNIST data from `00-Data`, then compare inference latency between:
  + TensorFlow embedded TensorRT (`TF-TRT`, `TrtGraphConverterV2`)
  + TensorFlow dynamic compiler (`XLA`, `tf.function(jit_compile=True)`)

+ Steps to run.

```bash
python3 main.py
```
