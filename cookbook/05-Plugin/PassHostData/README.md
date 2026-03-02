# Pass Host Data

+ Example of passing a host pointer (pointing to anything like array, structure or even nullptr) into plugin at runtime.

+ This is usually useful in C++ environment, which we can pass a complex structure or even nullptr into plugin without constrian of TensorRT's data type or nullptr check.

+ Steps to run.

```bash
make test
```
