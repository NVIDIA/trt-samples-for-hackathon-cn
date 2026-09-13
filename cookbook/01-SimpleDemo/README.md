# 01-SimpleDemo

+ Simple stand-alone examples of using TensorRT to build a network and do inference.

+ We have 4 equivalent implementations, 3 in Python and 1 in C++.

+ Now only newest TensorRT-10 / 11 is recommended.

+ For Python workflow, here are two equivalent choices for buffer management, using package `numpy` or `torch` respectively.

```bash
python3 main_numpy.py

python3 main_pytorch.py
```

+ One more example uses code wrappers. It is worth getting used to this style, since most other examples in the cookbook use it.

```bash
python3 main_cookbook_flavor.py
```

+ For C++ workflow, we need to build an executable file and then run it.

```bash
make clean && make

./main.exe
```
