# TensorRT-10

+ Basic example of using TensorRT-10.

+ We have totally 4 equivalent implementations.

+ For Python workflow, here are two equivalent choices for buffer management, using package `numpy` or `torch` respectively.

```bash
python3 main_numpy.py

python3 main_pytorch.py
```

+ For C++ workflow, we need to build an executable file and then run it.

```bash
make clean && make

./main.exe
```

+ One more example uses code wrappers. It is worth getting used to this style, since most other examples in the cookbook use it.

```bash
python3 main_cookbook_flavor.py
```
