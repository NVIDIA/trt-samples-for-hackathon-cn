# Simple demo in TensorRT-10

+ A simple stand-alone example of using TensorRT to build a network and then do inference.

+ We have equivalent implementations in Python and C++.

+ For Python workflow, here are two equivalent choices for buffer management, using apckage numpy or torch respectively.

```shell
python3 main_numpy.py

python3 main_torch.py
```

+ For C++ workflow, we need to build an executive file and then run it.

```shell
make clean && make

./main.exe
```
