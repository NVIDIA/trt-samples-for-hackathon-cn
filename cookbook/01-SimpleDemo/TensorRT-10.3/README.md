# TensorRT-10

+ We have totally 4 equivalent implementations.

+ For Python workflow, here are two equivalent choices for buffer management, using apckage numpy or torch respectively.

```bash
python3 main_numpy.py

python3 main_torch.py
```

+ For C++ workflow, we need to build an executive file and then run it.

```bash
make clean && make

./main.exe
```

+ The one more example uses some code wrappers, we'd better to get used to it since all the other examples in cookbook is using it.

```bash
python3 main_cookbook_flavor.py
```
