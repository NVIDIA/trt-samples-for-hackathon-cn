# Uncategorized scripts

## build_data_type_md.py and Number/

+ Get the range and layout information of a kind of floating data type.

+ For example, use command below to get information of a data type with 1 bit sign, 8 bit exponent, 23 bit mantissa, i.e. FP32 type.

```python
python3 build_data_type_md.py -s 1 -e 8 -m 23
```

+ Some output files of typical data types are listed in `Number/`.

+ There are also some information of integers data type and a picture of floating data type in `Number/README.md`.

## get_device_info.py

+ Get all properties of the GPU device 0.

```python
python3 get_device_info.py
```

## get_library_info.py

+ Get software version of Driver, CUDA, cuDNN, cuBLAS, TensorRT, pyTorch, TensorFlow, TensorRT(python)

```python
python3 get_version.py
```
