# 00-MNISTData -- Related dataset

+ Prepare the dataset and models which some examples need.

+ MNIST dataset can be found from [here](http://yann.lecun.com/exdb/mnist/) or [here](https://storage.googleapis.com/cvdf-datasets/mnist/) or the Baidu Netdisk mentioned in root README.md.

+ Download dataset (4 gz files in total) and put them here as `<PathToRepo>/00-MNISTData/*.gz`

+ Run the script below to extract the dataset as .jpg, as well as example inference / calibration input data as .npz.

```shell
python3 extractMnistData.py
```

+ Run the script below to create the ONNX models and corresponding weight files.

+ `model0.onnx` is a untrained model, and `model1.onnx` is a trained model

```shell
python3 getOnnxModel.py
```
