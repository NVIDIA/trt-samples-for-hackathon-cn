# sample TF-UFF-TRT

## environment

| module         | version  |
| :------------- | :------- |
| python         | 3.6.9    |
| numpy          | 1.18.1   |
| CUDA           | 10.0     |
| pycuda         | 2019.1.2 |
| cudnn          | 7.6.4    |
| tensorflow-gpu | 1.13.1   |
| uff            | 0.6.5    |
| tensorrt       | 6.0.1.5  |
| opencv-python  | 4.1.2.30 |

## set up
+ setup TensorFlow model and save it as .pb
```python 
python TF.py
```

+ transform the mdoel from .pb to .uff
```python 
python converToUff.py
```

+ load model and do inference in TensorRT
```python 
python TRT.py
```

## comment
+ Need MNIST dataset as 4 .gz file from [here](http://yann.lecun.com/exdb/mnist/) or [here](https://storage.googleapis.com/cvdf-datasets/mnist/), put them into path *./mnistData*
+ *loadMnistData.py* is for loading data from .gz files above, it is a simplification of original *input_data.py*
+ Input for *TRT.py* is image (*8.png* for example) instead of Mnist data directly
+ you can save the Mnist data as image using the script below, it saves **count** picture(s) into **outputPath** from train-set (**isTrain**\==True) or test-set (**isTrain**\==False)
```python
import loadMnistData
mnist = loadMnistData.MnistData(dataPath, isOneHot=False)
mnist.saveImage(count, outputPath, isTrain) 
```
+ there are TF model, UFF model and TRT engine we generated in path *./backup*
