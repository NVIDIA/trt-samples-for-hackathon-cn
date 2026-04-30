# Data -- Dataset and models needed in this cookbook

+ Preparation work for the data and model files required for the cookbook.

## Get MNIST dataset

+ [Baidu-Netdisk](https://pan.baidu.com/s/14HNCFbySLXndumicFPD-Ww?pwd=gpq2)
+ [HuggingFace](https://huggingface.co/datasets/ylecun/mnist)
+ [Kaggle](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)
+ [LeCun](http://yann.lecun.com/exdb/mnist/) (invalid)
+ [GoogleAPIs](https://storage.googleapis.com/cvdf-datasets/mnist/) (invalid)

### Using Baidu-Netdisk

+ Download the dataset (4 .gz files) and put them as `<PathToCookbook>/00-Data/data-gz/*.gz`

```bash
cd <PathToCookbook>/00-Data
python3 extract-data-gz.py
python3 get-data.py
```

### Using HuggingFace

```bash
cd <PathToCookbook>/00-Data/data-hf
git clone https://huggingface.co/datasets/ylecun/mnist
cd ..
python3 extract-data-hf.py
python3 get-data.py
```

### Using Kaggle

+ Download the dataset (1 .zip files) and put it as `<PathToCookbook>/00-Data/data-kg/archive.zip`

```bash
cd <PathToCookbook>/00-Data
python3 extract-data-kg.py
python3 get-data.py
```

### Output

+ By default, 6000 train images and 500 test images are generated and used for cookbook's data.
+ `data/test/*.jpg`: data for test
+ `data/train/*.jpg`: data for training
+ `data/CalibrationData.npy`: example calibration input data
+ `data/InferenceData.npy`: example inference input data
+ `data/TestData.npz`: example test data
+ `data/TrainData.npz`: example train data

## Build models

```bash
python3 get-model-part1.py  # models created by pytorch
python3 get-model-part2.py  # models created by ONNX
```

+ Output:
  + `model/model-untrained.onnx`: an untrained model
  + `model/model-untrained.npz`: extrnal weight file of model-untrained.onnx (random numbers)
  + `model/model-trained.onnx`: a trained model
  + `model/model-trained.npz`: external weight file of model-trained.onnx
  + `model/model-trained-no-weight.onnx`: model-trained.onnx excludes weight
  + `model/model-trained-no-weight.onnx.weight`: external weight file of model-trained-no-weight.onnx, in ONNX format
  + `model/model-trained-int8-qat.onnx`: an INT8-QAT trained model

  + `model/model-addscalar.onnx`: a model with only a customed "AddScalar" operator
  + `model/model-half-mnist.onnx`: a model with the first half of model-trained.onnx
  + `model/model-if.onnx`: a model with loop and if operator (containing subgraph)
  + `model/model-invalid.onnx`: a model with a devide-zero operator
  + `model/model-labeled.onnx`: a moel with labeled-dimension
  + `model/model-redundant.onnx`: a model with redundant shape operators
  + `model/model-reshape.onnx`: a model with a customeed "MyReshape" operator
  + `model/model-unknown.onnx`: a model with an unknown operator

+ Here we use `gpt2-medium` as another model in some examples, you can choose any other ONNX files to replace this.

```bash
wget https://huggingface.co/openai-community/gpt2-medium/blob/main/onnx/decoder_model.onnx -o model-large.onnx
```
