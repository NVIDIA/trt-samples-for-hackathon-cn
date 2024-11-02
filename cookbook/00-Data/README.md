# Data -- Dataset and models needed in this cookbook

+ Preparation work for the data and model files required for the cookbook.

+ MNIST dataset can be found from [here](http://yann.lecun.com/exdb/mnist/) or [here](https://storage.googleapis.com/cvdf-datasets/mnist/) or the Baidu Netdisk mentioned in root README.md.

+ Download dataset (4 gz files) and put them here as `<PathToRepo>/00-Data/*.gz`

+ Run the script below to extract the dataset.

```bash
python3 extract_MNIST.py
```

+ Output:
  + `data/train/*.jpg`: data for training
  + `data/test/*.jpg`: data for test
  + `data/TrainData.npz`: example train data
  + `data/TestData.npz`: example test data
  + `data/InferenceData.npy`: example inference input data
  + `data/CalibrationData.npy`: example calibration input data

+ Run the script below to build the ONNX models and corresponding weight files.

```bash
python3 get_model_part1.py  # models created by pytorch
python3 get_model_part2.py  # models created by ONNX
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
