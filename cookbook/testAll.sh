#!/bin/bash
if false; then
# 00 ---------------------------------------------------------------------------
cd 00-MNISTData
python3 extractMnistData.py
cd ..

echo "Finish 00-MNISTData"

# 01 ---------------------------------------------------------------------------
cd 01-SimpleDemo

#cd TensorRT6
#make test

#cd TensorRT7
#make test

#cd TensorRT8
#make test > result.md
#cd ..

cd TensorRT8.4
make test > result.md
cd ..

cd ..
echo "Finish 01-SimpleDemo"

# 02 ---------------------------------------------------------------------------
cd 02-API

cd Int8-QDQ
python3 main.py > result.md
cd ..

cd PrintNetwork
python3 main.py > result.md
cd ..

cd ..
echo "Finish 02-API"
fi
# 03 ---------------------------------------------------------------------------
cd 03-APIModel

#cd MNISTExample-pyTorch
#python3 main.py > result.md
#cd ..

cd MNISTExample-TensorFlow1
python3 main.py > result.md
cd ..

#cd MNISTExample-TensorFlow2
#python3 main.py > result.md
#cd ..

#cd TensorFlow1
#python3 Convolution.py > result-Convolution.md
#python3 FullyConnected.py > result-FullyConnected.md
#python3 RNN-LSTM.py > result-RNN-LSTM.md
#cd ..

cd ..
echo "Finish 03-APIModel"

# 04 ---------------------------------------------------------------------------