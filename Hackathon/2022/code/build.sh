#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

set -e

mkdir ./data/
python3 createFakeIOData.py

cd ./AttentionPlugin
make clean
make
cp Attention.so ..
cd ..

cd ./LayerNormPlugin
make clean
make
cp LayerNorm.so ..
cd ..

#cd ./MaskPlugin-1output
#make clean
#make
#cp Mask.so ..
#cd ..

#cd ./MaskPlugin-2output
#make clean
#make
#cp Mask.so ..
#cd ..

rm -rf ./*.plan #./*.onnx

polygraphy surgeon sanitize ./encoder.onnx --fold-constant -o ./encoderV2.onnx

python ./encoder-surgeonV6.py
#python ./encoder-surgeonV5.py

polygraphy surgeon sanitize ./encoderV3.onnx --fold-constant -o ./encoderV4.onnx

python ./encoder.py

polygraphy surgeon sanitize ./decoder.onnx --fold-constant -o ./decoderV2.onnx

python ./decoder-surgeonV3.py

polygraphy surgeon sanitize ./decoderV3.onnx --fold-constant -o ./decoderV4.onnx

python ./decoder.py

python testEncoderAndDecoder.py
