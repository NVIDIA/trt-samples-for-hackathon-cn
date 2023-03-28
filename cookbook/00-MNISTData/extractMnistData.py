#
# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
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

import sys

import loadMnistData

nTrain = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 3000
nTest = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 500

mnist = loadMnistData.MnistData("./", isOneHot=False)
mnist.saveImage(nTrain, "./train/", True)  # 60000 images in total
mnist.saveImage(nTest, "./test/", False)  # 10000 images in total
