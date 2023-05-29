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

import numpy as np
import cv2
from glob import glob

nCalibrationData = 100
nHeight = 28
nWidth = 28
dataPath = "../../../00-MNISTData/"
calibrationDataPath = dataPath + "test/"
inferenceDataFile = dataPath + "8.png"

inferenceData = cv2.imread(inferenceDataFile, cv2.IMREAD_GRAYSCALE).astype(np.float32)

calibrationDataFileList = sorted(glob(calibrationDataPath + "*.jpg"))[:nCalibrationData]
calibrationData = np.empty([nCalibrationData, 1, nHeight, nWidth])
for i in range(nCalibrationData):
    calibrationData[i, 0] = cv2.imread(calibrationDataFileList[i], cv2.IMREAD_GRAYSCALE).astype(np.float32)

dataDictionary = {}
dataDictionary["inferenceData"] = inferenceData
dataDictionary["calibrationData"] = calibrationData
np.savez("data.npz", **dataDictionary)
print("Succeeded creating data for calibration and inference!")