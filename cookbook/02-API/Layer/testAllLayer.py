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

import os
from glob import glob

for layerKind in sorted(glob("./*")):
    for pyFile in sorted(glob(layerKind + "/*.py")):

        resultFile = layerKind + "/result-" + pyFile.split("/")[-1][:-3] + ".log"
        os.system("python3 %s > %s 2>&1" % (pyFile, resultFile))
        print("\tFinish %s" % pyFile)

    print("Finish %s" % layerKind)

print("Finish all layer!")