#
# Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
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
from pathlib import Path

for layer_kind in sorted(Path(".").glob("*Layer")) + sorted(Path(".").glob("*Structure")):
    print(f"==== Start {layer_kind}")
    for script in sorted(Path(layer_kind).glob("*.py")):
        print(f"---- Start {script}")
        result_file = str(script).replace("/", "/result-").replace(".py", ".log")
        os.system(f"python3 {script} > {result_file} 2>&1")
        print(f"---- End   {script}")
    print(f"==== End   {layer_kind}")

print("Finish test all layer")
