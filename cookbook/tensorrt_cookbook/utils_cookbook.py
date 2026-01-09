# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from pathlib import Path

def build_readme(build_script: str = None, outline: str = ""):
    path = Path(build_script).parent.resolve()
    print(f"Build README.md for {path.name}")
    output = f"# {path.name}\n" + outline

    for sub_dir in sorted(path.glob("*/")):
        if not sub_dir.is_dir():
            continue
        sub_readme = sub_dir / "README.md"
        if not sub_readme.exists():
            print(f"{sub_readme} does not exist")
            continue
        with open(sub_readme, 'r') as file:
            lines = file.readlines()
            output += f"\n#{''.join(lines[:3])}"

    with open("README.md", 'w') as f:
        f.write(f"{output}")

    print("Finish")
