# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
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

import re
from pathlib import Path

type_p = (".py", )
type_c = (".c", ".cpp", ".h", ".hpp", ".cu", ".cuh")
type_sh = (".sh", )

pattern_p_start = r"\A(?:(?:#(?!\!).*\n)|(?:#\n)|(?:# \n))+\n*"

pattern_c_start = r"\A/\*[\s\S]*?\*/\n*"

text = """# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
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
"""

text_p = text + "\n"

text_c = "/*\n" + text.replace("#", " *") + " */\n\n"

dry_run = False

def update(f):

    if f.name.endswith(type_p):
        pattern = re.compile(pattern_p_start)
        header = text_p
        shebang = re.compile(r'^(#![^\n]*\n)')
        default_shebang = ""
    elif f.name.endswith(type_c):
        pattern = re.compile(pattern_c_start)
        header = text_c
        shebang = None
        default_shebang = ""
    elif f.name.endswith(type_sh):
        pattern = re.compile(pattern_p_start)
        header = text_p
        shebang = re.compile(r'^(#![^\n]*\n)')
        default_shebang = "#!/bin/bash\n\n"
    else:
        print(f"Skip: {f}")
        return

    with open(f, "r", encoding="utf-8") as file:
        data = file.read()

        prefix = ""
        body = data
        if shebang:
            shebang_match = shebang.match(data)
            if shebang_match:
                prefix = shebang_match.group(1)
                body = data[shebang_match.end():]
            else:
                prefix = default_shebang

        match = pattern.search(body)
        if match:
            print(f"Fix : {f}")
            new_body = pattern.sub(header, body, count=1)
        else:
            print(f"Add : {f}")
            new_body = header + body

        new_data = prefix + new_body

        if not dry_run and new_data != data:
            with open(f, "w", encoding="utf-8") as file:
                file.write(new_data)

def copyright_scan(directory: Path | None = None, depth: int = 100, exclude_list: list[str] | None = None):
    if directory is None:
        directory = Path(__file__).resolve().parent
    if exclude_list is None:
        exclude_list = []

    for f in sorted(directory.glob("*")):
        if f.name in exclude_list:
            continue
        if f.is_dir() and (depth > 0):
            copyright_scan(f, depth - 1, exclude_list)
        elif f.name.endswith(type_p + type_c + type_sh):
            update(f)

if __name__ == '__main__':
    copyright_scan(exclude_list=[".git", ".vscode", "dist", "tensorrt_cookbook.egg-info"])

    print("Finish")
