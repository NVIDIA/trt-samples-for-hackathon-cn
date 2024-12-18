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

import re
from pathlib import Path

copyright_year = "2024"

type_p = (".py", )
type_c = (".c", ".cpp", ".h", ".hpp", ".cu", ".cuh")
type_sh = (".sh", )

pattern_p = "((#\s)|(# \s))?((# .+\s)|(#\s)|(# \s))+\n+"

pattern_c = "/\*\s(( \* .+\s)|( \*\s)|(\s))+ \*/\n+"

pattern_sh = "#!/bin/bash\n\n(" + pattern_p + ")?"

text = """# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""

text_p = text + "\n"

text_c = "/*\n" + text.replace("#", " *") + " */\n\n"

text_sh = "#!/bin/bash\n\n" + text + "\n"

force_update = True
dry_run = False

def update(f):

    if f.name.endswith(type_p):
        pattern = re.compile(pattern_p)
        header = text_p
        shebang = re.compile(r'^(\#\!.*\n)', re.MULTILINE)  # `#!/bin/bash`
    elif f.name.endswith(type_c):
        pattern = re.compile(pattern_c)
        header = text_c
        shebang = None
    elif f.name.endswith(type_sh):
        pattern = re.compile(pattern_sh)
        header = text_sh
        shebang = re.compile(r'^(\#\!.*\n)', re.MULTILINE)  # `#!/bin/bash`
    else:
        print(f"Skip: {f}")
        return

    with open(f, "r+") as file:
        data = file.read()
        match = pattern.search(data)
        if match:
            print(f"Fix : {f}")
            new_data = pattern.sub(header, data, count=1)
        else:
            match = shebang.search(data) if shebang else None
            print(f"Add : {f}")
            if match:
                new_data = shebang.sub(match.group(1) + header, data, count=1)
            else:
                new_data = header + data

        if not dry_run:
            with open(f, "w") as file:
                file.write(new_data)

def copyright_scan(directory: Path = Path("."), depth: int = 100, exclude_list: list = []):
    for f in sorted(directory.glob("*")):
        if f.name in exclude_list:
            continue
        if f.is_dir() and (depth > 0):
            copyright_scan(f, depth - 1, exclude_list)
        elif f.name.endswith(type_p + type_c + type_sh):
            update(f)

if __name__ == '__main__':
    copyright_scan(exclude_list=[".git", ".vscode", "build", "dist", "tensorrt_cookbook.egg-info"])

    print("Finish")
