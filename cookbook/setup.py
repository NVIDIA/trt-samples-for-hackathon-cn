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

from setuptools import find_packages, setup

description = 'TensorRT-Cookbook: TensorRT-related learning and reference materials, as well as code examples.'

with open("requirements.txt", "r") as f:
    deps = []
    extra_URLs = []
    for line in f.read().splitlines():
        if line.startswith("#") or line.startswith("-r"):
            continue
        # handle -i and --extra-index-url options
        if "-i " in line or "--extra-index-url" in line:
            extra_URLs.append(next(filter(lambda x: x[0] != '-', line.split())))
        else:
            deps.append(line)
# print(deps)
# print(extra_URLs)

version = "0.0.0"
with open("tensorrt_cookbook/version.py", "r") as f:
    for line in f.read().splitlines():
        if line.startswith("__version__"):
            version = line.split(" ")[2][1:-1]
print(version)

# https://setuptools.pypa.io/en/latest/references/keywords.html
setup(
    name='tensorrt-cookbook',
    version=version,
    description=description,
    long_description=description,
    author="NVIDIA Corporation",
    packages=find_packages(),
    # TODO Add windows support for python bindings.
    license="Apache License 2.0",
    keywords="nvidia tensorrt deeplearning inference",
    zip_safe=True,
    install_requires=deps,
    python_requires=">=3.7, <4",
)
