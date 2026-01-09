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

from collections import OrderedDict

from matplotlib import pyplot as plt

data_type_dict = OrderedDict([
    # ["FP256(E19M236)", [1, 11, 52]],
    # ["FP128(E15M112)", [1, 11, 52]],
    # ["FP64(E11M52)", [1, 11, 52]],
    ["FP32(E11M23)", [1, 8, 23]],
    ["TF32(E8M10)", [1, 8, 10]],
    ["FP16(E5M11)", [1, 5, 11]],
    ["BF16(E8M7)", [1, 8, 7]],
    ["FP8E4M3", [1, 4, 3]],
    ["FP8E5M2", [1, 5, 2]],
    ["FP8E8M0", [0, 8, 0]],
    ["FP6E2M3", [1, 2, 3]],
    ["FP6E3M2", [1, 3, 2]],
    ["FP4E2M1", [1, 2, 1]],
    ["FP4E0M3", [1, 0, 3]],
])

max_exponent_bit = max(dtype[1] for dtype in data_type_dict.values()) + 1
max_mantissa_bit = max(dtype[2] for dtype in data_type_dict.values())

color_box = [[0.75, 0, 0], [0, 0.2, 0.75], [0, 0.6, 0.2]]
transp = 0.25

def my_bar(position, s, e, m):
    height = 1.3
    linewidth = 1.2
    plt.barh(y=position, width=s, height=height, color=color_box[0] + [transp], left=0, linewidth=linewidth, edgecolor=color_box[0])
    for i in range(e):
        plt.barh(y=position, width=1, height=height, color=color_box[1] + [transp], left=max_exponent_bit - e + i, linewidth=linewidth, edgecolor=color_box[1])
    for i in range(m):
        plt.barh(y=position, width=1, height=height, color=color_box[2] + [transp], left=max_exponent_bit + i, linewidth=linewidth, edgecolor=color_box[2])
    plt.text(max_exponent_bit - 0.5, position, str(e), size=12, ha="center", va="center")
    plt.text(max_exponent_bit + 0.5, position, str(m), size=12, ha="center", va="center")

fig = plt.figure(figsize=(20, 8))
ax1 = plt.axes()
ax1.set_axisbelow(True)

for i, (key, value) in enumerate(data_type_dict.items()):
    my_bar(-1 - 2 * i, *value)

plt.ylim(-len(data_type_dict) * 2, 0)
plt.yticks(range(-1, -len(data_type_dict) * 2 - 1, -2), data_type_dict.keys(), color=[0, 0, 0], fontsize=15)
plt.xlim(0, max_exponent_bit + max_mantissa_bit)
plt.xticks([0, 1, max_exponent_bit], [" " * 4 + "Sign", " " * 40 + "Exponent", " " * 190 + "Mantissa"], fontsize=15)
plt.gca().grid(axis="y")

fig.savefig("output/DataType.png")
plt.close()
