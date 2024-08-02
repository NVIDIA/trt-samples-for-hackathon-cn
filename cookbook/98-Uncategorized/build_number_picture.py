#
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
    ["FP64(e11m52)", [1, 11, 52]],
    ["FP32(e8m23)", [1, 8, 23]],
    ["TF32(e8m10)", [1, 8, 10]],
    ["BF16(e8m7)", [1, 8, 7]],
    ["FP16(e5m11)", [1, 5, 11]],
    ["FP8e4m3", [1, 4, 3]],
    ["FP8e5m2", [1, 5, 2]],
    ["FP8e8m0", [0, 8, 0]],
    ["FP6e2m3", [1, 2, 3]],
    ["FP6e3m2", [1, 3, 2]],
    ["FP4e0m3", [1, 0, 3]],
    ["FP4e2m1", [1, 2, 1]],
])

max_exponent_bit = 12
max_mantissa_bit = 52

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
    #plt.text(max_exponent_bit - 0.5, position, str(e), size = 10, ha="center", va="center")
    #plt.text(max_exponent_bit + 0.5, position, str(m), size = 10, ha="center", va="center")

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

fig.savefig("Number/DataType.png")
plt.close()
