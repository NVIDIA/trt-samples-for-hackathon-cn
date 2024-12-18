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

from polygraphy import mod

gs = mod.lazy_import("onnx_graphsurgeon>=0.5.0")
from typing import Dict, List

def get_plugin_pattern():
    """
    Toy plugin pattern:
        A     B
        \   /
          C, attrs['x'] < 2.0
        /   \
        D     E
    """
    pattern = gs.GraphPattern()
    in_0 = pattern.variable()
    in_1 = pattern.variable()
    a_out = pattern.add("Anode", "A", inputs=[in_0])
    b_out = pattern.add("Bnode", "B", inputs=[in_1])
    check_function = lambda node: node.attrs["x"] < 2.0
    c_out = pattern.add("Cnode", "C", inputs=[a_out, b_out], check_func=check_function)
    d_out = pattern.add("Dnode", "D", inputs=[c_out])
    e_out = pattern.add("Enode", "E", inputs=[c_out])
    pattern.set_output_tensors([d_out, e_out])

    return pattern

def get_matching_subgraphs(graph) -> List[Dict[str, str]]:
    gp = get_plugin_pattern()
    matches = gp.match_all(graph)
    ans = []
    for m in matches:
        # save the input and output tensor names of the matching subgraph(s)
        input_tensors = list(set([ip_tensor.name for ip_tensor in m.inputs]))
        output_tensors = list(set([op_tensor.name for op_tensor in m.outputs]))

        attrs = {"ToyX": int(m.get("Cnode").attrs["x"]) * 2}
        ioa = {'inputs': input_tensors, 'outputs': output_tensors, 'attributes': attrs}
        ans.append(ioa)
    return ans

def get_plugin_metadata() -> Dict[str, str]:
    return {
        'name': 'toyPlugin',
        'op': 'CustomToyPlugin',
    }
