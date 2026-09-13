# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
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
"""Outline 4 repeated blocks into ONE shared `FunctionProto` -- `onnx_graphsurgeon` API.

Same job as `outline_with_onnx_api.py`, written twice on purpose so the two APIs can be
compared on real code rather than on documentation. See README.md.
"""

import onnx, onnx_graphsurgeon as gs
import numpy as np, onnxruntime as ort

from prepare_input import SRC  # `../02-PackageSurvey/model-flat.onnx`, rebuilt from code when absent

DOMAIN, FNAME = "cookbook.outlined", "Block"

graph = gs.import_onnx(onnx.load(SRC))
graph.toposort()
instances = [graph.nodes[6 * i:6 * i + 6] for i in range(4)]

def interface(inst):
    """gs Tensors already know their producer/consumers, no maps to build"""
    inside = set(id(n) for n in inst)
    produced = {id(t) for n in inst for t in n.outputs}
    ext_in, seen = [], set()
    for n in inst:
        for t in n.inputs:
            if id(t) not in produced and id(t) not in seen:
                seen.add(id(t))
                ext_in.append(t)
    ext_out = [t for n in inst for t in n.outputs if any(id(c) not in inside for c in t.outputs) or t in graph.outputs]
    return ext_in, ext_out

iface = [interface(inst) for inst in instances]
assert len({(len(a), len(b)) for a, b in iface}) == 1, "interface mismatch"

# ---- function body from instance 0: copy nodes, swap boundary tensors for fresh Variables
in0, out0 = iface[0]
f_in = [gs.Variable(f"f_in_{k}", dtype=t.dtype, shape=t.shape) for k, t in enumerate(in0)]
f_out = [gs.Variable(f"f_out_{k}", dtype=t.dtype, shape=t.shape) for k, t in enumerate(out0)]
remap = {id(t): v for t, v in zip(in0, f_in)} | {id(t): v for t, v in zip(out0, f_out)}
body = []
for n in instances[0]:
    c = n.copy()
    c.name = f"{FNAME}_{n.name}"
    c.inputs = [remap.get(id(t), t) for t in n.inputs]
    c.outputs = [remap.get(id(t), t) for t in n.outputs]
    body.append(c)
func = gs.Function(FNAME, domain=DOMAIN, nodes=body, inputs=f_in, outputs=f_out, opset=graph.opset, import_domains=graph.import_domains)
graph.functions.append(func)

# ---- replace instances with call nodes
for k, (inst, (ext_in, ext_out)) in enumerate(zip(instances, iface)):
    call = gs.Node(op=FNAME, domain=DOMAIN, name=f"{FNAME}_call_{k}", inputs=list(ext_in), outputs=list(ext_out))
    for n in inst:
        n.inputs.clear()
        n.outputs.clear()  # detach; cleanup() will drop them
    graph.nodes.append(call)
graph.cleanup().toposort()

m = gs.export_onnx(graph)
onnx.checker.check_model(m)
onnx.save(m, "out_gs.onnx")
print("main nodes:", len(m.graph.node), "functions:", len(m.functions), "body:", [n.op_type for n in m.functions[0].node])
d = {"x": np.random.rand(2, 8).astype(np.float32) - 0.5}
a = ort.InferenceSession(SRC, providers=["CPUExecutionProvider"]).run(None, d)[0]
b = ort.InferenceSession("out_gs.onnx", providers=["CPUExecutionProvider"]).run(None, d)[0]
print("max diff:", np.max(np.abs(a - b)))
