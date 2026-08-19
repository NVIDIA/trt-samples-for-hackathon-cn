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
"""Outline 4 repeated blocks into ONE shared `FunctionProto` -- raw `onnx` API.

Same job as `outline_with_graphsurgeon.py`, written twice on purpose so the two APIs can be
compared on real code rather than on documentation. See README.md.
"""

import onnx, onnx.helper as oh
import numpy as np, onnxruntime as ort

from prepare_input import SRC  # `../02-PackageSurvey/model-flat.onnx`, rebuilt from code when absent

DOMAIN, FNAME = "cookbook.outlined", "Block"

m = onnx.load(SRC)
g = m.graph
init_names = {t.name for t in g.initializer}
producer = {o: i for i, n in enumerate(g.node) for o in n.output}
# instances: block i uses nodes 6i..6i+5
instances = [list(range(6 * i, 6 * i + 6)) for i in range(4)]

def interface(inst):
    """external inputs (in first-use order) and external outputs of one instance"""
    inside = set(inst)
    produced = {o for i in inst for o in g.node[i].output}
    ext_in, seen = [], set()
    for i in inst:
        for name in g.node[i].input:
            if name not in produced and name not in seen:
                seen.add(name)
                ext_in.append(name)
    consumers_outside = set()
    for j, n in enumerate(g.node):
        if j in inside: continue
        for name in n.input:
            if name in produced: consumers_outside.add(name)
    for out in g.output:
        if out.name in produced: consumers_outside.add(out.name)
    ext_out = [o for i in inst for o in g.node[i].output if o in consumers_outside]
    return ext_in, ext_out

iface = [interface(inst) for inst in instances]
assert len({(len(a), len(b)) for a, b in iface}) == 1, "interface mismatch"

# ---- build the function body from instance 0
inst0, (in0, out0) = instances[0], iface[0]
rename = {name: f"f_in_{k}" for k, name in enumerate(in0)}
for k, name in enumerate(out0):
    rename[name] = f"f_out_{k}"
counter = 0
for i in inst0:
    for o in g.node[i].output:
        if o not in rename:
            rename[o] = f"f_t_{counter}"
            counter += 1
body = []
for i in inst0:
    n = onnx.NodeProto()
    n.CopyFrom(g.node[i])
    n.name = f"{FNAME}_{n.name}"
    for k, name in enumerate(n.input):
        n.input[k] = rename.get(name, name)
    for k, name in enumerate(n.output):
        n.output[k] = rename.get(name, name)
    body.append(n)
used_domains = sorted({n.domain for n in body})
func = oh.make_function(
    DOMAIN,
    FNAME,
    [rename[x] for x in in0],
    [rename[x] for x in out0],
    body,
    [oh.make_opsetid(d, next(o.version for o in m.opset_import if o.domain == d)) for d in used_domains],
)

# ---- replace every instance with one call node
keep = [n for j, n in enumerate(g.node) if j not in {j for inst in instances for j in inst}]
call = [oh.make_node(FNAME, ext_in, ext_out, f"{FNAME}_call_{k}", domain=DOMAIN) for k, (ext_in, ext_out) in enumerate(iface)]
new_nodes = call + keep
del g.node[:]
g.node.extend(new_nodes)
m.functions.extend([func])
m.opset_import.extend([oh.make_opsetid(DOMAIN, 1)])
# topological sort
order, emitted, pending = [], set(t.name for t in g.initializer) | {i.name for i in g.input}, list(g.node)
while pending:
    progressed = False
    for n in list(pending):
        if all(x in emitted or x == "" for x in n.input):
            order.append(n)
            emitted.update(n.output)
            pending.remove(n)
            progressed = True
    assert progressed, "cycle"
del g.node[:]
g.node.extend(order)

onnx.checker.check_model(m)
onnx.save(m, "out_raw.onnx")
print("main nodes:", len(m.graph.node), "functions:", len(m.functions), "body:", [n.op_type for n in m.functions[0].node])
d = {"x": np.random.rand(2, 8).astype(np.float32) - 0.5}
a = ort.InferenceSession(SRC, providers=["CPUExecutionProvider"]).run(None, d)[0]
b = ort.InferenceSession("out_raw.onnx", providers=["CPUExecutionProvider"]).run(None, d)[0]
print("max diff:", np.max(np.abs(a - b)))
