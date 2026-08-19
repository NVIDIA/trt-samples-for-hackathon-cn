# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The runner itself: a NumPy interpreter for a small set of ONNX ops.

It is what `PluginRefRunner` would be if its op table were not three entries
long, plus one thing that table cannot express -- a choice of working precision,
so that `polygraphy run --trt --cookbook-ref --cookbook-ref-precision float64`
answers "is this difference just float32 rounding?".
"""

import copy
import time
from collections import OrderedDict

from polygraphy import mod, util
from polygraphy.backend.base import BaseRunner
from polygraphy.common import TensorMetadata
from polygraphy.logger import G_LOGGER

# Lazy so that importing this module never drags NumPy in by itself; Polygraphy
# can then install missing dependencies on demand.
np = mod.lazy_import("numpy")
gs = mod.lazy_import("onnx_graphsurgeon")

PRECISIONS = ["float32", "float64"]

def _op_table():
    """Op name -> callable(attrs, *inputs) -> list of outputs."""
    return {
        "Add": lambda attrs, a, b: [a + b],
        "Sub": lambda attrs, a, b: [a - b],
        "Mul": lambda attrs, a, b: [a * b],
        "Div": lambda attrs, a, b: [a / b],
        "Relu": lambda attrs, x: [np.maximum(x, 0)],
        "Sqrt": lambda attrs, x: [np.sqrt(x)],
        "Identity": lambda attrs, x: [x],
        # The cookbook's custom op, so a graph containing it needs no plugin at all.
        "AddScalar": lambda attrs, x: [x + attrs["scalar"]],
    }

@mod.export()
class CookbookRefRunner(BaseRunner):
    """
    Runs inference on the CPU with NumPy, at a configurable working precision.
    """

    def __init__(self, graph, name=None, precision: str = None):
        """
        Args:
            graph (Union[onnx_graphsurgeon.Graph, Callable() -> onnx_graphsurgeon.Graph]):
                    An ONNX-GraphSurgeon graph or a callable that returns one.
            name (str):
                    The human-readable name prefix to use for this runner.
            precision (str):
                    The dtype to compute in. Outputs are cast back to the graph's
                    declared output dtype either way. Defaults to "float32".
        """
        super().__init__(name=name, prefix="cookbook-ref-runner")
        self._graph = graph
        self.precision = util.default(precision, "float32")
        if self.precision not in PRECISIONS:
            G_LOGGER.critical(f"Invalid precision: {self.precision}. Note: valid precisions are: {PRECISIONS}")

    @util.check_called_by("activate")
    def activate_impl(self):
        self.graph, _ = util.invoke_if_callable(self._graph)

    @util.check_called_by("get_input_metadata")
    def get_input_metadata_impl(self):
        meta = TensorMetadata()
        for tensor in self.graph.inputs:
            meta.add(tensor.name, tensor.dtype, tensor.shape)
        return meta

    @util.check_called_by("infer")
    def infer_impl(self, feed_dict):
        start = time.time()
        table = _op_table()
        work_dtype = np.dtype(self.precision)

        values = {name: np.asarray(value).astype(work_dtype) for name, value in copy.copy(feed_dict).items()}

        for node in self.graph.nodes:
            if node.op not in table:
                G_LOGGER.critical(f"Op: {node.op} is not supported by CookbookRefRunner.\nNote: Supported ops are: {sorted(table)}")

            inputs = []
            for tensor in node.inputs:
                if isinstance(tensor, gs.Constant):
                    inputs.append(tensor.values.astype(work_dtype))
                else:
                    inputs.append(values[tensor.name])

            for tensor, value in zip(node.outputs, table[node.op](node.attrs, *inputs)):
                values[tensor.name] = value

        outputs = OrderedDict()
        for tensor in self.graph.outputs:
            # Cast back so the comparison against another runner is apples to
            # apples; only the intermediate arithmetic used the wider type.
            outputs[tensor.name] = values[tensor.name].astype(tensor.dtype)

        self.inference_time = time.time() - start
        return outputs

    @util.check_called_by("deactivate")
    def deactivate_impl(self):
        del self.graph
