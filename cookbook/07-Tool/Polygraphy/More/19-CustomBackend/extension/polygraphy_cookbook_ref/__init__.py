# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A NumPy reference backend for `polygraphy run`.

Importing this package registers `AddScalar` with Polygraphy's own
`PluginRefRunner` op registry as a side effect, which is the only way to add an
op to `--pluginref` -- there is no command-line option for it.
"""

__version__ = "0.1.0"

from polygraphy.backend.pluginref.references import register

@register("AddScalar")
def run_add_scalar(attrs, x):
    """The cookbook's `AddScalar` plugin, in NumPy.

    See `05-Plugin/ONNXParserWithPlugin/AddScalarPlugin.cu` for the CUDA version.
    """
    return [x + attrs["scalar"]]
