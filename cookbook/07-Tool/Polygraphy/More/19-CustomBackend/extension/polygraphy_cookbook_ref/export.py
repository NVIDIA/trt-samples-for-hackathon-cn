# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The entry point named in `pyproject.toml`.

It takes no arguments and returns argument-group instances. `polygraphy run`
parses these after all of its own argument groups, in the order given here.
"""

from polygraphy_cookbook_ref.args import CookbookRefRunnerArgs

def export_argument_groups():
    return [CookbookRefRunnerArgs()]
