#!/bin/bash
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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

set -xeuo pipefail

rm -rf *.log *.onnx *.so config.yaml

# 00-Build the toy model to work on
# Notice:
# + `polygraphy plugin` works on the *ONNX* level: it matches a graph pattern described by a
#   `pattern.py` shipped next to the plugin, and rewrites the matched subgraph into a single node.
#   It never loads a `.so`, so pointing `--plugin-dir` at a directory of shared libraries (which an
#   earlier version of this example did) finds nothing and prints an empty `{}`.
# + The plugin directory layout is `<plugin-dir>/<plugin name>/pattern.py`, see
#   ./match_and_replace_plug/plugins/toyPlugin/pattern.py for the pattern used here and
#   ./match_and_replace_plug/README.md for the workflow it comes from.
python3 build_toy_subgraph.py

export MODEL_TOY=./toy_subgraph.onnx
export PLUGIN_DIR=./match_and_replace_plug/plugins

# 01-List the potential substitutions, without writing anything (dry run)
# The log shows why each candidate was rejected: wrong op type, or `check_func` returning false
# for the `C` node whose attribute `x` is 9 (the pattern requires `x < 2.0`).
polygraphy plugin list $MODEL_TOY \
    --plugin-dir $PLUGIN_DIR \
    > result-01.log 2>&1

# 02-Save the potential substitutions into an intermediate file
# Notice:
# + Without `-o`, `config.yaml` is written **into the directory of the model**, which silently
#   pollutes a shared model directory. Always pass `-o`.
# + This file is meant to be reviewed and edited by hand before the next step: deleting an entry
#   from it is how a particular match is kept out of the replacement.
polygraphy plugin match $MODEL_TOY \
    --plugin-dir $PLUGIN_DIR \
    -o config.yaml \
    > result-02.log 2>&1

cat config.yaml >> result-02.log 2>&1

# 03-Replace the matched subgraph with the plugin node
# The 5 nodes `A,B -> C -> D,E` become a single `CustomToyPlugin` node, and the attribute of the
# new node comes from `get_matching_subgraphs` in pattern.py (`ToyX = 2 * x`), not from the model.
polygraphy plugin replace $MODEL_TOY \
    --plugin-dir $PLUGIN_DIR \
    --config config.yaml \
    -o toy_subgraph-replaced.onnx \
    > result-03.log 2>&1

polygraphy inspect model toy_subgraph-replaced.onnx --show layers attrs >> result-03.log 2>&1

# 04-`plugin autotune` is NOT covered here
# It builds engines to time each candidate substitution, so it needs a plugin that TensorRT can
# actually load and run, which `toyPlugin` (a pattern only, no implementation) is not. Loading a
# real plugin library is its own subject, see ../More/16-PluginReference/ for why `--plugins` /
# `LoadPlugins` cannot register an `IPluginCreatorV3One` at all.

echo "Finish"
