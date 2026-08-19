#!/bin/bash
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

set -xeuo pipefail

rm -rf *.json *.log *.onnx

export MODEL_TRAINED=${TRT_COOKBOOK_PATH}/00-Data/model/model-trained.onnx

# 00-Build the toy transformer block to shard
# Notice:
# + Sharding is pattern driven, and CP and TP look for two different patterns (attention body for
#   CP, SwiGLU MLP for TP). No model in `00-Data/model/` holds either one, see the docstring of
#   build_transformer_block.py.
# + Everything below is a pure ONNX rewrite and runs on a single GPU. What needs several GPUs is
#   *executing* the result, which is case 06.
python3 build_transformer_block.py

export MODEL_BLOCK=./model-transformer-block.onnx

# 01-Generate the sharding hints of both modes
# Notice:
# + `-o` must end in `.json`, otherwise the tool stops with `[!] Output file must be a json`
#   *after* it has already loaded and analysed the model.
# + The hints file is meant to be reviewed and edited by hand before sharding, exactly like the
#   `config.yaml` of `polygraphy plugin match` (see ../Plugin/main.sh).
# + The interesting fields: `attention_layers` (what was detected), `inputs` / `outputs` (which
#   tensors get a collective attached, with `seq_len_idx` = which axis is the sequence length),
#   and `dist_collectives` (`group_size` from `--gpus`, `nb_rank` from `--nb-rank`, `reduce_op`).
polygraphy template shard-hints \
    $MODEL_BLOCK \
    --parallelism CP \
    --gpus 2 \
    -o hints-CP.json \
    > result-01.log 2>&1

cat hints-CP.json >> result-01.log 2>&1
echo >> result-01.log

polygraphy template shard-hints \
    $MODEL_BLOCK \
    --parallelism TP \
    --gpus 2 \
    --nb-rank 2 \
    -o hints-TP.json \
    >> result-01.log 2>&1

cat hints-TP.json >> result-01.log 2>&1
echo >> result-01.log

# 02-Shard for context parallelism (CP)
# CP splits the *sequence* across ranks, so every rank keeps a full copy of the weights and pays
# for it in communication: `reduce_scatter` on the way in, `all_gather` before attention needs the
# whole K/V, `all_gather` on the output. Result: same initializers, +6 `DistCollective` nodes.
polygraphy multi-device shard \
    $MODEL_BLOCK \
    --hint hints-CP.json \
    --output model-CP.onnx \
    > result-02.log 2>&1

polygraphy inspect model model-CP.onnx --show layers attrs >> result-02.log 2>&1

# 03-Shard for tensor parallelism (TP)
# TP splits the *weights*, so it writes one model per rank (`<name>_tp<N>_rank<i>.onnx`) with the
# MLP matrices cut in half -- `w_gate` / `w_up` by column, `w_down` by row -- and needs a single
# `all_reduce` at the end to sum the partial results.
# Notice:
# + The number of ranks comes from `--nb-rank`, NOT from `--gpus`. With the default `--nb-rank 1`
#   the tool cheerfully writes a single `_tp1_rank0` file that is a copy of the input.
# + `--no-save-all-tensors-to-one-file` is rejected for TP, since the weights differ per rank.
polygraphy multi-device shard \
    $MODEL_BLOCK \
    --hint hints-TP.json \
    --output model-TP.onnx \
    > result-03.log 2>&1

ls -l model-TP*.onnx >> result-03.log 2>&1
polygraphy inspect model model-TP_tp2_rank0.onnx --show layers attrs weights >> result-03.log 2>&1

# 04-Skip the hints file entirely
# `--one-shot` runs the same analysis `template shard-hints` does and feeds the result straight to
# the sharder. Use it when the defaults are already right; use the two-step form when the hints
# have to be edited (which is the normal case for a real model).
polygraphy multi-device shard \
    $MODEL_BLOCK \
    --one-shot \
    --parallelism CP \
    --gpus 2 \
    --output model-CP-one-shot.onnx \
    > result-04.log 2>&1

# The two paths agree: same node count, same collectives, only the order in which the three input
# tensors were visited differs (the hints file lists them in an unordered set order).
python3 -c "
import onnx
a = onnx.load('model-CP.onnx')
b = onnx.load('model-CP-one-shot.onnx')
count = lambda m: sum(n.op_type == 'DistCollective' for n in m.graph.node)
print(f'two-step: {len(a.graph.node)} nodes, {count(a)} DistCollective')
print(f'one-shot: {len(b.graph.node)} nodes, {count(b)} DistCollective')
" >> result-04.log 2>&1

# 05-The negative case: a model without either pattern
# Notice:
# + This is the trap of the whole tool. `model-trained.onnx` (MNIST) has no attention and no
#   SwiGLU MLP, and NOTHING says so: the hints file is written successfully with
#   `"attention_layers": []` and empty `inputs` / `outputs`, and sharding with it produces a
#   perfect copy of the model. Both commands exit 0 and print `[I] PASSED`-shaped output.
# + So the thing to check is the hints file, not the exit code: if `attention_layers` is empty,
#   the sharder has nothing to do.
polygraphy template shard-hints \
    $MODEL_TRAINED \
    --parallelism CP \
    --gpus 2 \
    -o hints-no-attention.json \
    > result-05.log 2>&1

cat hints-no-attention.json >> result-05.log 2>&1
echo >> result-05.log

# Sharding with those hints succeeds and changes nothing at all
polygraphy multi-device shard \
    $MODEL_TRAINED \
    --hint hints-no-attention.json \
    --output model-trained-sharded.onnx \
    >> result-05.log 2>&1

python3 -c "
import onnx
a = onnx.load('${MODEL_TRAINED}')
b = onnx.load('model-trained-sharded.onnx')
n_collective = sum(n.op_type == 'DistCollective' for n in b.graph.node)
print(f'input: {len(a.graph.node)} nodes -> output: {len(b.graph.node)} nodes, {n_collective} DistCollective inserted')
" >> result-05.log 2>&1

python3 -c "
import json
hints = json.load(open('hints-no-attention.json'))
print(f\"attention_layers={hints['attention_layers']}, inputs={hints['inputs']}, outputs={hints['outputs']}\")
print('-> nothing to shard, and no warning anywhere says so')
" >> result-05.log 2>&1

# 06-Running the sharded models: NOT covered here, needs more than one GPU
# The sharded graphs contain `DistCollective` nodes, which are NCCL collectives. They are not
# standard ONNX (onnxruntime cannot run them) and one process per rank has to be launched with the
# ranks wired together. This machine has a single GPU, so this example stops at the ONNX rewrite.
# What is still missing, for whoever has the machine:
#   + build each `model-TP_tp2_rank<i>.onnx` into an engine and run rank i on GPU i
#   + compare the gathered output against the single-GPU `model-transformer-block.onnx`, which is
#     an ordinary ONNX file and runs in onnxruntime, so the reference is free
#   + see ../../../05-Plugin/NcclPlugin/ and ../../../08-Advance/MultiDevice/ for the multi-GPU
#     plumbing this needs

echo "Finish"
