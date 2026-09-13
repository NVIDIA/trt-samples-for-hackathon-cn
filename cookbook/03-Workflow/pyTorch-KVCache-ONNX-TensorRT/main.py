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
"""An autoregressive model end to end: PyTorch -> ONNX -> TensorRT, with a KV cache.

Every other `03-Workflow` example converts a model that is called **once**. A decoder-only language
model is called once per generated token and drags a KV cache along, which changes three things:

+ the cache turns into a wall of graph I/O -- gpt2-medium's decode graph has **99 I/O tensors**;
+ the cache is read and written every step, so where it lives matters more than how fast the
  kernels are;
+ the host does the same thing to the output every step, which is work the graph could do instead.

The techniques here come from a hand-written ChatGLM-6B pipeline (`99-Todo/chatglm-6b.md`), reduced
to `gpt2` (124M) so the whole thing builds in seconds.

+ Steps to run.

```bash
python3 main.py
```
"""

import sys
import time
from pathlib import Path

import onnx
import tensorrt as trt
import torch
from tensorrt_cookbook import case_mark, cookbook_path

# The checkpoint is read from `00-Data/model/gpt2/`, not fetched from the network at run time. See
# `00-Data/README.md` for the one command that populates it; `require_local_model()` below prints
# that command if it is missing. Keeping the download out of the example run is deliberate: an
# example that silently pulls 528 MB the first time it is invoked cannot be run offline, cannot be
# run reproducibly, and turns a network outage into a confusing failure in the middle of a case.
# `cookbook_path` raises when the target is absent, which is the right behaviour for the checked-in
# assets but not for one that may legitimately not be downloaded yet - so resolve the parent, which
# always exists, and join the last component ourselves.
MODEL_PATH = cookbook_path("00-Data", "model") / "gpt2"
MAX_LENGTH = 256  # Largest total sequence length the engine will accept
N_NEW_TOKEN = 12
PROMPT = "TensorRT is a"

current_path = Path(__file__).parent
step_onnx_file = current_path / "model-gpt2-step.onnx"
logit_onnx_file = current_path / "model-gpt2-step-logit.onnx"
step_trt_file = current_path / "model-gpt2-step.trt"
logit_trt_file = current_path / "model-gpt2-step-logit.trt"

def require_local_model() -> bool:
    """Is the checkpoint present locally? If not, say exactly how to get it."""
    if (MODEL_PATH / "config.json").exists():
        return True
    print(f"    `gpt2` is not in {MODEL_PATH}, so this example cannot run.")
    print("    It is a one-off download, deliberately kept out of the example run:")
    print("")
    print("        cd <PathToCookbook>/00-Data")
    print("        python3 get-model-gpt2.py")
    print("")
    return False

def load_torch_model():
    """`gpt2` on GPU in eval mode, read from the local `00-Data` copy.

    `from_pretrained` takes a path just as happily as a hub id, and given a path it never touches
    the network - no implicit `~/.cache/huggingface`, no `local_files_only` flag needed.
    """
    from transformers import GPT2LMHeadModel
    model = GPT2LMHeadModel.from_pretrained(str(MODEL_PATH)).eval().cuda()
    config = model.config
    return model, config.n_layer, config.n_head, config.n_embd // config.n_head, config.vocab_size

class Gpt2Step(torch.nn.Module):
    """One decode step, with the whole KV cache packed into a single tensor.

    This is the piece worth copying. `transformers` hands the cache around as a `Cache` object, one
    `[B, H, L, D]` tensor per layer per key/value, which `torch.onnx.export` turns into 2*n_layer
    graph inputs and as many outputs. Wrapping the model in a `Module` whose signature is
    *flat tensors* moves the packing into PyTorch, where it is six lines, instead of into
    ONNX-GraphSurgeon, where it is a `Split`/`Concat` rewrite over ~100 tensors.

    The cache is packed **sequence-first**, `[L, 2 * n_layer, B, H, D]`, not in the natural
    `[2 * n_layer, B, H, L, D]` order. That costs a `permute` at each end and buys the whole of
    `case_the_layout_decides_whether_the_cache_can_alias`.
    """

    def __init__(self, model, n_layer: int, b_return_logit: bool = False):
        super().__init__()
        self.model = model
        self.n_layer = n_layer
        self.b_return_logit = b_return_logit

    def forward(self, input_ids, attention_mask, past_kv):
        from transformers.cache_utils import DynamicCache

        past = past_kv.permute(1, 2, 3, 0, 4)  # [L_past, 2N, B, H, D] -> [2N, B, H, L_past, D]
        cache = DynamicCache()
        for i in range(self.n_layer):
            cache.update(past[2 * i], past[2 * i + 1], i)

        output = self.model(input_ids=input_ids, attention_mask=attention_mask, past_key_values=cache, use_cache=True)

        updated_cache = output.past_key_values
        present = torch.stack([t for i in range(self.n_layer) for t in (updated_cache.layers[i].keys, updated_cache.layers[i].values)], dim=0)
        present = present.permute(3, 0, 1, 2, 4).contiguous()  # back to sequence-first

        if self.b_return_logit:  # The comparison case: hand the whole vocabulary back to the host
            return output.logits[:, -1, :], present
        # Greedy sampling inside the graph: the engine returns a token id, not 50257 floats
        next_token = torch.argmax(output.logits[:, -1, :], dim=-1, keepdim=True).to(torch.int32)
        return next_token, present

def export_step_onnx(onnx_file: Path, b_return_logit: bool):
    """Trace one decode step into ONNX. Traced at `L_past = 3`; the result is not specialized to it."""
    if onnx_file.exists():
        return
    model, n_layer, n_head, head_dimension, _ = load_torch_model()
    wrapper = Gpt2Step(model, n_layer, b_return_logit).eval()

    n_past = 3
    input_ids = torch.tensor([[100]], dtype=torch.int64, device="cuda")
    attention_mask = torch.ones(1, n_past + 1, dtype=torch.int64, device="cuda")
    past_kv = torch.zeros(n_past, 2 * n_layer, 1, n_head, head_dimension, dtype=torch.float32, device="cuda")

    output_name = "logit" if b_return_logit else "next_token"
    torch.onnx.export(
        wrapper,
        (input_ids, attention_mask, past_kv),
        str(onnx_file),
        input_names=["input_ids", "attention_mask", "past_kv"],
        output_names=[output_name, "present_kv"],
        dynamic_axes={
            "input_ids": {
                0: "B",
                1: "L"
            },
            "attention_mask": {
                0: "B",
                1: "LTotal"
            },
            "past_kv": {
                0: "LPast",
                2: "B"
            },
            output_name: {
                0: "B"
            },
            "present_kv": {
                0: "LTotal",
                2: "B"
            },
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    return

def build_engine(onnx_file: Path, trt_file: Path, n_layer, n_head, head_dimension):
    """Parse and build. One optimization profile covering `L_past` from 0 to `MAX_LENGTH - 1`.

    The engine is rebuilt on every run, deliberately. A serialized plan is tied to the compute
    capability it was built on, so a stale one left over from another machine fails at load with
    `Error Code 6: The engine plan file is generated on an incompatible device`. Caching it would
    save ~12 s and cost a confusing failure the first time this directory is reused elsewhere. The
    exported ONNX above is still cached: it is device independent, and re-exporting it is the slow
    part.
    """
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network()
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_file.read_bytes(), str(onnx_file)):
        raise RuntimeError(f"Failed parsing {onnx_file}: {parser.get_error(0)}")

    builder_config = builder.create_builder_config()
    profile = builder.create_optimization_profile()
    profile.set_shape("input_ids", [1, 1], [1, 1], [1, 1])
    profile.set_shape("attention_mask", [1, 1], [1, 64], [1, MAX_LENGTH])
    cache_shape = lambda n: [n, 2 * n_layer, 1, n_head, head_dimension]  # noqa: E731
    profile.set_shape("past_kv", cache_shape(0), cache_shape(63), cache_shape(MAX_LENGTH - 1))
    builder_config.add_optimization_profile(profile)

    start_time = time.time()
    engine_bytes = builder.build_serialized_network(network, builder_config)
    build_time = time.time() - start_time
    if engine_bytes is None:
        raise RuntimeError(f"Failed building {trt_file}")
    trt_file.write_bytes(memoryview(engine_bytes))
    return trt_file.stat().st_size, build_time

def load_engine(trt_file: Path):
    logger = trt.Logger(trt.Logger.ERROR)
    engine = trt.Runtime(logger).deserialize_cuda_engine(trt_file.read_bytes())
    return engine, engine.create_execution_context()

@case_mark
def case_the_io_explosion():
    """Why an autoregressive model needs its own workflow example: count the graph I/O.

    Uses `00-Data/model/model-large.onnx`, which is already in the tree -- it is gpt2-**medium**'s
    `decoder_model.onnx`, so this costs no download.
    """
    onnx_file = cookbook_path("00-Data", "model") / "model-large.onnx"
    model = onnx.load(onnx_file, load_external_data=False)  # Metadata only, the file is 1.6 GB
    input_name = [t.name for t in model.graph.input]
    output_name = [t.name for t in model.graph.output]
    n_layer_medium = (len(output_name) - 1) // 2

    print(f"    {onnx_file.name} (gpt2-medium, prefill only)")
    print(f"        inputs  {len(input_name):3d}: {input_name}")
    print(f"        outputs {len(output_name):3d}: {output_name[:3]} ... ({n_layer_medium} layers x key/value)")
    print(f"    the matching decode graph adds {2 * n_layer_medium} past inputs -> "
          f"{2 + 2 * n_layer_medium} in + {1 + 2 * n_layer_medium} out = {3 + 4 * n_layer_medium} I/O tensors")
    print(f"    -> every one of them needs a set_input_shape + set_tensor_address, every token")
    return

@case_mark
def case_export_the_packed_step_graph():
    """Export one decode step with the cache packed into a single tensor, and check the numbers."""
    export_step_onnx(step_onnx_file, b_return_logit=False)
    model = onnx.load(step_onnx_file, load_external_data=False)
    describe = lambda t: (t.name, [d.dim_param or d.dim_value for d in t.type.tensor_type.shape.dim])  # noqa: E731
    print(f"    {step_onnx_file.name}: {len(model.graph.node)} nodes")
    for tensor in list(model.graph.input) + list(model.graph.output):
        print(f"        {'in ' if tensor in model.graph.input else 'out'} {describe(tensor)}")
    print(f"    -> 5 I/O tensors instead of 51 (gpt2) or 99 (gpt2-medium)")

    # `transformers` emits a pile of TracerWarnings that look like the graph is being frozen to the
    # traced L_past. It is not -- run it at other lengths and compare against eager PyTorch.
    torch_model, n_layer, n_head, head_dimension, _ = load_torch_model()
    engine_size, build_time = build_engine(step_onnx_file, step_trt_file, n_layer, n_head, head_dimension)
    print(f"    engine {engine_size / 2**20:.0f} MiB, built in {build_time:.1f} s")
    _, context = load_engine(step_trt_file)

    from transformers.cache_utils import DynamicCache
    generator = torch.Generator(device="cuda").manual_seed(31193)
    for n_past in [1, 7, 16]:  # Traced at 3
        past_kv = torch.randn(n_past, 2 * n_layer, 1, n_head, head_dimension, generator=generator, device="cuda") * 0.1
        input_ids = torch.tensor([[1000]], dtype=torch.int64, device="cuda")
        attention_mask = torch.ones(1, n_past + 1, dtype=torch.int64, device="cuda")

        next_token = torch.zeros(1, 1, dtype=torch.int32, device="cuda")
        present_kv = torch.zeros(n_past + 1, 2 * n_layer, 1, n_head, head_dimension, dtype=torch.float32, device="cuda")
        run_step(context, input_ids, attention_mask, past_kv, next_token, present_kv)

        cache = DynamicCache()
        permuted = past_kv.permute(1, 2, 3, 0, 4)
        for i in range(n_layer):
            cache.update(permuted[2 * i], permuted[2 * i + 1], i)
        with torch.no_grad():
            output = torch_model(input_ids=input_ids, attention_mask=attention_mask, past_key_values=cache, use_cache=True)
        reference_token = int(torch.argmax(output.logits[:, -1, :], dim=-1))
        print(f"        L_past={n_past:2d}: TensorRT token {int(next_token[0, 0]):5d}, PyTorch token {reference_token:5d}, "
              f"match={int(next_token[0, 0]) == reference_token}")
        assert int(next_token[0, 0]) == reference_token, "the engine disagrees with PyTorch"
    return

def run_step(context, input_ids, attention_mask, past_kv, output_tensor, present_kv):
    """One `execute_async_v3`, with the caller deciding where every tensor lives."""
    context.set_input_shape("input_ids", list(input_ids.shape))
    context.set_input_shape("attention_mask", list(attention_mask.shape))
    context.set_input_shape("past_kv", list(past_kv.shape))
    context.set_tensor_address("input_ids", input_ids.data_ptr())
    context.set_tensor_address("attention_mask", attention_mask.data_ptr())
    context.set_tensor_address("past_kv", past_kv.data_ptr())
    context.set_tensor_address(context.engine.get_tensor_name(3), output_tensor.data_ptr())
    context.set_tensor_address("present_kv", present_kv.data_ptr())
    assert context.execute_async_v3(0), "execute_async_v3 failed"
    torch.cuda.synchronize()
    return

@case_mark
def case_generate_with_an_in_place_cache():
    """The headline: `past_kv` and `present_kv` bound to **one** allocation, and text comes out.

    The engine writes `L_past + 1` entries where it read `L_past`, appending at the end of the
    allocation, so the same buffer is both the input and the output. No copy, no ping-pong, and the
    cache is allocated once at `MAX_LENGTH` instead of growing.
    """
    from transformers import GPT2Tokenizer
    _, n_layer, n_head, head_dimension, _ = load_torch_model()
    engine, context = load_engine(step_trt_file)
    print(f"    engine I/O: {[engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]}")

    tokenizer = GPT2Tokenizer.from_pretrained(str(MODEL_PATH))
    prompt_ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"].cuda()

    cache = torch.zeros(MAX_LENGTH, 2 * n_layer, 1, n_head, head_dimension, dtype=torch.float32, device="cuda")
    next_token = torch.zeros(1, 1, dtype=torch.int32, device="cuda")
    attention_mask = torch.ones(1, MAX_LENGTH, dtype=torch.int64, device="cuda")

    generated_token_list = []
    n_past = 0
    n_step = prompt_ids.shape[1] + N_NEW_TOKEN
    for step in range(n_step):
        step_ids = prompt_ids[:, step:step + 1] if step < prompt_ids.shape[1] else next_token.to(torch.int64)
        context.set_input_shape("input_ids", [1, 1])
        context.set_input_shape("attention_mask", [1, n_past + 1])
        context.set_input_shape("past_kv", [n_past, 2 * n_layer, 1, n_head, head_dimension])
        context.set_tensor_address("input_ids", step_ids.contiguous().data_ptr())
        context.set_tensor_address("attention_mask", attention_mask.data_ptr())
        context.set_tensor_address("past_kv", cache.data_ptr())  # The same
        context.set_tensor_address("present_kv", cache.data_ptr())  # allocation
        context.set_tensor_address("next_token", next_token.data_ptr())
        assert context.execute_async_v3(0), "execute_async_v3 failed"
        torch.cuda.synchronize()
        n_past += 1
        if step >= prompt_ids.shape[1] - 1:
            generated_token_list.append(int(next_token[0, 0]))

    cache_byte = MAX_LENGTH * 2 * n_layer * n_head * head_dimension * 4
    print(f"    one cache allocation of {cache_byte / 2**20:.0f} MiB serves both bindings for all {n_step} steps")
    print(f"    prompt    : {PROMPT}")
    print(f"    continued : {tokenizer.decode(generated_token_list)}")
    assert len(generated_token_list) == N_NEW_TOKEN + 1
    return

@case_mark
def case_the_layout_decides_whether_the_cache_can_alias():
    """Why the wrapper repacks the cache sequence-first, instead of leaving it as it comes.

    Aliasing works only if the engine *appends*: the first `L_past` elements of the output must be
    the input, byte for byte. That is a property of the memory layout, not a TensorRT feature.
    """
    torch_model, n_layer, n_head, head_dimension, _ = load_torch_model()
    from transformers.cache_utils import DynamicCache

    n_past = 5
    generator = torch.Generator(device="cuda").manual_seed(97)
    past = torch.randn(2 * n_layer, 1, n_head, n_past, head_dimension, generator=generator, device="cuda") * 0.1
    cache = DynamicCache()
    for i in range(n_layer):
        cache.update(past[2 * i], past[2 * i + 1], i)
    with torch.no_grad():
        output = torch_model(input_ids=torch.tensor([[1000]], device="cuda"), attention_mask=torch.ones(1, n_past + 1, dtype=torch.int64, device="cuda"), past_key_values=cache, use_cache=True)
    updated = output.past_key_values
    present = torch.stack([t for i in range(n_layer) for t in (updated.layers[i].keys, updated.layers[i].values)], dim=0)

    # (a) the layout HuggingFace hands out: [2N, B, H, L, D], sequence on axis 3
    natural = present.contiguous()
    prefix_preserved_natural = torch.equal(natural.reshape(-1)[:past.numel()], past.reshape(-1))
    # (b) what the wrapper exports: [L, 2N, B, H, D], sequence outermost
    sequence_first = present.permute(3, 0, 1, 2, 4).contiguous()
    prefix_preserved_sequence_first = torch.equal(sequence_first.reshape(-1)[:past.numel()], past.permute(3, 0, 1, 2, 4).reshape(-1))

    print(f"    [2N, B, H, L, D] (as transformers stores it): first {past.numel()} values equal the input? "
          f"{prefix_preserved_natural}")
    print(f"    [L, 2N, B, H, D] (what this example exports): first {past.numel()} values equal the input? "
          f"{prefix_preserved_sequence_first}")
    assert not prefix_preserved_natural and prefix_preserved_sequence_first
    print("    -> only the sequence-first layout appends; with the natural one a shared buffer is silently wrong")
    return

@case_mark
def case_sampling_inside_the_engine():
    """`argmax` in the graph vs handing 50257 floats back to the host, every token."""
    _, n_layer, n_head, head_dimension, vocabulary_size = load_torch_model()
    export_step_onnx(logit_onnx_file, b_return_logit=True)
    build_engine(logit_onnx_file, logit_trt_file, n_layer, n_head, head_dimension)

    time_dict = {}
    for tag, trt_file, output_shape, dtype in [
        ("argmax in the engine", step_trt_file, (1, 1), torch.int32),
        ("logit to the host   ", logit_trt_file, (1, vocabulary_size), torch.float32),
    ]:
        _, context = load_engine(trt_file)
        n_past = 64
        cache = torch.zeros(MAX_LENGTH, 2 * n_layer, 1, n_head, head_dimension, dtype=torch.float32, device="cuda")
        output_tensor = torch.zeros(*output_shape, dtype=dtype, device="cuda")
        input_ids = torch.tensor([[1000]], dtype=torch.int64, device="cuda")
        attention_mask = torch.ones(1, n_past + 1, dtype=torch.int64, device="cuda")

        context.set_input_shape("input_ids", [1, 1])
        context.set_input_shape("attention_mask", [1, n_past + 1])
        context.set_input_shape("past_kv", [n_past, 2 * n_layer, 1, n_head, head_dimension])
        context.set_tensor_address("input_ids", input_ids.data_ptr())
        context.set_tensor_address("attention_mask", attention_mask.data_ptr())
        context.set_tensor_address("past_kv", cache.data_ptr())
        context.set_tensor_address("present_kv", cache.data_ptr())
        context.set_tensor_address(context.engine.get_tensor_name(3), output_tensor.data_ptr())

        for _ in range(10):  # The host-side step: bring the result back and pick a token
            context.execute_async_v3(0)
            _ = output_tensor.cpu() if dtype == torch.int32 else int(torch.argmax(output_tensor.cpu()))
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(100):
            context.execute_async_v3(0)
            _ = output_tensor.cpu() if dtype == torch.int32 else int(torch.argmax(output_tensor.cpu()))
        torch.cuda.synchronize()
        time_dict[tag] = (time.time() - start_time) * 1000 / 100
        print(f"    {tag}: {time_dict[tag]:6.3f} ms/token, output {tuple(output_shape)} {str(dtype).split('.')[-1]}")

    saved_byte = vocabulary_size * 4 - 4
    print(f"    -> {saved_byte / 2**10:.0f} KiB less to move per token, "
          f"{time_dict['logit to the host   '] / time_dict['argmax in the engine']:.2f}x on the whole step")
    return

if __name__ == "__main__":
    if not require_local_model():
        print("Skipped")
        sys.exit(0)

    case_the_io_explosion()
    case_export_the_packed_step_graph()
    case_generate_with_an_in_place_cache()
    case_the_layout_decides_whether_the_cache_can_alias()
    case_sampling_inside_the_engine()

    print("\nFinish")
