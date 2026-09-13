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
"""Keeping tensors on the GPU across a `TrtRunner` call.

`polygraphy.cuda` is three small classes: `DeviceArray` owns a device
allocation, `DeviceView` borrows one, and `Stream` wraps a CUDA stream. Feed a
`DeviceView` to `TrtRunner.infer` and TensorRT reads your pointer directly; ask
for `copy_outputs_to_host=False` and you get a `DeviceView` back instead of a
host array.

Whether that is worth doing is a question about tensor size, so
`case_what_the_round_trip_actually_costs` measures both ends of the range: on
MNIST it saves 0.19 ms (1.46x), on a 25 MB tensor it saves 9.0 ms (32x), which is
97% of the wall clock.

The two ways to get hurt are both quiet. A `DeviceView` does not keep its memory
alive (`case_a_view_does_not_keep_the_memory_alive`), and the `stream=` argument
does not actually make copies asynchronous unless the host buffer is pinned --
which means the missing `synchronize()` in your code is invisible until the day
someone speeds it up (`case_the_stream_that_is_not_asynchronous`).

This also closes the loose end from `../10-PyTorchTensors/`: the object that had
no `.device` and no `.cpu()` is a `DeviceView`, and its whole interface is
`copy_to` / `numpy` / `ptr` / `shape` / `dtype` / `nbytes`.
"""

import subprocess
import sys
import textwrap
import time
import warnings

import numpy as np
import tensorrt as trt
import torch
from polygraphy import func
from polygraphy.backend.trt import CreateConfig, CreateNetwork, EngineFromNetwork, NetworkFromOnnxPath, TrtRunner
from polygraphy.cuda import DeviceArray, DeviceView, Stream
from polygraphy.logger import G_LOGGER

from tensorrt_cookbook import case_mark, cookbook_path

G_LOGGER.module_severity = G_LOGGER.ERROR

onnx_file = str(cookbook_path("00-Data", "model", "model-trained.onnx"))
mnist_input = np.load(cookbook_path("00-Data", "data", "InferenceData.npy"))

BIG_SHAPE = (8, 3, 512, 512)  # 25.2 MB of float32, enough for copies to matter
RACE_NBYTES = 256 * 1024 * 1024  # wide enough that a half-finished copy is visible
N_WARMUP, N_TEST = 10, 50

def build_mnist_engine() -> trt.ICudaEngine:
    return EngineFromNetwork(NetworkFromOnnxPath(onnx_file), config=CreateConfig())()

@func.extend(CreateNetwork())
def big_network(builder, network) -> None:
    """`y = x + x` on a large tensor: real bytes to move, almost nothing to compute."""
    tensor = network.add_input("x", trt.float32, BIG_SHAPE)
    layer = network.add_elementwise(tensor, tensor, trt.ElementWiseOperation.SUM)
    layer.get_output(0).name = "y"
    network.mark_output(layer.get_output(0))

def benchmark(work) -> float:
    """Mean wall-clock milliseconds per call."""
    for _ in range(N_WARMUP):
        work()
    start = time.perf_counter()
    for _ in range(N_TEST):
        work()
    return (time.perf_counter() - start) * 1000 / N_TEST

@case_mark
def case_owning_versus_borrowing() -> None:
    """`DeviceArray` owns the allocation; `DeviceView` only remembers an address.

    `array.view()` hands back the *same* pointer -- it allocates nothing and
    copies nothing, which is the entire point. The difference is what each type
    lets you do: only the owner has `copy_from`, `resize` and `free`.

    Note that reading `.dtype` is itself deprecated (removal in Polygraphy
    0.55.0). Today it returns a NumPy dtype; the replacement returns a
    Polygraphy `DataType`, so `np.empty(..., dtype=view.dtype)` is code that
    will break silently later.
    """
    host = np.arange(6, dtype=np.float32).reshape(2, 3)

    with DeviceArray(shape=host.shape, dtype=host.dtype) as array:
        array.copy_from(host)
        view = array.view()

        print(f"    array : {array}")
        print(f"    view  : {view}")
        print(f"    view.ptr == array.ptr : {view.ptr == array.ptr}  (view() is not a copy)")

        public = lambda obj: {name for name in dir(obj) if not name.startswith("_")}
        print(f"    DeviceView  can do : {sorted(public(view))}")
        print(f"    DeviceArray adds   : {sorted(public(array) - public(view))}")
        print(f"    round trip through the device: {view.numpy().tolist()}")

        # `resize` never shrinks the allocation, so `nbytes` and the amount of
        # memory actually held apart from each other once you resize down.
        array.resize((1, 3))
        print(f"    after resize((1, 3)): nbytes {array.nbytes}, still holding {array.allocated_nbytes} B")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            dtype = view.dtype
        print(f"    view.dtype returns {dtype!r} and warns: {caught[0].message if caught else 'nothing'}")
    return

@case_mark
def case_feeding_the_runner_without_a_copy() -> None:
    """A `DeviceView` input is used in place -- the proof is the tensor address.

    With a NumPy feed, `TrtRunner` allocates its own device buffer, copies into
    it, and binds *that*. With a `DeviceView` feed it binds your pointer. Reading
    `context.get_tensor_address` after each call shows which happened, and the
    two produce identical results.
    """
    engine = build_mnist_engine()
    with DeviceArray(shape=mnist_input.shape, dtype=mnist_input.dtype) as array:
        array.copy_from(mnist_input)

        with TrtRunner(engine) as runner:
            from_host = runner.infer({"x": mnist_input})
            host_addr = runner.context.get_tensor_address("x")

            from_device = runner.infer({"x": array.view()})
            device_addr = runner.context.get_tensor_address("x")

        print(f"    our DeviceArray lives at        : {hex(array.ptr)}")
        print(f"    bound address after NumPy feed  : {hex(host_addr)}  -> runner's own buffer")
        print(f"    bound address after view feed   : {hex(device_addr)}  -> ours ({device_addr == array.ptr})")
        print(f"    outputs identical               : {np.array_equal(from_host['y'], from_device['y'])}")
        print("    a DeviceArray works as a feed too -- it is a DeviceView subclass")
    return

@case_mark
def case_keeping_outputs_on_the_device() -> None:
    """`copy_outputs_to_host=False` returns `DeviceView`s, which the runner reuses.

    Two things follow. First, the way back to the host is `numpy()` or
    `copy_to(buffer)` -- there is no `.cpu()` and no `.device`, which is exactly
    the `AttributeError` `../10-PyTorchTensors/` ran into.

    Second, the address is the runner's output-allocator buffer, and it is the
    same address on the next inference. Holding a returned `DeviceView` across
    two `infer` calls therefore hands you the *second* result under the first
    result's name -- the device-side twin of the buffer reuse in
    `../04-ExtendInterop/`, and harder to spot because nothing about a
    `DeviceView` looks like a value.

    The upside is chaining: one runner's output view feeds straight into the
    next runner with no host memory involved at all.
    """
    engine = build_mnist_engine()
    other_input = mnist_input * 0.0  # a visibly different input

    with TrtRunner(engine) as runner:
        first = runner.infer({"x": mnist_input}, copy_outputs_to_host=False)
        kept = first["y"]
        first_values = kept.numpy()

        print(f"    output type: {type(kept).__name__}, has .cpu(): {hasattr(kept, 'cpu')}, has .device: {hasattr(kept, 'device')}")
        print(f"    back to host via numpy(): {np.round(first_values[0, :4], 3).tolist()} ...")

        second = runner.infer({"x": other_input}, copy_outputs_to_host=False)
        print(f"    output address is the same buffer both times: {kept.ptr == second['y'].ptr}")
        print(f"    the reference we kept now reads              : {np.round(kept.numpy()[0, :4], 3).tolist()} ...")
        print(f"    ...which is the *second* result              : {np.array_equal(kept.numpy(), second['y'].numpy())}")
        print(f"    ...and no longer the first                   : {np.array_equal(kept.numpy(), first_values)}")

    # Chaining: feed one engine's device output into another engine, no host hop.
    with TrtRunner(build_mnist_engine()) as producer, TrtRunner(build_mnist_engine()) as consumer:
        produced = producer.infer({"x": mnist_input}, copy_outputs_to_host=False)
        chained = consumer.infer({"x": DeviceView(produced["z"].ptr, mnist_input.shape, mnist_input.dtype)})
        print(f"    chained a device output straight into a second runner: got {list(chained.keys())}")
    return

@case_mark
def case_what_the_round_trip_actually_costs() -> None:
    """The measurement that decides whether any of this is worth the trouble.

    Four combinations of (input on host / on device) x (output to host / left on
    device), on two models.

    On MNIST the tensors are ~3 KB, so almost none of the 0.18 ms saved is
    bandwidth -- it is the fixed cost of issuing two copies and synchronizing,
    and 1.46x of a 0.58 ms call is not what anyone is tuning for. On the 25 MB
    tensor the arithmetic is one add per element and everything else is PCIe:
    32x, with 97% of the original wall clock being copies.

    So the technique scales with bytes, not with how clever it feels. Below
    roughly a megabyte the payoff is a fixed fraction of a millisecond, and it
    is not obviously worth the lifetime hazards in the next two cases.
    """
    for label, engine, host_array, shape in [
        ("MNIST  3 KB", build_mnist_engine(), mnist_input, mnist_input.shape),
        ("synthetic 25 MB", EngineFromNetwork(big_network, config=CreateConfig())(), np.ones(BIG_SHAPE, dtype=np.float32), BIG_SHAPE),
    ]:
        with DeviceArray(shape=shape, dtype=np.float32) as array:
            array.copy_from(host_array)
            view = array.view()

            latency = {}
            for tag, feed, to_host in [
                ("host in, host out  ", host_array, True),
                ("host in, device out", host_array, False),
                ("device in, host out", view, True),
                ("device in, dev out ", view, False),
            ]:
                with TrtRunner(engine) as runner:
                    latency[tag] = benchmark(lambda feed=feed, to_host=to_host, runner=runner: runner.infer({"x": feed}, copy_outputs_to_host=to_host))

            print(f"    {label} ({host_array.nbytes / 1e6:.3f} MB in)")
            for tag, value in latency.items():
                print(f"      {tag}: {value:7.3f} ms")
            slowest, fastest = latency["host in, host out  "], latency["device in, dev out "]
            print(f"      staying on the device is {slowest / fastest:5.2f}x, saving {slowest - fastest:.3f} ms -- {(slowest - fastest) / slowest * 100:.0f}% of the round trip was copying")
    return

@case_mark
def case_a_view_does_not_keep_the_memory_alive() -> None:
    """A `DeviceView` is an integer. Nothing stops the memory under it going away.

    Two ways it goes away, both of them things the owner does to itself:

    `free()` -- the view keeps the old address, CUDA hands that same address to
    the next allocation, and the view now reads someone else's tensor. No
    exception, no warning, just different numbers.

    `resize()` to something larger -- `resize` frees and re-mallocs internally,
    so any view taken beforehand is dangling too. This one is run in a child
    process because when the new allocation lands elsewhere, reading the view
    takes the process down with SIGSEGV rather than raising anything Python can
    catch.

    Neither is visible in the `DeviceView` itself, which is why the rule has to
    be structural: a view must not outlive its `DeviceArray`. Note this cuts
    against the context-manager habit -- `with DeviceArray(...) as a:` frees on
    exit, so a view that escapes the `with` block is already dangling.
    """
    array = DeviceArray(shape=(4, ), dtype=np.float32)
    array.copy_from(np.full(4, 1.0, dtype=np.float32))
    view = array.view()
    print(f"    view of a live array at {hex(view.ptr)} reads {view.numpy().tolist()}")

    array.free()
    print(f"    after free(): owner ptr is {array.ptr}, the view still says {hex(view.ptr)}")

    reused = DeviceArray(shape=(4, ), dtype=np.float32)
    reused.copy_from(np.full(4, 9.0, dtype=np.float32))
    print(f"    a fresh allocation landed at {hex(reused.ptr)} -- same address: {reused.ptr == view.ptr}")
    print(f"    the stale view now reads {view.numpy().tolist()} and reports no error at all")
    reused.free()

    child = textwrap.dedent("""
        import numpy as np
        from polygraphy.cuda import DeviceArray
        from polygraphy.logger import G_LOGGER
        G_LOGGER.module_severity = G_LOGGER.ERROR
        array = DeviceArray(shape=(4,), dtype=np.float32)
        array.copy_from(np.full(4, 2.0, dtype=np.float32))
        view = array.view()
        array.resize((256 * 1024 * 1024,))   # frees and re-mallocs elsewhere
        print(view.numpy())
        """)
    result = subprocess.run([sys.executable, "-c", child], capture_output=True, text=True)
    print(f"    reading a view across resize(), in a child process: returncode {result.returncode} (-11 is SIGSEGV)")
    print(f"    stdout {result.stdout.strip()!r}, python-level traceback: {'Traceback' in result.stderr}")
    return

@case_mark
def case_the_stream_that_is_not_asynchronous() -> None:
    """`copy_from(..., stream)` is only asynchronous if the host buffer is pinned.

    `cudaMemcpyAsync` on ordinary pageable memory has to stage through a driver
    buffer, so it blocks until the host side is consumed. With a NumPy array the
    call therefore behaves synchronously and omitting `stream.synchronize()`
    still yields correct data -- the bug is written but cannot be observed.

    Pin the host buffer and the call returns immediately, which is the win; it
    also means the same code, unchanged, now reads a half-written buffer. The
    D2H half below prints how much of a 256 MB copy had landed before the
    synchronize. That is a race, so the exact number moves, but "not all of it"
    is reliable, and it is the failure the pageable version was hiding.

    Rule: pass a `Stream` and you own the `synchronize()`, whether or not the
    current buffer type makes forgetting it survivable.
    """
    count = RACE_NBYTES // 4
    pageable = np.ones(count, dtype=np.float32)
    pinned = torch.ones(count, dtype=torch.float32).pin_memory()

    with DeviceArray(shape=(count, ), dtype=np.float32) as array, Stream() as stream:
        for tag, buffer in [("pageable (numpy)", pageable), ("pinned   (torch)", pinned)]:
            array.copy_from(buffer, stream)  # warm up the path
            stream.synchronize()

            start = time.perf_counter()
            array.copy_from(buffer, stream)
            returned = (time.perf_counter() - start) * 1000
            stream.synchronize()
            total = (time.perf_counter() - start) * 1000
            print(f"    H2D {RACE_NBYTES // 1024 // 1024} MB, {tag}: copy_from returned after {returned:6.3f} ms, synchronized at {total:6.3f} ms")

        array.copy_from(np.full(count, 7.0, dtype=np.float32))

        for tag, host_buffer in [("pageable (numpy)", np.zeros(count, dtype=np.float32)), ("pinned   (torch)", torch.zeros(count, dtype=torch.float32).pin_memory())]:
            array.copy_to(host_buffer, stream)
            landed_early = int((host_buffer == 7.0).sum())
            stream.synchronize()
            landed_after = int((host_buffer == 7.0).sum())
            print(f"    D2H {tag}: {landed_early / count:6.1%} of the data was there before synchronize(), {landed_after / count:6.1%} after")
    return

if __name__ == "__main__":
    case_owning_versus_borrowing()
    case_feeding_the_runner_without_a_copy()
    case_keeping_outputs_on_the_device()
    case_what_the_round_trip_actually_costs()
    case_a_view_does_not_keep_the_memory_alive()
    case_the_stream_that_is_not_asynchronous()

    print("\nFinish")
