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
"""`IOutputAllocator` on a detection head, where the output size is decided by the data.

`02-API/Layer/NonZero` and `02-API/Layer/NMS` already show that a layer *can* have a
data-dependent shape (DDS), and `TRTWrapperDDS` hides the allocator so those examples
stay about the layer. This example is about the allocator itself, on the workload that
actually forces you to write one: object detection, where the number of surviving boxes
is a property of the picture, not of the network.

The engine is a small detection tail:

    boxes  [1, 2048, 4]     ---+
                               +--> NMS --> selected_indices [n, 3]   (DDS)
    scores [1, 2048, 4]     ---+           num_output_boxes  []
                                            |
                                            +--> Gather --> selected_boxes [n, 4] (DDS)

Cases:

1. `case_worst_case_allocation` - what you must do *without* an allocator:
   `get_max_output_size` and allocate for the worst case, every time.
2. `case_output_allocator` - a real `IOutputAllocator` that only ever grows, run over
   three "images" with very different detection counts, printing every callback.
3. `case_shape_reporting` - where the true shape comes from, and the trap that the
   buffer is *not* the shape.
4. `case_memory_comparison` - the number that justifies the work: peak bytes held by
   the allocator against the worst-case allocation.
"""

from collections import OrderedDict

import cuda.bindings.runtime as cudart
import numpy as np
import tensorrt as trt

from tensorrt_cookbook import case_mark

np.random.seed(31193)

N_ANCHOR = 2048  # Candidate boxes coming out of the (imaginary) backbone
N_CLASS = 4
N_MAX_OUTPUT_BOX_PER_CLASS = 512  # What the builder must assume in the worst case
IOU_THRESHOLD = 0.5
SCORE_THRESHOLD = 0.5

result = OrderedDict()

# ================================================================ The allocator

class DetectionOutputAllocator(trt.IOutputAllocator):
    """A grow-only output allocator, with enough bookkeeping to see what TensorRT does.

    The contract is two methods. `reallocate_output_async` is asked for memory and must
    return a device address that holds at least `size` bytes; `notify_shape` is then told
    what shape TensorRT actually produced. Both are called **during** `execute_async_v3`,
    which is the whole point: neither answer exists before the data has been seen.

    Grow-only is the right default. A detector's box count jitters from frame to frame,
    and freeing plus re-allocating on every frame would put a `cudaMalloc` on the critical
    path for no reason. Here the buffer is kept and reused whenever it is already large
    enough, so the allocation count converges to a small constant.
    """

    def __init__(self, name: str) -> None:
        """Create an empty allocator for one output tensor."""
        super().__init__()
        self.name = name
        self.address = 0  # Device address currently owned, 0 = nothing yet
        self.n_byte = 0  # Capacity of that allocation
        self.shape = None  # Last shape reported by `notify_shape`
        self.n_reallocation = 0  # How many times we actually called `cudaMalloc`
        self.n_call = 0  # How many times TensorRT asked
        self.call_log = []  # (requested_size, grew?) for the current image, cleared per image
        self.request_history = []  # Every size ever requested, never cleared

    def reallocate_output_async(self, tensor_name, old_address, size, alignment, stream) -> int:
        """Return a device address holding at least `size` bytes."""
        self.n_call += 1
        if size <= self.n_byte and self.address != 0:  # Already big enough, hand back the same memory
            self.call_log.append((size, False))
            self.request_history.append(size)
            return self.address

        if self.address != 0:
            cudart.cudaFree(self.address)
        # `size` can legitimately be 0 when nothing was detected. `cudaMalloc(0)` returns a
        # NULL address, and a NULL output address makes `enqueueV3` fail, so round up to 1.
        status, address = cudart.cudaMalloc(max(size, 1))
        assert status == cudart.cudaError_t.cudaSuccess, f"Failed allocating {size} B for {tensor_name}"
        self.address = address
        self.n_byte = max(size, 1)
        self.n_reallocation += 1
        self.call_log.append((size, True))
        self.request_history.append(size)
        return self.address

    def notify_shape(self, tensor_name, shape) -> None:
        """Receive the shape TensorRT really produced for this tensor."""
        self.shape = tuple(shape)
        return

    def free(self) -> None:
        """Release the device memory this allocator owns."""
        if self.address != 0:
            cudart.cudaFree(self.address)
            self.address, self.n_byte = 0, 0
        return

# ================================================================ The engine

def build_detection_engine() -> bytes:
    """A detection tail: NMS over `N_ANCHOR` boxes, then gather the surviving boxes."""
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    builder_config = builder.create_builder_config()

    boxes = network.add_input("boxes", trt.float32, [1, N_ANCHOR, 4])
    scores = network.add_input("scores", trt.float32, [1, N_ANCHOR, N_CLASS])

    layer_max_output = network.add_constant([], np.array([N_MAX_OUTPUT_BOX_PER_CLASS], dtype=np.int32))
    layer_iou = network.add_constant([], np.array([IOU_THRESHOLD], dtype=np.float32))
    layer_score = network.add_constant([], np.array([SCORE_THRESHOLD], dtype=np.float32))

    layer_nms = network.add_nms(boxes, scores, layer_max_output.get_output(0), trt.DataType.INT32)
    layer_nms.set_input(3, layer_iou.get_output(0))
    layer_nms.set_input(4, layer_score.get_output(0))
    layer_nms.bounding_box_format = trt.BoundingBoxFormat.CORNER_PAIRS
    selected_indices = layer_nms.get_output(0)  # [n_selected, 3] = (batch, class, box), DDS
    num_output_boxes = layer_nms.get_output(1)  # [] scalar, NOT data dependent

    # Turn the indices into the thing a caller actually wants: the surviving boxes.
    # Column 2 of `selected_indices` is the anchor index.
    # A Slice cannot be used here: its size input would have to carry the DDS row count in
    # dimension 0 while pinning dimension 1 to 1, and feeding `shape(selected_indices)`
    # straight in makes the builder reject it with "ISliceLayer has out of bounds access on
    # axis 1". Gathering column 2 keeps the DDS dimension untouched and needs no shape maths.
    layer_column = network.add_constant([1], np.array([2], dtype=np.int32))
    layer_box_index = network.add_gather(selected_indices, layer_column.get_output(0), 1)  # [n, 3] -> [n, 1]
    layer_squeeze = network.add_shuffle(layer_box_index.get_output(0))
    layer_squeeze.reshape_dims = [-1]

    layer_boxes_2d = network.add_shuffle(boxes)  # [1, N_ANCHOR, 4] -> [N_ANCHOR, 4]
    layer_boxes_2d.reshape_dims = [N_ANCHOR, 4]
    layer_gather = network.add_gather(layer_boxes_2d.get_output(0), layer_squeeze.get_output(0), 0)
    selected_boxes = layer_gather.get_output(0)  # [n_selected, 4], DDS

    selected_indices.name = "selected_indices"
    num_output_boxes.name = "num_output_boxes"
    selected_boxes.name = "selected_boxes"
    for tensor in [selected_indices, num_output_boxes, selected_boxes]:
        network.mark_output(tensor)

    engine_bytes = builder.build_serialized_network(network, builder_config)
    assert engine_bytes is not None, "Failed building the detection engine"
    return bytes(engine_bytes)

def make_image(n_object: int) -> dict:
    """Fake one image's detection head output with roughly `n_object` confident boxes.

    Boxes are placed far apart so NMS keeps them instead of merging them, which makes the
    surviving count predictable enough to reason about.
    """
    boxes = np.zeros((1, N_ANCHOR, 4), dtype=np.float32)
    scores = np.full((1, N_ANCHOR, N_CLASS), 0.01, dtype=np.float32)

    side = int(np.ceil(np.sqrt(max(n_object, 1))))
    for i in range(min(n_object, N_ANCHOR)):
        x, y = (i % side) * 20.0, (i // side) * 20.0
        boxes[0, i] = [x, y, x + 8.0, y + 8.0]  # 8x8 boxes on a 20-pixel grid, no overlap
        scores[0, i, i % N_CLASS] = 0.9
    return {"boxes": boxes, "scores": scores}

# ================================================================ Runtime helpers

def setup_context(engine):
    """Create a context and bind the two (fixed-shape) inputs."""
    context = engine.create_execution_context()
    name_list = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    input_buffer = {}
    for name in name_list:
        if engine.get_tensor_mode(name) != trt.TensorIOMode.INPUT:
            continue
        shape = engine.get_tensor_shape(name)
        n_byte = trt.volume(shape) * engine.get_tensor_dtype(name).itemsize
        input_buffer[name] = cudart.cudaMalloc(n_byte)[1]
        context.set_tensor_address(name, input_buffer[name])
    return context, name_list, input_buffer

def upload_inputs(context, input_buffer, data) -> None:
    """Copy one image's inputs to the device."""
    for name, address in input_buffer.items():
        array = np.ascontiguousarray(data[name])
        cudart.cudaMemcpy(address, array.ctypes.data, array.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
    return

def download(address, shape, dtype) -> np.ndarray:
    """Read a DDS output back, using the shape the allocator was told."""
    host = np.empty(shape, dtype=dtype)
    if host.nbytes > 0:
        cudart.cudaMemcpy(host.ctypes.data, address, host.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
    return host

# ================================================================ Cases

@case_mark
def case_worst_case_allocation() -> None:
    """Life without an allocator: ask how bad it could get, and pay for that.

    `get_max_output_size` is the upper bound the builder derived, so a caller that will
    not implement `IOutputAllocator` has to allocate this much for every DDS output, on
    every context, regardless of what the picture actually contains.
    """
    engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(build_detection_engine())
    context, name_list, input_buffer = setup_context(engine)

    total = 0
    for name in name_list:
        if engine.get_tensor_mode(name) != trt.TensorIOMode.OUTPUT:
            continue
        shape = context.get_tensor_shape(name)
        is_dds = -1 in tuple(shape)
        n_byte = context.get_max_output_size(name) if is_dds else trt.volume(shape) * engine.get_tensor_dtype(name).itemsize
        total += n_byte
        print(f"    {name:<20} build-time shape={str(tuple(shape)):<12} DDS={is_dds!s:<5} worst-case={n_byte:>10} B")
    print(f"    Worst-case output memory for one context: {total} B ({total / (1 << 20):.2f} MiB)")

    result["worst_case_byte"] = total
    for address in input_buffer.values():
        cudart.cudaFree(address)
    return

@case_mark
def case_output_allocator() -> None:
    """The real thing: one allocator per DDS output, three images, every callback printed."""
    engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(build_detection_engine())
    context, name_list, input_buffer = setup_context(engine)

    allocator_map = OrderedDict()
    plain_output = OrderedDict()
    for name in name_list:
        if engine.get_tensor_mode(name) != trt.TensorIOMode.OUTPUT:
            continue
        if -1 in tuple(context.get_tensor_shape(name)):
            allocator_map[name] = DetectionOutputAllocator(name)
            context.set_output_allocator(name, allocator_map[name])  # This is the whole integration
        else:  # `num_output_boxes` is an ordinary fixed-shape output, it needs no allocator
            n_byte = trt.volume(context.get_tensor_shape(name)) * engine.get_tensor_dtype(name).itemsize
            plain_output[name] = cudart.cudaMalloc(n_byte)[1]
            context.set_tensor_address(name, plain_output[name])

    print(f"    DDS outputs needing an allocator: {list(allocator_map)}")
    print(f"    Fixed-shape outputs             : {list(plain_output)}")

    row_list = []
    for n_object in [3, 200, 17]:  # Grow a lot, then shrink: the shrink must NOT reallocate
        upload_inputs(context, input_buffer, make_image(n_object))
        for allocator in allocator_map.values():
            allocator.call_log.clear()
        context.execute_async_v3(0)
        cudart.cudaStreamSynchronize(0)

        n_selected = download(plain_output["num_output_boxes"], (), np.int32).item() if "num_output_boxes" in plain_output else -1
        shape = allocator_map["selected_boxes"].shape
        print(f"    image with {n_object:>3} objects -> num_output_boxes={n_selected}, selected_boxes shape={shape}")
        for name, allocator in allocator_map.items():
            for size, grew in allocator.call_log:
                print(f"        {name:<20} reallocate_output_async(size={size:>8}) -> {'cudaMalloc (grew)' if grew else 'reused existing buffer'}")
        row_list.append((n_object, n_selected, {name: a.n_byte for name, a in allocator_map.items()}))

    print(f"    Allocation counts: {[(name, a.n_reallocation, a.n_call) for name, a in allocator_map.items()]} (cudaMalloc calls, total requests)")
    # Not all DDS outputs are equal: `selected_indices` comes straight off the NMS layer and
    # TensorRT asks for its **upper bound** every single time, no matter what the image holds.
    # `selected_boxes` is a Gather downstream of it and is requested at its true size. So an
    # allocator only saves memory on the second kind, which is why the ratio below is 2x and
    # not 100x. Worth knowing before designing a memory budget around "DDS = pay for what you use".
    request_set = {name: sorted(set(allocator.request_history)) for name, allocator in allocator_map.items()}
    print(f"    Sizes requested across the three images: {request_set}")
    result["peak_byte"] = sum(a.n_byte for a in allocator_map.values())
    result["row_list"] = row_list
    result["n_reallocation"] = {name: a.n_reallocation for name, a in allocator_map.items()}

    for allocator in allocator_map.values():
        allocator.free()
    for address in list(input_buffer.values()) + list(plain_output.values()):
        cudart.cudaFree(address)
    return

@case_mark
def case_shape_reporting() -> None:
    """Where the shape comes from, and two traps.

    After inference there are three different "sizes" in play and only **one** of them is
    the answer:

    + `allocator.shape`      - what `notify_shape` was told. This is the answer.
    + `allocator.n_byte`     - how much memory is held. Grow-only, so it is the high-water
                               mark of every image seen so far, **not** this image's size.
    + `context.get_tensor_shape(name)` - still returns `(-1, 4)` *after* a completed
                               inference. Unlike a merely dynamic shape, a data-dependent
                               one is never resolved on the context; `notify_shape` is the
                               only channel. Code ported from a dynamic-shape engine, where
                               `get_tensor_shape` is the normal way to size the copy back,
                               will read -1 here and compute a negative or zero byte count.

    Reading `n_byte` and dividing by the item size is the other natural mistake, and it
    silently returns stale boxes from a previous, larger image.
    """
    engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(build_detection_engine())
    context, name_list, input_buffer = setup_context(engine)

    allocator = DetectionOutputAllocator("selected_boxes")
    context.set_output_allocator("selected_boxes", allocator)
    other = {}
    for name in ["selected_indices", "num_output_boxes"]:
        if -1 in tuple(context.get_tensor_shape(name)):
            other[name] = DetectionOutputAllocator(name)
            context.set_output_allocator(name, other[name])
        else:
            n_byte = trt.volume(context.get_tensor_shape(name)) * engine.get_tensor_dtype(name).itemsize
            other[name] = cudart.cudaMalloc(n_byte)[1]
            context.set_tensor_address(name, other[name])

    for n_object in [200, 5]:  # Big image first, so the buffer is oversized for the small one
        upload_inputs(context, input_buffer, make_image(n_object))
        context.execute_async_v3(0)
        cudart.cudaStreamSynchronize(0)
        capacity_row = allocator.n_byte // (4 * np.dtype(np.float32).itemsize)
        context_shape = tuple(context.get_tensor_shape("selected_boxes"))
        print(f"    {n_object:>3} objects: notify_shape={allocator.shape}, context.get_tensor_shape={context_shape}, "
              f"buffer capacity would suggest {capacity_row} rows")
        assert -1 in context_shape, "Expected the context to keep reporting -1 for a DDS output"
        if n_object == 5:
            assert allocator.shape[0] < capacity_row, "This case is meant to show capacity > shape"
            print(f"        -> trusting the buffer size would report {capacity_row} boxes instead of {allocator.shape[0]}")
            print(f"        -> trusting context.get_tensor_shape would give {context_shape}, i.e. a negative byte count")

    allocator.free()
    for value in other.values():
        if isinstance(value, DetectionOutputAllocator):
            value.free()
        else:
            cudart.cudaFree(value)
    for address in input_buffer.values():
        cudart.cudaFree(address)
    return

@case_mark
def case_memory_comparison() -> None:
    """What the allocator bought, in bytes."""
    worst = result["worst_case_byte"]
    peak = result["peak_byte"]
    print(f"    Worst-case pre-allocation : {worst:>10} B ({worst / (1 << 20):.2f} MiB)")
    print(f"    Peak held by the allocator: {peak:>10} B ({peak / (1 << 20):.3f} MiB)")
    print(f"    Saved                     : {worst - peak:>10} B, {worst / max(peak, 1):.1f}x smaller")
    print(f"    cudaMalloc calls over 3 images (grow, grow, shrink): {result['n_reallocation']}")
    return

# ================================================================ Entrance

def main() -> None:
    case_worst_case_allocation()
    case_output_allocator()
    case_shape_reporting()
    case_memory_comparison()
    return

if __name__ == "__main__":
    main()
    print("\nFinish")
