import tensorrt as trt

import numpy as np

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network:

    data = network.add_input("data", trt.DataType.FLOAT, (-1, 64, 128, 128))

    input_shape = network.add_shape(data).get_output(0)

    shape_offset = network.add_constant([4], np.array([0, -61, 0, 0], dtype=np.int32)).get_output(0)

    output_shape = network.add_elementwise(input_shape, shape_offset, trt.ElementWiseOperation.SUM).get_output(0)

    slice = network.add_slice(data, start=(0, 0, 0, 0), shape=(-1, -1, -1, -1), stride=(1, 1, 1, 1))

    slice.set_input(2, output_shape)

    print('slice output shape: ', slice.get_output(0).shape)

    network.mark_output(slice.get_output(0))

    op = builder.create_optimization_profile()

    op.set_shape('data', (1, 64, 128, 128), (3, 64, 128, 128), (5, 64, 128, 128))

    config = builder.create_builder_config()

    config.add_optimization_profile(op)

    config.max_workspace_size = 1 << 30

    engine = builder.build_engine(network, config)

    in_shape = (3, 64, 128, 128)

    size = 3 * 64 * 128 * 128

    dtype = trt.nptype(engine.get_binding_dtype(0))

    contexts = []

    for i in range(3):

        context = engine.create_execution_context()

        context.active_optimization_profile = 0

        context.set_binding_shape(0, (i + 1, 64, 128, 128))

        contexts.append(context)

    for i in range(3):

        for j in range(engine.num_bindings):

            print(contexts[i].get_binding_shape(j))