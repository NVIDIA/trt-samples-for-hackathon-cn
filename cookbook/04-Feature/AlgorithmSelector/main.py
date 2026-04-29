# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import tensorrt as trt
from tensorrt_cookbook import (CookbookAlgorithmSelector, TRTWrapperV1, case_mark, datatype_cast)

trt_file = Path("engine.trt")

@case_mark
def case_simple():
    data = {"x": np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)}
    callback = {"algorithm_selector": CookbookAlgorithmSelector(i_strategy=0)}
    tw = TRTWrapperV1(callback_object_dict=callback)

    w = trt.Weights(np.ones((1, 1, 3, 3), dtype=np.float32))
    b = trt.Weights(np.zeros((1, ), dtype=np.float32))

    x = tw.network.add_input("x", datatype_cast(data["x"].dtype, "trt"), data["x"].shape)
    conv = tw.network.add_convolution_nd(x, 1, [3, 3], w, b)

    tw.build([conv.get_output(0)])
    tw.serialize_engine(trt_file)

if __name__ == "__main__":

    case_simple()  # TODO: check this

    print("Finish")
