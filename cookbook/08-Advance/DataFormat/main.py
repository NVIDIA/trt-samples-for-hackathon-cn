# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import tensorrt as trt

from tensorrt_cookbook import (TRTWrapperV1, case_mark, ceil_divide, check_array, round_up)

@case_mark
def case_(shape, data_type, format):
    tw = TRTWrapperV1()
    tw.config.set_flag(trt.BuilderFlag.DIRECT_IO)
    if data_type == trt.DataType.HALF:
        tw.config.set_flag(trt.BuilderFlag.FP16)
    if data_type == trt.DataType.INT8:
        tw.config.set_flag(trt.BuilderFlag.INT8)
    """
    # Not support yet
    if data_type == trt.DataType.BF16:
        tw.config.set_flag(trt.BuilderFlag.BF16)
    if data_type == trt.DataType.FP8:
        tw.config.set_flag(trt.BuilderFlag.FP8)
    if data_type == trt.DataType.INT4:
        tw.config.set_flag(trt.BuilderFlag.INT4)
    """

    inputT0 = tw.network.add_input("inputT0", trt.float32, shape)
    if data_type == trt.DataType.INT8:
        inputT0.set_dynamic_range(0, 384)

    layer = tw.network.add_identity(inputT0)
    layer.set_output_type(0, data_type)

    tw.network.mark_output(layer.get_output(0))
    output_tensor = tw.network.get_output(0)
    output_tensor.name = "outputT0"
    output_tensor.dtype = data_type
    tw.network.get_output(0).allowed_formats = 1 << int(format)

    tw.build()

    input_data = {"inputT0": np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + 1}

    tw.setup(input_data, b_print_io=False)
    tw.infer()

    # Check correctness manually
    ii, oo = tw.buffer["inputT0"][0], tw.buffer["outputT0"][0]  # Just for short name

    if (data_type, format) == (trt.DataType.FLOAT, trt.TensorFormat.LINEAR):  # Use `match-case` when yapf supports
        check_array(oo, ii)
        check_array(ii, oo)

    elif (data_type, format) == (trt.DataType.HALF, trt.TensorFormat.CHW2):
        if shape[1] % 2 == 0:  # no pad
            check_array(oo, ii.reshape(shape[0], shape[1] // 2, 2, shape[2], shape[3]).transpose(0, 1, 3, 4, 2).reshape(shape))
            check_array(ii, oo.reshape(shape[0], shape[1] // 2, shape[2], shape[3], 2).transpose(0, 1, 4, 2, 3).reshape(shape))
        else:  # need pad, also correct for shape[1] % 2 == 0
            n_tile, n_padded = ceil_divide(shape[1], 2), round_up(shape[1], 2)
            n_pad_width = n_padded - shape[1]
            buffer = np.concatenate([ii, np.zeros([shape[0], n_pad_width, shape[2], shape[3]], dtype=ii.dtype)], axis=1)
            buffer = buffer.reshape(shape[0], n_tile, 2, shape[2], shape[3]).transpose(0, 1, 3, 4, 2).reshape(shape[0], n_padded, shape[2], shape[3])[:, :shape[1], :, :]
            check_array(oo, buffer)
            buffer = np.concatenate([oo, np.zeros([shape[0], n_pad_width, shape[2], shape[3]], dtype=oo.dtype)], axis=1)
            buffer = buffer.reshape(shape[0], n_tile, shape[2], shape[3], 2).transpose(0, 1, 4, 2, 3).reshape(shape[0], n_padded, shape[2], shape[3])[:, :shape[1], :, :]
            nonZero = np.nonzero(buffer)
            check_array(ii[nonZero], buffer[nonZero])  # lose the last ((c + 1) // 2 * h * w - c * h * w // 2) element

    elif (data_type, format) == (trt.DataType.HALF, trt.TensorFormat.HWC8):
        if shape[1] % 8 == 0:  # no pad
            check_array(oo, ii.reshape(shape[0], shape[1] // 8, 8, shape[2], shape[3]).transpose(0, 1, 3, 4, 2).reshape(shape))
            check_array(ii, oo.reshape(shape[0], shape[1] // 8, shape[2], shape[3], 8).transpose(0, 1, 4, 2, 3).reshape(shape))
        else:  # need pad, also correct for shape[1] % 8 == 0
            n_tile, n_padded = ceil_divide(shape[1], 8), round_up(shape[1], 8)
            n_pad_width = n_padded - shape[1]
            buffer = np.concatenate([ii, np.zeros([shape[0], n_pad_width, shape[2], shape[3]], dtype=ii.dtype)], axis=1)
            buffer = buffer.transpose(0, 2, 3, 1).reshape(shape[0], n_padded, shape[2], shape[3])[:, :shape[1], :, :]
            check_array(oo, buffer)
            buffer = np.concatenate([oo, np.zeros([shape[0], n_pad_width, shape[2], shape[3]], dtype=oo.dtype)], axis=1)
            buffer = buffer.reshape(shape[0], n_tile, shape[2], shape[3], 8).transpose(0, 1, 4, 2, 3).reshape(shape[0], n_padded, shape[2], shape[3])[:, :shape[1], :, :]
            nonZero = np.nonzero(buffer)
            check_array(ii[nonZero], buffer[nonZero])  # lose the last ((c + 7) // 8 * 8) * (h * w-1) element

    elif (data_type, format) == (trt.DataType.HALF, trt.TensorFormat.CHW4):
        if shape[1] % 4 == 0:  # no pad
            check_array(oo, ii.reshape(shape[0], shape[1] // 4, 4, shape[2], shape[3]).transpose(0, 1, 3, 4, 2).reshape(shape))
            check_array(ii, oo.reshape(shape[0], shape[1] // 4, shape[2], shape[3], 4).transpose(0, 1, 4, 2, 3).reshape(shape))
        else:  # need pad, also correct for shape[1] % 4 == 0
            n_tile, n_padded = ceil_divide(shape[1], 4), round_up(shape[1], 4)
            n_pad_width = n_padded - shape[1]
            buffer = np.concatenate([ii, np.zeros([shape[0], n_pad_width, shape[2], shape[3]], dtype=ii.dtype)], axis=1)
            buffer = buffer.reshape(shape[0], n_tile, 4, shape[2], shape[3]).transpose(0, 1, 3, 4, 2).reshape(shape[0], n_padded, shape[2], shape[3])[:, :shape[1], :, :]
            check_array(oo, buffer)
            buffer = np.concatenate([oo, np.zeros([shape[0], n_pad_width, shape[2], shape[3]], dtype=oo.dtype)], axis=1)
            buffer = buffer.reshape(shape[0], n_tile, shape[2], shape[3], 4).transpose(0, 1, 4, 2, 3).reshape(shape[0], n_padded, shape[2], shape[3])[:, :shape[1], :, :]
            nonZero = np.nonzero(buffer)
            check_array(ii[nonZero], buffer[nonZero])  # lose the last ((c + 1) // 4 * h * w - c * h * w // 4) element

    elif (data_type, format) == (trt.DataType.HALF, trt.TensorFormat.CHW16):
        if shape[1] % 16 == 0:  # no pad
            check_array(oo, ii.reshape(shape[0], shape[1] // 16, 16, shape[2], shape[3]).transpose(0, 1, 3, 4, 2).reshape(shape))
            check_array(ii, oo.reshape(shape[0], shape[1] // 16, shape[2], shape[3], 16).transpose(0, 1, 4, 2, 3).reshape(shape))
        else:  # need pad, also correct for shape[1] % 16 == 0
            n_tile, n_padded = ceil_divide(shape[1], 16), round_up(shape[1], 16)
            n_pad_width = n_padded - shape[1]
            buffer = np.concatenate([ii, np.zeros([shape[0], n_pad_width, shape[2], shape[3]], dtype=ii.dtype)], axis=1)
            buffer = buffer.reshape(shape[0], n_tile, 16, shape[2], shape[3]).transpose(0, 1, 3, 4, 2).reshape(shape[0], n_padded, shape[2], shape[3])[:, :shape[1], :, :]
            check_array(oo, buffer)
            buffer = np.concatenate([oo, np.zeros([shape[0], n_pad_width, shape[2], shape[3]], dtype=oo.dtype)], axis=1)
            buffer = buffer.reshape(shape[0], n_tile, shape[2], shape[3], 16).transpose(0, 1, 4, 2, 3).reshape(shape[0], n_padded, shape[2], shape[3])[:, :shape[1], :, :]
            nonZero = np.nonzero(buffer)
            check_array(ii[nonZero], buffer[nonZero])  # lose the last ((c + 1) // 16 * h * w - c * h * w // 16) element

    elif (data_type, format) == (trt.DataType.FLOAT, trt.TensorFormat.CHW32):
        if shape[1] % 32 == 0:  #  no pad
            check_array(oo, ii.reshape(shape[0], shape[1] // 32, 32, shape[2], shape[3]).transpose(0, 1, 3, 4, 2).reshape(shape))
            check_array(ii, oo.reshape(shape[0], shape[1] // 32, shape[2], shape[3], 32).transpose(0, 1, 4, 2, 3).reshape(shape))
        else:  # need pad, also correct for shape[1] % 32 == 0
            n_tile, n_padded = ceil_divide(shape[1], 32), round_up(shape[1], 32)
            n_pad_width = n_padded - shape[1]
            buffer = np.concatenate([ii, np.zeros([shape[0], n_pad_width, shape[2], shape[3]], dtype=ii.dtype)], axis=1)
            buffer = buffer.reshape(shape[0], n_tile, 32, shape[2], shape[3]).transpose(0, 1, 3, 4, 2).reshape(shape[0], n_padded, shape[2], shape[3])[:, :shape[1], :, :]
            check_array(oo, buffer)
            buffer = np.concatenate([oo, np.zeros([shape[0], n_pad_width, shape[2], shape[3]], dtype=oo.dtype)], axis=1)
            buffer = buffer.reshape(shape[0], n_tile, shape[2], shape[3], 32).transpose(0, 1, 4, 2, 3).reshape(shape[0], n_padded, shape[2], shape[3])[:, :shape[1], :, :]
            nonZero = np.nonzero(buffer)
            check_array(ii[nonZero], buffer[nonZero])  # lose the last ((c + 1) // 32 * h * w - c * h * w // 32) element

    elif (data_type, format) == (trt.DataType.HALF, trt.TensorFormat.DHWC8):
        if shape[1] % 8 == 0:  # no pad
            check_array(oo, ii.reshape(shape[0], shape[1] // 8, 8, shape[2], shape[3], shape[4]).transpose(0, 1, 3, 4, 5, 2).reshape(shape))
            check_array(ii, oo.reshape(shape[0], shape[1] // 8, shape[2], shape[3], shape[4], 8).transpose(0, 1, 5, 2, 3, 4).reshape(shape))
        else:  # need pad, also correct for shape[1] % 8 == 0
            n_tile, n_padded = ceil_divide(shape[1], 8), round_up(shape[1], 8)
            n_pad_width = n_padded - shape[1]
            buffer = np.concatenate([ii, np.zeros([shape[0], n_pad_width, shape[2], shape[3], shape[4]], dtype=ii.dtype)], axis=1)
            buffer = buffer.transpose(0, 2, 3, 4, 1).reshape(shape[0], n_padded, shape[2], shape[3], shape[4])[:, :shape[1], :, :]
            check_array(oo, buffer)
            buffer = np.concatenate([oo, np.zeros([shape[0], n_pad_width, shape[2], shape[3], shape[4]], dtype=oo.dtype)], axis=1)
            buffer = buffer.reshape(shape[0], n_tile, shape[2], shape[3], shape[4], 8).transpose(0, 1, 5, 2, 3, 4).reshape(shape[0], n_padded, shape[2], shape[3], shape[4])[:, :shape[1], :, :, :]
            nonZero = np.nonzero(buffer)
            check_array(ii[nonZero], buffer[nonZero])  # lose the last ((c + 7) // 8 * 8) * (h * w-1) element

    elif (data_type, format) == (trt.DataType.HALF, trt.TensorFormat.CDHW32):
        if shape[1] % 32 == 0:  #  no pad
            check_array(oo, ii.reshape(shape[0], shape[1] // 32, 32, shape[2], shape[3], shape[4]).transpose(0, 1, 3, 4, 5, 2).reshape(shape))
            check_array(ii, oo.reshape(shape[0], shape[1] // 32, shape[2], shape[3], shape[4], 32).transpose(0, 1, 5, 2, 3, 4).reshape(shape))
        else:  # need pad, also correct for shape[1] % 32 == 0
            n_tile, n_padded = ceil_divide(shape[1], 32), round_up(shape[1], 32)
            n_pad_width = n_padded - shape[1]
            buffer = np.concatenate([ii, np.zeros([shape[0], n_pad_width, shape[2], shape[3], shape[4]], dtype=ii.dtype)], axis=1)
            buffer = buffer.reshape(shape[0], n_tile, 32, shape[2], shape[3], shape[4]).transpose(0, 1, 3, 4, 5, 2).reshape(shape[0], n_padded, shape[2], shape[3], shape[4])[:, :shape[1], :, :, :]
            check_array(oo, buffer)
            buffer = np.concatenate([oo, np.zeros([shape[0], n_pad_width, shape[2], shape[3], shape[4]], dtype=oo.dtype)], axis=1)
            buffer = buffer.reshape(shape[0], n_tile, shape[2], shape[3], shape[4], 32).transpose(0, 1, 5, 2, 3, 4).reshape(shape[0], n_padded, shape[2], shape[3], shape[4])[:, :shape[1], :, :, :]
            nonZero = np.nonzero(buffer)
            check_array(ii[nonZero], buffer[nonZero])  # lose the last ((c + 1) // 32 * h * w - c * h * w // 32) element

    elif (data_type, format) == (trt.DataType.FLOAT, trt.TensorFormat.HWC):
        check_array(oo, ii.transpose(0, 2, 3, 1).reshape(shape))
        check_array(ii, oo.reshape(shape[0], shape[2], shape[3], shape[1]).transpose(0, 3, 1, 2).reshape(shape))

    elif (data_type, format) == (trt.DataType.HALF, trt.TensorFormat.HWC16):
        if shape[1] % 16 == 0:  # no pad
            check_array(oo, ii.reshape(shape[0], shape[1] // 16, 16, shape[2], shape[3]).transpose(0, 1, 3, 4, 2).reshape(shape))
            check_array(ii, oo.reshape(shape[0], shape[1] // 16, shape[2], shape[3], 16).transpose(0, 4, 1, 2, 3).reshape(shape))
        else:  # need pad, also correct for shape[1] % 16 == 0
            n_tile, n_padded = ceil_divide(shape[1], 16), round_up(shape[1], 16)
            n_pad_width = n_padded - shape[1]
            buffer = np.concatenate([ii, np.zeros([shape[0], n_pad_width, shape[2], shape[3]], dtype=ii.dtype)], axis=1)
            buffer = buffer.transpose(0, 2, 3, 1).reshape(shape[0], n_padded, shape[2], shape[3])[:, :shape[1], :, :]
            check_array(oo, buffer)
            buffer = np.concatenate([oo, np.zeros([shape[0], n_pad_width, shape[2], shape[3]], dtype=oo.dtype)], axis=1)
            buffer = buffer.reshape(shape[0], n_tile, shape[2], shape[3], 16).transpose(0, 1, 4, 2, 3).reshape(shape[0], n_padded, shape[2], shape[3])[:, :shape[1], :, :]
            nonZero = np.nonzero(buffer)
            check_array(ii[nonZero], buffer[nonZero])  # lose the last ((c + 7) // 16 * 16) * (h * w-1) element

    elif (data_type, format) == (trt.DataType.FLOAT, trt.TensorFormat.DHWC):
        check_array(oo, ii.transpose(0, 2, 3, 4, 1).reshape(shape))
        check_array(ii, oo.reshape(shape[0], shape[2], shape[3], shape[4], shape[1]).transpose(0, 4, 1, 2, 3).reshape(shape))

    else:
        print("No such combination")
        raise Exception

if __name__ == "__main__":
    case_([1, 2, 3, 4], trt.DataType.FLOAT, trt.TensorFormat.LINEAR)
    case_([1, 4, 2, 3], trt.DataType.HALF, trt.TensorFormat.CHW2)  # no pad
    case_([1, 3, 2, 3], trt.DataType.HALF, trt.TensorFormat.CHW2)  # pad 1 channel
    case_([1, 8, 2, 3], trt.DataType.HALF, trt.TensorFormat.HWC8)  # no pad
    case_([1, 7, 2, 3], trt.DataType.HALF, trt.TensorFormat.HWC8)  # pad 1 channel
    case_([1, 4, 2, 3], trt.DataType.HALF, trt.TensorFormat.CHW4)  # no pad
    case_([1, 3, 2, 3], trt.DataType.HALF, trt.TensorFormat.CHW4)  # pad 1 channel
    case_([1, 16, 2, 3], trt.DataType.HALF, trt.TensorFormat.CHW16)  # no pad
    case_([1, 15, 2, 3], trt.DataType.HALF, trt.TensorFormat.CHW16)  # pad 1 channel
    case_([1, 32, 2, 3], trt.DataType.FLOAT, trt.TensorFormat.CHW32)  # no pad
    case_([1, 31, 2, 3], trt.DataType.FLOAT, trt.TensorFormat.CHW32)  # pad 1 channel
    case_([1, 8, 1, 2, 3], trt.DataType.HALF, trt.TensorFormat.DHWC8)  # no pad
    case_([1, 7, 1, 2, 3], trt.DataType.HALF, trt.TensorFormat.DHWC8)  # pad 1 channel
    case_([1, 32, 1, 2, 3], trt.DataType.HALF, trt.TensorFormat.CDHW32)  # no pad
    case_([1, 31, 1, 2, 3], trt.DataType.HALF, trt.TensorFormat.CDHW32)  # pad 1 channel
    case_([1, 2, 3, 4], trt.DataType.FLOAT, trt.TensorFormat.HWC)
    #case_([1, 2, 3, 4], trt.DataType.FLOAT, trt.TensorFormat.DLA_LINEAR)
    #case_([1, 4, 2, 3], trt.DataType.HALF, trt.TensorFormat.DLA_HWC4)  # no pad
    #case_([1, 3, 2, 3], trt.DataType.HALF, trt.TensorFormat.DLA_HWC4)  # pad 1 channel
    case_([1, 16, 2, 3], trt.DataType.HALF, trt.TensorFormat.HWC16)  # no pad
    case_([1, 15, 2, 3], trt.DataType.HALF, trt.TensorFormat.HWC16)  # pad 1 channel
    case_([1, 2, 3, 4, 5], trt.DataType.FLOAT, trt.TensorFormat.DHWC)

    print("Finish")
