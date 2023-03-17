#
# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
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
#

import ctypes
import numpy as np
from cuda import cudart

data = np.zeros([2, 3], dtype=np.float32)
nElement = np.prod(data.shape)
nByteSize = np.nbytes[np.float32] * nElement

# 申请页锁定内存（Pinned memory）
_, pBuffer = cudart.cudaHostAlloc(nByteSize, cudart.cudaHostAllocWriteCombined)

# 将页锁定内存数组映射到 numpy 数组上，并使用 numpy 的方法写入新数据
pBufferCtype = ctypes.cast(pBuffer, ctypes.POINTER(ctypes.c_float * nElement))

numpyArray = np.ndarray(shape=data.shape, buffer=pBufferCtype[0], dtype=np.float32)

for i in range(nElement):
    numpyArray.reshape(-1)[i] = i

# 将页锁定内存数组拷贝到另一个 numpy 数组上，并打印
anotherArray = np.zeros(data.shape, dtype=np.float32)

cudart.cudaMemcpy(anotherArray.ctypes.data, pBuffer, nByteSize, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

print(anotherArray)
