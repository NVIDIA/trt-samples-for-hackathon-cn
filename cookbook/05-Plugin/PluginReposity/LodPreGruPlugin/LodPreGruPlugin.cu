/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "LodPreGruPlugin.h"

/*
input[0]:   [int],          (nLength0),             sequence0
input[1]:   [int],          (nLength0),             sequence1
input[2]:   [int],          (nLength2),             sequence2
input[3]:   [int],          (nLength2),             sequence3
input[4]:   [int],          (nLength4),             sequence4
input[5]:   [int],          (nLength4),             sequence5
input[6]:   [int],          (nLength6),             sequence6           (7)
input[7]:   [int],          (nGroup+1),             lod0
input[8]:   [int],          (nGroup+1),             lod2
input[9]:   [int],          (nGroup+1),             lod4
input[10]:  [int],          (nGroup+1),             lod6
input[11]:  [int],          (nWidth0),              width0
input[12]:  [int],          (nWidth2),              width2
input[13]:  [int],          (nWidth4),              width4
input[14]:  [int],          (nWidth6),              width6              (8)
input[15]:  [float/float16],(nLength,nEmbed),       embedBook           (1) 16

output[0]:  [float/float16],(nGroup,nWidth0,128),   q_basic_embH
output[1]:  [float/float16],(nGroup,nWidth6,128),   pcq_basic_embH
output[2]:  [float/float16],(nGroup,nWidth0,128),   merge_q_embH
output[3]:  [float/float16],(nGroup,nWidth2,128),   merge_pt_embH
output[4]:  [float/float16],(nGroup,nWidth4,128),   merge_pa_embH
output[5]:  [float/float16],(nGroup,nWidth0,128),   merge_r_q_embH
output[6]:  [float/float16],(nGroup,nWidth2,128),   merge_r_pt_embH     (7)
output[7]:  [float/float16],(nGroup,nWidth0,1),     mask0 - 1/0
output[8]:  [float/float16],(nGroup,nWidth2,1),     mask2 - 1/0
output[9]:  [float/float16],(nGroup,nWidth4,1),     mask4 - 1/0
output[10]: [float/float16],(nGroup,nWidth6,1),     mask6 - 1/0
output[11]: [float/float16],(nGroup,nWidth0,1),     mask0 - 0/-60000
output[12]: [float/float16],(nGroup,nWidth2,1),     mask2 - 0/-60000
output[13]: [float/float16],(nGroup,nWidth4,1),     mask4 - 0/-60000    (7)
output[14]: [int],          (nGroup),               validWidth0
output[15]: [int],          (nGroup),               validWidth2
output[16]: [int],          (nGroup),               validWidth4
output[17]: [int],          (nGroup),               validWidth6         (4) 18

input[0] ━(embed)→ q_basic_embH    ━┳━(elementwiseAdd)→ merge_q_embH  ━(reverse)→ merge_r_q_embH
input[1] ━(embed)→ q_bigram0_embH  ━┛
input[2] ━(embed)→ pt_basic_embH   ━┳━(elementwiseAdd)→ merge_pt_embH ━(reverse)→ merge_r_pt_embH
input[3] ━(embed)→ pt_bigram0_embH ━┛
input[4] ━(embed)→ pa_basic_embH   ━┳━(elementwiseAdd)→ merge_pa_embH
input[5] ━(embed)→ pa_bigram0_embH ━┛
input[6] ━(embed)→ pcq_basic_embH
*/

__global__ void lodPreGruKernel(int nGroup, int nWidth0, int nWidth2, int nWidth4, int nWidth6, int *sequence0, int *sequence1, int *sequence2, int *sequence3, int *sequence4, int *sequence5, int *sequence6, int *lod0, int *lod2, int *lod4, int *lod6, float *embedBook, float *out0, float *out1, float *out2, float *out3, float *out4, float *out5, float *out6, float *out7, float *out8, float *out9, float *out10, float *out11, float *out12, float *out13, int *out14, int *out15, int *out16, int *out17)
{
    const int gx = gridDim.x, row = blockIdx.y, col = blockIdx.x, tx = threadIdx.x;
    int       lodL, lodR, src, dst0, dst1, index0, index1, nValidWidth;
    float     value0, value1, valueSum;

    lodL        = lod0[row];
    lodR        = lod0[row + 1];
    nValidWidth = lodR - lodL;
    if (col < nValidWidth)
    {
        src        = lodL + col;
        index0     = sequence0[src];
        index1     = sequence1[src];
        value0     = embedBook[index0 * N_EMBED + tx];
        value1     = embedBook[index1 * N_EMBED + tx];
        valueSum   = value0 + value1;
        dst0       = (row * nWidth0 + col) * N_EMBED + tx;
        dst1       = (row * nWidth0 + nValidWidth - 1 - col) * N_EMBED + tx;
        out0[dst0] = value0;
        out2[dst0] = valueSum;
        out5[dst1] = valueSum;
    }
    if (col == N_GRID_DIMX - 1)
    {
        dst0   = row * nWidth0 + tx;
        value0 = float(tx < nValidWidth);
        value1 = value0 * MASK_VALUE_FP32 - MASK_VALUE_FP32;
        if (tx < nWidth0)
        {
            out7[dst0]  = value0;
            out11[dst0] = value1;
        }
        if (tx == N_EMBED - 1)
            out14[row] = nValidWidth;
    }

    lodL        = lod2[row];
    lodR        = lod2[row + 1];
    nValidWidth = lodR - lodL;
    if (col < nValidWidth)
    {
        src        = lodL + col;
        index0     = sequence2[src];
        index1     = sequence3[src];
        value0     = embedBook[index0 * N_EMBED + tx];
        value1     = embedBook[index1 * N_EMBED + tx];
        valueSum   = value0 + value1;
        dst0       = (row * nWidth2 + col) * N_EMBED + tx;
        dst1       = (row * nWidth2 + nValidWidth - 1 - col) * N_EMBED + tx;
        out3[dst0] = valueSum;
        out6[dst1] = valueSum;
    }
    if (col == N_GRID_DIMX - 1)
    {
        dst0   = row * nWidth2 + tx;
        value0 = float(tx < nValidWidth);
        value1 = value0 * MASK_VALUE_FP32 - MASK_VALUE_FP32;
        if (tx < nWidth2)
        {
            out8[dst0]  = value0;
            out12[dst0] = value1;
        }
        if (tx == N_EMBED - 1)
            out15[row] = nValidWidth;
    }

    lodL        = lod4[row];
    lodR        = lod4[row + 1];
    nValidWidth = lodR - lodL;
    if (col < nValidWidth)
    {
        src        = lodL + col;
        index0     = sequence4[src];
        index1     = sequence5[src];
        value0     = embedBook[index0 * N_EMBED + tx];
        value1     = embedBook[index1 * N_EMBED + tx];
        valueSum   = value0 + value1;
        dst0       = (row * nWidth4 + col) * N_EMBED + tx;
        out4[dst0] = valueSum;
    }
    if (col == N_GRID_DIMX - 1)
    {
        dst0   = row * nWidth4 + tx;
        value0 = float(tx < nValidWidth);
        value1 = value0 * MASK_VALUE_FP32 - MASK_VALUE_FP32;
        if (tx < nWidth4)
        {
            out9[dst0]  = value0;
            out13[dst0] = value1;
        }
        if (tx == N_EMBED - 1)
            out16[row] = nValidWidth;
    }

    lodL        = lod6[row];
    lodR        = lod6[row + 1];
    nValidWidth = lodR - lodL;
    for (dst1 = col; dst1 < nValidWidth; dst1 += gx)
    {
        src        = lodL + dst1;
        index0     = sequence6[src];
        value0     = embedBook[index0 * N_EMBED + tx];
        dst0       = (row * nWidth6 + dst1) * N_EMBED + tx;
        out1[dst0] = value0;
    }
    if (col >= N_GRID_DIMX - 2)
    {
        dst1   = ((col + 2 - N_GRID_DIMX) << 7) + tx;
        dst0   = row * nWidth6 + dst1;
        value0 = float(dst1 < nValidWidth);
        if (dst1 < nWidth6)
            out10[dst0] = value0;
        if (tx == N_EMBED - 1)
            out17[row] = nValidWidth;
    }

    return;
}

__global__ void lodPreGruKernelHalf(int nGroup, int nWidth0, int nWidth2, int nWidth4, int nWidth6, int *sequence0, int *sequence1, int *sequence2, int *sequence3, int *sequence4, int *sequence5, int *sequence6, int *lod0, int *lod2, int *lod4, int *lod6, __half *embedBook, __half *out0, __half *out1, __half *out2, __half *out3, __half *out4, __half *out5, __half *out6, __half *out7, __half *out8, __half *out9, __half *out10, __half *out11, __half *out12, __half *out13, int *out14, int *out15, int *out16, int *out17)
{
    const int gx = gridDim.x, row = blockIdx.y, col = blockIdx.x, tx = threadIdx.x;
    int       lodL, lodR, src, dst0, dst1, index0, index1, nValidWidth;
    __half    value0, value1, valueSum;

    lodL        = lod0[row];
    lodR        = lod0[row + 1];
    nValidWidth = lodR - lodL;
    if (col < nValidWidth)
    {
        src        = lodL + col;
        index0     = sequence0[src];
        index1     = sequence1[src];
        value0     = embedBook[index0 * N_EMBED + tx];
        value1     = embedBook[index1 * N_EMBED + tx];
        valueSum   = value0 + value1;
        dst0       = (row * nWidth0 + col) * N_EMBED + tx;
        dst1       = (row * nWidth0 + nValidWidth - 1 - col) * N_EMBED + tx;
        out0[dst0] = value0;
        out2[dst0] = valueSum;
        out5[dst1] = valueSum;
    }
    if (col == N_GRID_DIMX - 1)
    {
        dst0   = row * nWidth0 + tx;
        value0 = __int2half_rd(tx < nValidWidth);
        value1 = __hfma(value0, __int2half_rd(MASK_VALUE_FP16), __int2half_rd(-MASK_VALUE_FP16));
        if (tx < nWidth0)
        {
            out7[dst0]  = value0;
            out11[dst0] = value1;
        }
        if (tx == N_EMBED - 1)
            out14[row] = nValidWidth;
    }

    lodL        = lod2[row];
    lodR        = lod2[row + 1];
    nValidWidth = lodR - lodL;
    if (col < nValidWidth)
    {
        src        = lodL + col;
        index0     = sequence2[src];
        index1     = sequence3[src];
        value0     = embedBook[index0 * N_EMBED + tx];
        value1     = embedBook[index1 * N_EMBED + tx];
        valueSum   = value0 + value1;
        dst0       = (row * nWidth2 + col) * N_EMBED + tx;
        dst1       = (row * nWidth2 + nValidWidth - 1 - col) * N_EMBED + tx;
        out3[dst0] = valueSum;
        out6[dst1] = valueSum;
    }
    if (col == N_GRID_DIMX - 1)
    {
        dst0   = row * nWidth2 + tx;
        value0 = __int2half_rd(tx < nValidWidth);
        value1 = __hfma(value0, __int2half_rd(MASK_VALUE_FP16), __int2half_rd(-MASK_VALUE_FP16));
        if (tx < nWidth2)
        {
            out8[dst0]  = value0;
            out12[dst0] = value1;
        }
        if (tx == N_EMBED - 1)
            out15[row] = nValidWidth;
    }

    lodL        = lod4[row];
    lodR        = lod4[row + 1];
    nValidWidth = lodR - lodL;
    if (col < nValidWidth)
    {
        src        = lodL + col;
        index0     = sequence4[src];
        index1     = sequence5[src];
        value0     = embedBook[index0 * N_EMBED + tx];
        value1     = embedBook[index1 * N_EMBED + tx];
        valueSum   = value0 + value1;
        dst0       = (row * nWidth4 + col) * N_EMBED + tx;
        out4[dst0] = valueSum;
    }
    if (col == N_GRID_DIMX - 1)
    {
        dst0   = row * nWidth4 + tx;
        value0 = __int2half_rd(tx < nValidWidth);
        value1 = __hfma(value0, __int2half_rd(MASK_VALUE_FP16), __int2half_rd(-MASK_VALUE_FP16));
        if (tx < nWidth4)
        {
            out9[dst0]  = value0;
            out13[dst0] = value1;
        }
        if (tx == N_EMBED - 1)
            out16[row] = nValidWidth;
    }

    lodL        = lod6[row];
    lodR        = lod6[row + 1];
    nValidWidth = lodR - lodL;
    for (dst1 = col; dst1 < nValidWidth; dst1 += gx)
    {
        src        = lodL + dst1;
        index0     = sequence6[src];
        value0     = embedBook[index0 * N_EMBED + tx];
        dst0       = (row * nWidth6 + dst1) * N_EMBED + tx;
        out1[dst0] = value0;
    }
    if (col >= N_GRID_DIMX - 2)
    {
        dst1   = ((col + 2 - N_GRID_DIMX) << 7) + tx;
        dst0   = row * nWidth6 + dst1;
        value0 = __int2half_rd(dst1 < nValidWidth);
        if (dst1 < nWidth6)
            out10[dst0] = value0;
        if (tx == N_EMBED - 1)
            out17[row] = nValidWidth;
    }

    return;
}

int LodPreGruPlugin::enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream)
{
    //printf("d=%2d,g=%2d,w0=%2d,w2=%2d,w4=%2d,w6=%2d\n",m.datatype,m.nGroup,m.nWidth0,m.nWidth2,m.nWidth4,m.nWidth6);
    switch (m.datatype)
    {
    case 0:
        lodPreGruKernel<<<dim3(N_GRID_DIMX, m.nGroup), N_EMBED, 0, stream>>>(m.nGroup, m.nWidth0, m.nWidth2, m.nWidth4, m.nWidth6, (int *)inputs[0], (int *)inputs[1], (int *)inputs[2], (int *)inputs[3], (int *)inputs[4], (int *)inputs[5], (int *)inputs[6], (int *)inputs[7], (int *)inputs[8], (int *)inputs[9], (int *)inputs[10], (float *)inputs[15], (float *)outputs[0], (float *)outputs[1], (float *)outputs[2], (float *)outputs[3], (float *)outputs[4], (float *)outputs[5], (float *)outputs[6], (float *)outputs[7], (float *)outputs[8], (float *)outputs[9], (float *)outputs[10], (float *)outputs[11], (float *)outputs[12], (float *)outputs[13], (int *)outputs[14], (int *)outputs[15], (int *)outputs[16], (int *)outputs[17]);
        break;
    case 1:
        lodPreGruKernelHalf<<<dim3(N_GRID_DIMX, m.nGroup), N_EMBED, 0, stream>>>(m.nGroup, m.nWidth0, m.nWidth2, m.nWidth4, m.nWidth6, (int *)inputs[0], (int *)inputs[1], (int *)inputs[2], (int *)inputs[3], (int *)inputs[4], (int *)inputs[5], (int *)inputs[6], (int *)inputs[7], (int *)inputs[8], (int *)inputs[9], (int *)inputs[10], (__half *)inputs[15], (__half *)outputs[0], (__half *)outputs[1], (__half *)outputs[2], (__half *)outputs[3], (__half *)outputs[4], (__half *)outputs[5], (__half *)outputs[6], (__half *)outputs[7], (__half *)outputs[8], (__half *)outputs[9], (__half *)outputs[10], (__half *)outputs[11], (__half *)outputs[12], (__half *)outputs[13], (int *)outputs[14], (int *)outputs[15], (int *)outputs[16], (int *)outputs[17]);
        break;
    default:
        //printf("[LodPreGruPlugin::enqueue]Error datatype!\n");
        break;
    }
    return 0;
}

REGISTER_TENSORRT_PLUGIN(LodPreGruPluginCreator);
