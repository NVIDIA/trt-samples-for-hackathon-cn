/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
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

#include "TopKAveragePlugin.h"

static constexpr int32_t WARP_SIZE       = 32;
static constexpr int32_t WARP_CNT        = 2;
static constexpr int32_t ARRAY_LEN       = 8;
static constexpr float   topKOutPadValue = -10000.0f;

__global__ void topKAverageKernelV3(const float *in_data, const int32_t *row_lod, const int32_t *col_lod, const int32_t *k_array, int32_t k_num, int32_t max_row, int32_t max_col, int32_t *pos_data, float *out_data)
{
    const int32_t max_k   = k_array[k_num - 1];
    const int32_t row_pos = blockDim.y * blockIdx.x + threadIdx.y;
    if (row_pos >= row_lod[blockIdx.z])
        return;
    int32_t                input_offset  = (blockIdx.z * gridDim.y + blockIdx.y) * max_row + row_pos;
    int32_t                output_offset = (blockIdx.z * max_row + row_pos) * gridDim.y + blockIdx.y;
    const float *          in_data_ptr   = in_data + input_offset * max_col;
    int32_t *              pos_data_ptr  = pos_data + output_offset * max_k;
    float *                out_data_ptr  = out_data + output_offset * k_num;
    extern __shared__ char shared_mem[];
    int32_t *              pos_ret_smem_ptr   = (int32_t *)(shared_mem + threadIdx.y * max_k * (sizeof(int32_t) + sizeof(float)));
    float *                score_ret_smem_ptr = (float *)(pos_ret_smem_ptr + max_k);
    const int32_t          seq_len            = col_lod[blockIdx.z];
    const int32_t          level_cnt          = (seq_len + WARP_SIZE - 1) / WARP_SIZE;
    int32_t                selected_level     = -1;
    int32_t                curr_level         = -1;
    int32_t                selected_pos       = threadIdx.x;
    float                  selected_score     = topKOutPadValue;
    float                  score_arr[ARRAY_LEN];
    score_arr[level_cnt - 1] = topKOutPadValue;
    for (int level_idx = 0; level_idx < level_cnt; ++level_idx)
    {
        int32_t curr_idx = threadIdx.x + level_idx * WARP_SIZE;
        if (curr_idx < seq_len)
        {
            score_arr[level_idx] = in_data_ptr[curr_idx];
        }
    }
    if (threadIdx.x < max_k)
    {
        pos_ret_smem_ptr[threadIdx.x]   = -1;
        score_ret_smem_ptr[threadIdx.x] = 0.0f;
    }
    int32_t loop_cnt = max_k;
    if (loop_cnt > seq_len)
    {
        loop_cnt = seq_len;
    }
    for (int32_t loop_idx = 0; loop_idx < loop_cnt; ++loop_idx)
    {
        selected_level = -1;
        selected_pos   = threadIdx.x;
        selected_score = topKOutPadValue;
        for (int level = 0; level < level_cnt; ++level)
        {
            if (selected_score < score_arr[level])
            {
                selected_level = level;
                selected_score = score_arr[level];
            }
        }
        for (int32_t offset = WARP_SIZE >> 1; offset > 0; offset >>= 1)
        {
            int32_t tmp_pos   = __shfl_down_sync(0xffffffff, selected_pos, offset);
            float   tmp_score = __shfl_down_sync(0xffffffff, selected_score, offset);
            if (selected_score < tmp_score)
            {
                selected_pos   = tmp_pos;
                selected_score = tmp_score;
            }
        }
        curr_level = __shfl_down_sync(0xffffffff, selected_level, selected_pos);
        if (threadIdx.x == 0)
        {
            pos_ret_smem_ptr[loop_idx]   = selected_pos + curr_level * WARP_SIZE;
            score_ret_smem_ptr[loop_idx] = selected_score;
        }
        selected_pos = __shfl_sync(0xffffffff, selected_pos, 0);
        if (threadIdx.x == selected_pos)
        {
            score_arr[selected_level] = topKOutPadValue;
        }
    }
    if (threadIdx.x == 0)
    {
        float   accumulate_score = 0.0f;
        int32_t curr_k_idx       = 0;
        for (int32_t idx = 0; idx < max_k; ++idx)
        {
            accumulate_score += score_ret_smem_ptr[idx];
            if (idx + 1 == k_array[curr_k_idx])
            {
                out_data_ptr[curr_k_idx++] = accumulate_score / (idx + 1);
            }
        }
    }
    if (threadIdx.x < max_k)
    {
        pos_data_ptr[threadIdx.x] = pos_ret_smem_ptr[threadIdx.x];
    }
}

__global__ void topKAverageKernelHalfV3(const __half *in_data, const int32_t *row_lod, const int32_t *col_lod, const int32_t *k_array, int32_t k_num, int32_t max_row, int32_t max_col, int32_t *pos_data, __half *out_data)
{
    const int32_t max_k   = k_array[k_num - 1];
    const int32_t row_pos = blockDim.y * blockIdx.x + threadIdx.y;
    if (row_pos >= row_lod[blockIdx.z])
        return;
    int32_t                input_offset  = (blockIdx.z * gridDim.y + blockIdx.y) * max_row + row_pos;
    int32_t                output_offset = (blockIdx.z * max_row + row_pos) * gridDim.y + blockIdx.y;
    const __half *         in_data_ptr   = in_data + input_offset * max_col;
    int32_t *              pos_data_ptr  = pos_data + output_offset * max_k;
    __half *               out_data_ptr  = out_data + output_offset * k_num;
    extern __shared__ char shared_mem[];
    int32_t *              pos_ret_smem_ptr   = (int32_t *)(shared_mem + threadIdx.y * max_k * (sizeof(int32_t) + sizeof(__half)));
    __half *               score_ret_smem_ptr = (__half *)(pos_ret_smem_ptr + max_k);
    const int32_t          seq_len            = col_lod[blockIdx.z];
    const int32_t          level_cnt          = (seq_len + WARP_SIZE - 1) / WARP_SIZE;
    int32_t                selected_level     = -1;
    int32_t                curr_level         = -1;
    int32_t                selected_pos       = threadIdx.x;
    __half                 selected_score     = topKOutPadValue;
    __half                 score_arr[ARRAY_LEN];
    score_arr[level_cnt - 1] = topKOutPadValue;
    for (int level_idx = 0; level_idx < level_cnt; ++level_idx)
    {
        int32_t curr_idx = threadIdx.x + level_idx * WARP_SIZE;
        if (curr_idx < seq_len)
        {
            score_arr[level_idx] = in_data_ptr[curr_idx];
        }
    }
    if (threadIdx.x < max_k)
    {
        pos_ret_smem_ptr[threadIdx.x]   = -1;
        score_ret_smem_ptr[threadIdx.x] = 0.0f;
    }
    int32_t loop_cnt = max_k;
    if (loop_cnt > seq_len)
    {
        loop_cnt = seq_len;
    }
    for (int32_t loop_idx = 0; loop_idx < loop_cnt; ++loop_idx)
    {
        selected_level = -1;
        selected_pos   = threadIdx.x;
        selected_score = topKOutPadValue;
        for (int level = 0; level < level_cnt; ++level)
        {
            if (selected_score < score_arr[level])
            {
                selected_level = level;
                selected_score = score_arr[level];
            }
        }
        for (int32_t offset = WARP_SIZE >> 1; offset > 0; offset >>= 1)
        {
            int32_t tmp_pos   = __shfl_down_sync(0xffffffff, selected_pos, offset);
            __half  tmp_score = __shfl_down_sync(0xffffffff, selected_score, offset);
            if (selected_score < tmp_score)
            {
                selected_pos   = tmp_pos;
                selected_score = tmp_score;
            }
        }
        curr_level = __shfl_down_sync(0xffffffff, selected_level, selected_pos);
        if (threadIdx.x == 0)
        {
            pos_ret_smem_ptr[loop_idx]   = selected_pos + curr_level * WARP_SIZE;
            score_ret_smem_ptr[loop_idx] = selected_score;
        }
        selected_pos = __shfl_sync(0xffffffff, selected_pos, 0);
        if (threadIdx.x == selected_pos)
        {
            score_arr[selected_level] = topKOutPadValue;
        }
    }
    if (threadIdx.x == 0)
    {
        __half  accumulate_score = 0.0f;
        int32_t curr_k_idx       = 0;
        for (int32_t idx = 0; idx < max_k; ++idx)
        {
            accumulate_score += score_ret_smem_ptr[idx];
            if (idx + 1 == k_array[curr_k_idx])
            {
                out_data_ptr[curr_k_idx++] = __hdiv(accumulate_score, __int2half_rd(idx + 1));
            }
        }
    }
    if (threadIdx.x < max_k)
    {
        pos_data_ptr[threadIdx.x] = pos_ret_smem_ptr[threadIdx.x];
    }
}

int TopKAveragePlugin::enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream)
{
    //printf("d=%2d,nTopK=%2d,maxTopK=%2d,g=%2d,c=%2d,h=%2d,w=%2d\n",m.datatype,m.nTopK,m.maxTopK,m.nGroup,m.nChannel,m.nHeight,m.nWidth);
    dim3    grid(m.nHeight, m.nChannel, m.nGroup), block(WARP_SIZE, WARP_CNT);
    int32_t shareMemorySize = 0;
    switch (m.datatype)
    {
    case 0:
        shareMemorySize = WARP_CNT * m.maxTopK * (sizeof(int32_t) + sizeof(float));
        topKAverageKernelV3<<<grid, block, shareMemorySize, stream>>>((float *)inputs[0], (int32_t *)inputs[1], (int32_t *)inputs[2], (int32_t *)inputs[3], m.nTopK, m.nHeight, m.nWidth, (int32_t *)workspace, (float *)outputs[0]);
        break;
    case 1:
        shareMemorySize = WARP_CNT * m.maxTopK * (sizeof(int32_t) + sizeof(__half));
        topKAverageKernelHalfV3<<<grid, block, shareMemorySize, stream>>>((__half *)inputs[0], (int32_t *)inputs[1], (int32_t *)inputs[2], (int32_t *)inputs[3], m.nTopK, m.nHeight, m.nWidth, (int32_t *)workspace, (__half *)outputs[0]);
        break;
    default:
        break;
    }
    return 0;
}

REGISTER_TENSORRT_PLUGIN(TopKAveragePluginCreator);
