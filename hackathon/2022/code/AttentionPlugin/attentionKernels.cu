#include "attentionKernels.h"


inline void myAssert(bool result, const char* const file, int const line, std::string info = "")
{
    if (result != true) {
        throw std::runtime_error(std::string("[FT][ERROR] ") + info + std::string(" Assertion fail: ") + file + ":"
                                 + std::to_string(line) + " \n");
    }
}

#define FT_CHECK(val) myAssert(val, __FILE__, __LINE__)

template<int ITEMS_PER_THREAD, typename T, typename T_IN>
__global__ void add_softmax_kernel_v4(T* qk_buf_,
                                      const T_IN* qk_buf_src,
                                      const T_IN* qp_buf_src,
                                      const T* attr_mask,
                                      const int batch_size,
                                      const int head_num,
                                      const int seq_len,
                                      const T scalar)
{
    //printf("normal softmax");
    for (int seq_id = blockIdx.x; seq_id < seq_len; seq_id += gridDim.x) {
        float data[ITEMS_PER_THREAD];
        //printf("ITEMS_PER_THREAD: %d.", ITEMS_PER_THREAD);
        int qk_offset;
        __shared__ float s_mean, s_max;
        //float local_max = -1* pow(10,38);  this is very slow
        float local_max = -1e20f;
        for (int i = 0; blockDim.x * i + threadIdx.x < seq_len; i++) {
            qk_offset =
                ((blockIdx.y * head_num + blockIdx.z) * seq_len + seq_id) * seq_len + blockDim.x * i + threadIdx.x;
            //int mask_offset = (blockIdx.y * seq_len + seq_id) * seq_len + blockDim.x * i + threadIdx.x;
            int mask_offset = blockIdx.y * seq_len + (seq_id * seq_len + blockDim.x * i + threadIdx.x) % seq_len;

            //printf("qk_offset: %d, mask_offset: %d.", qk_offset, mask_offset);

            float qk = static_cast<float>(qk_buf_src[qk_offset]);
            qk += static_cast<float>(qp_buf_src[qk_offset]);

            float mask_val = static_cast<float>(ldg(&attr_mask[mask_offset]));
            // if (mask_val != 1 && qk_offset < 240*15)
            // {
            //     printf("mask_val: %f, mask_offset: %d.", mask_val, mask_offset);
            // }

            mask_val = (1.0f - mask_val) * -10000.0f;
            //mask_val = 0;

            data[i] = qk * static_cast<float>(scalar) + mask_val;
            local_max = fmax(local_max, data[i]);
        }

        float max_val = blockDim.x <= 32 ? warpReduceMax(local_max) : blockReduceMax<float>(local_max);
        if (threadIdx.x == 0) {
            s_max = max_val;
        }
        __syncthreads();

        float local_sum = 0;
        for (int i = 0; blockDim.x * i + threadIdx.x < seq_len; i++) {
            data[i] = __expf(data[i] - s_max);
            local_sum += data[i];
        }
        float sum_val = blockDim.x <= 32 ? warpReduceSum(local_sum) : blockReduceSum<float>(local_sum);
        if (threadIdx.x == 0) {
            s_mean = sum_val + 1e-6f;
            s_mean = __fdividef(1.0f, s_mean);
        }
        __syncthreads();

        for (int i = 0; blockDim.x * i + threadIdx.x < seq_len; i++) {
            qk_offset =
                ((blockIdx.y * head_num + blockIdx.z) * seq_len + seq_id) * seq_len + blockDim.x * i + threadIdx.x;
            int mask_offset = blockIdx.y * seq_len + (seq_id * seq_len + blockDim.x * i + threadIdx.x) % seq_len;
            float mask_val = static_cast<float>(ldg(&attr_mask[mask_offset]));
            if (mask_val < 1e-10f) {
                qk_buf_[qk_offset] = (T)(0);
            } else {
                qk_buf_[qk_offset] = (T)(data[i] * s_mean);
            }
            
        }
    }
}


template<typename T, int ITEMS_PER_THREAD>
__global__ void add_softmax_kernel_v4_half2(T* qk_buf_,
                                            const T* qp_buf_,
                                            const T* attr_mask,
                                            const int batch_size,
                                            const int head_num,
                                            const int seq_len,
                                            const T scalar)
{

    //printf("use v4 half2");
    using T2 = typename TypeConverter<T>::Type;
    T2* qk_buf_half2 = (T2*)qk_buf_;
    T2* qp_buf_half2 = (T2*)qp_buf_;
    const T2* attr_mask_half2 = (const T2*)attr_mask;

    for (int seq_id = blockIdx.x; seq_id < seq_len; seq_id += gridDim.x) {
        T2 data[ITEMS_PER_THREAD];
        int qk_offset;
        __shared__ float s_mean, s_max;
        float local_max = -1e20f;
        for (int i = 0; blockDim.x * i + threadIdx.x < (seq_len / 2) && i < ITEMS_PER_THREAD; i++) {
            qk_offset = ((blockIdx.y * head_num + blockIdx.z) * seq_len + seq_id) * (seq_len / 2) + blockDim.x * i
                        + threadIdx.x;
            //int mask_offset = (blockIdx.y * seq_len + seq_id) * (seq_len / 2) + blockDim.x * i + threadIdx.x;
            int mask_offset = blockIdx.y * (seq_len / 2) + (blockDim.x * i + threadIdx.x) % (seq_len / 2);

            T2 qk = qk_buf_half2[qk_offset];
            qk = hadd2<T2>(qk, qp_buf_half2[qk_offset]);
            T2 mask_val = ldg(&attr_mask_half2[mask_offset]);
            mask_val = hmul2<T2>(hsub2<T2>(float2type2<T2>(1.0f), mask_val), float2type2<T2>(-10000.0f));

            data[i] = hadd2<T2>(hmul2<T2>(qk, type2type2<T, T2>(scalar)), mask_val);

            local_max = fmax(local_max, fmax((float)data[i].x, (float)data[i].y));
        }

        float max_val = blockDim.x <= 32 ? warpReduceMax(local_max) : blockReduceMax<float>(local_max);
        if (threadIdx.x == 0) {
            s_max = max_val;
        }
        __syncthreads();

        float local_sum = 0;
        for (int i = 0; blockDim.x * i + threadIdx.x < (seq_len / 2) && i < ITEMS_PER_THREAD; i++) {
            data[i] = hexp2<T2>(hsub2<T2>(data[i], float2type2<T2>(s_max)));
            local_sum += (float)(data[i].x + data[i].y);
        }

        float sum_val = blockDim.x <= 32 ? warpReduceSum(local_sum) : blockReduceSum<float>(local_sum);

        if (threadIdx.x == 0) {
            s_mean = sum_val + 1e-6f;
            s_mean = __fdividef(1.0f, s_mean);
        }
        __syncthreads();

        for (int i = 0; blockDim.x * i + threadIdx.x < (seq_len / 2) && i < ITEMS_PER_THREAD; i++) {
            qk_offset = ((blockIdx.y * head_num + blockIdx.z) * seq_len + seq_id) * (seq_len / 2) + blockDim.x * i
                        + threadIdx.x;
            qk_buf_half2[qk_offset] = hmul2<T2>(data[i], float2type2<T2>(s_mean));
        }
        // this logic is must have when seq-length == 0
        // for (int i = 0; blockDim.x * i + threadIdx.x < seq_len && i < ITEMS_PER_THREAD; i++) {
        //     qk_offset = ((blockIdx.y * head_num + blockIdx.z) * seq_len + seq_id) * seq_len + blockDim.x * i
        //                 + threadIdx.x;
        //     int mask_offset = blockIdx.y * seq_len + (blockDim.x * i + threadIdx.x) % seq_len;
        //     if (attr_mask[mask_offset] < (T)1e-4)
        //     {
        //         qk_buf_[qk_offset] = (T)0;
        //     }
        // }
    }
}


template<typename T, int ITEMS_PER_THREAD, int NUM>
__global__ void add_softmax_kernel_v5_half2(T* qk_buf_,
                                            const T* qp_buf_,
                                            const T* attr_mask,
                                            const int batch_size,
                                            const int head_num,
                                            const int seq_len,
                                            const T scalar)
{
    //printf("use v5 half2");
    using T2 = typename TypeConverter<T>::Type;
    T2* qk_buf_half2 = (T2*)qk_buf_;
    T2* qp_buf_half2 = (T2*)qp_buf_;

    const T2* attr_mask_half2 = (const T2*)attr_mask;

    for (int seq_id = blockIdx.x; seq_id < seq_len; seq_id += gridDim.x * NUM) {
        T2 data[NUM][ITEMS_PER_THREAD];

        int qk_offset[NUM];

        __shared__ float s_sum[NUM], s_max[NUM];
        float local_max[NUM];
#pragma unroll
        for (int j = 0; j < NUM; j++) {
            local_max[j] = -1e20f;
        }

        for (int i = 0; blockDim.x * i + threadIdx.x < (seq_len / 2) && i < ITEMS_PER_THREAD; i++) {
            int mask_offset[NUM];
#pragma unroll
            for (int j = 0; j < NUM; j++) {
                qk_offset[j] = ((blockIdx.y * head_num + blockIdx.z) * seq_len + seq_id + j * gridDim.x) * (seq_len / 2)
                               + blockDim.x * i + threadIdx.x;

                // mask_offset[j] =
                //     (blockIdx.y * seq_len + seq_id + j * gridDim.x) * (seq_len / 2) + blockDim.x * i + threadIdx.x;
                mask_offset[j] = blockIdx.y * (seq_len / 2) + ( blockDim.x * i + threadIdx.x ) % (seq_len / 2);
            }

            T2 mask_val[NUM];
#pragma unroll
            for (int j = 0; j < NUM; j++) {
                mask_val[j] = ldg(&attr_mask_half2[mask_offset[j]]);
            }

            T2 qk[NUM];
#pragma unroll
            for (int j = 0; j < NUM; j++) {
                qk[j] = qk_buf_half2[qk_offset[j]];
                qk[j] = hadd2<T2>(qk[j], qp_buf_half2[qk_offset[j]]);
            }

#pragma unroll
            for (int j = 0; j < NUM; j++) {
                mask_val[j] = hmul2<T2>(hsub2<T2>(float2type2<T2>(1.0f), mask_val[j]), float2type2<T2>(-10000.0f));
            }

#pragma unroll
            for (int j = 0; j < NUM; j++) {
                data[j][i] = hadd2<T2>(hmul2<T2>(qk[j], type2type2<T, T2>(scalar)), mask_val[j]);
                local_max[j] = fmax(local_max[j], fmax((float)data[j][i].x, (float)data[j][i].y));
            }
        }

        if (blockDim.x <= 32) {
            warpReduceMaxV2<float, NUM>(local_max);
        }
        else {
            blockReduceMaxV2<float, NUM>(local_max);
        }

        if (threadIdx.x == 0) {
#pragma unroll
            for (int j = 0; j < NUM; j++) {
                s_max[j] = local_max[j];
            }
        }
        __syncthreads();

        float local_sum[NUM];
#pragma unroll
        for (int j = 0; j < NUM; j++) {
            local_sum[j] = {0.f};
        }

        for (int i = 0; blockDim.x * i + threadIdx.x < (seq_len / 2) && i < ITEMS_PER_THREAD; i++) {
#pragma unroll
            for (int j = 0; j < NUM; j++) {
                data[j][i] = hexp2<T2>(hsub2<T2>(data[j][i], float2type2<T2>(s_max[j])));
            }

#pragma unroll
            for (int j = 0; j < NUM; j++) {
                local_sum[j] += (float)(data[j][i].x + data[j][i].y);
            }
        }

        if (blockDim.x <= 32) {
            warpReduceSumV2<float, NUM>(local_sum);
        }
        else {
            blockReduceSumV2<float, NUM>(local_sum);
        }

        if (threadIdx.x == 0) {
#pragma unroll
            for (int j = 0; j < NUM; j++) {
                s_sum[j] = __fdividef(1.0f, local_sum[j] + 1e-6f);
            }
        }
        __syncthreads();

        for (int i = 0; blockDim.x * i + threadIdx.x < (seq_len / 2) && i < ITEMS_PER_THREAD; i++) {
#pragma unroll
            for (int j = 0; j < NUM; j++) {
                qk_offset[j] = ((blockIdx.y * head_num + blockIdx.z) * seq_len + seq_id + j * gridDim.x) * (seq_len / 2)
                               + blockDim.x * i + threadIdx.x;
            }

#pragma unroll
            for (int j = 0; j < NUM; j++) {
                qk_buf_half2[qk_offset[j]] = hmul2<T2>(data[j][i], float2type2<T2>(s_sum[j]));
            }
        }
    }
}




#define SOFTMAX_KERNEL(ITEMS_PER_THREAD)                                                                               \
    block.x /= ITEMS_PER_THREAD;                                                                                       \
    assert(block.x <= 1024);                                                                                           \
    if (is_half2) {                                                                                                    \
        if (grid.x % 4 == 0) {                                                                                         \
            grid.x /= 4;                                                                                               \
            add_softmax_kernel_v5_half2<half, ITEMS_PER_THREAD, 4><<<grid, block, 0, stream>>>((half*)buffer,          \
                                                                                               (const half*)qp_buf,    \
                                                                                               (const half*)attr_mask, \
                                                                                               batch_size,             \
                                                                                               head_num,               \
                                                                                               seq_len,                \
                                                                                               (const half)scalar);    \
        }                                                                                                              \
        else {                                                                                                         \
            add_softmax_kernel_v4_half2<half, ITEMS_PER_THREAD><<<grid, block, 0, stream>>>((half*)buffer,             \
                                                                                            (const half*)qp_buf,       \
                                                                                            (const half*)attr_mask,    \
                                                                                            batch_size,                \
                                                                                            head_num,                  \
                                                                                            seq_len,                   \
                                                                                            (const half)scalar);       \
        }                                                                                                              \
    }                                                                                                                  \
    else {                                                                                                             \
        add_softmax_kernel_v4<ITEMS_PER_THREAD, T, T_IN><<<grid, block, 0, stream>>>(                                  \
            buffer, buffer_src, qp_buf, attr_mask, batch_size, head_num, seq_len, scalar);                             \
    }


template<typename T, typename T_IN>
void invokeAddMaskedSoftMax(T* buffer,
                            const T_IN* buffer_src,
                            const T_IN* qp_buf,
                            const T* attr_mask,
                            const int batch_size,
                            const int seq_len,
                            const int head_num,
                            const T scalar,
                            cudaStream_t stream)
{

    dim3 grid(seq_len, batch_size, head_num);
    //printf("seq_len: %d, batch_size: %d, head_num: %d. ", seq_len, batch_size, head_num);
    if (batch_size * head_num > 360) {
        grid.x = ceil(float(seq_len) / 32.0f);
    }

    bool is_half2 = sizeof(T) == 2 && sizeof(T_IN) == 2 && seq_len % 2 == 0;
    //printf("size of T: %d, size of TN: %d, seq_len: %d", sizeof(T), sizeof(T_IN), seq_len  );
    dim3 block((seq_len / (is_half2 ? 2 : 1) + 31) / 32 * 32);
    //printf("block x: %d ", block.x);
    if (block.x > 3072 && block.x <= 4096) {
        SOFTMAX_KERNEL(4)
    }
    if (block.x > 2048) {
        SOFTMAX_KERNEL(3)
    }
    else if (block.x > 1024) {
        SOFTMAX_KERNEL(2)
    }
    else if (block.x > 0) {
        SOFTMAX_KERNEL(1)
    }
    else {
        FT_CHECK(seq_len <= 4096);
    }
}

template<typename T>
__global__ void addQKVPBiasTranspose(T* q_out,
                                     T* k_out,
                                     T* v_out,
                                     const T* __restrict q_in,
                                     const T* __restrict bias_q,
                                     const T* __restrict k_in,
                                     const T* __restrict bias_k,
                                     const T* __restrict v_in,
                                     const T* __restrict bias_v,
                                     T* p_buf,
                                     const T* P,
                                     T* q_buf_bias_v,
                                     const T* pos_bias_u,
                                     const T* pos_bias_v,
                                     const int batch_size,
                                     const int seq_len,
                                     const int head_num,
                                     const int size_per_head)
{
    const int n = head_num * size_per_head;
    const int batch_id = blockIdx.x;
    const int word_id = blockIdx.y;
    const int row_id = batch_id * seq_len + word_id;
    for (int col_id = threadIdx.x; col_id < n; col_id += blockDim.x) {
        const int head_id = col_id / size_per_head;
        const int size_id = col_id % size_per_head;
        const int target_id = batch_id * (head_num * seq_len * size_per_head) + head_id * seq_len * size_per_head
                              + word_id * size_per_head + size_id;
        const int src_id = row_id * n + col_id;
        /*
                q_out[target_id] = __ldg(&q_in[src_id]);
                q_out[target_id] = q_out[target_id] + __ldg(&bias_q[col_id]);

                k_out[target_id] = __ldg(&k_in[src_id]);
                k_out[target_id] = k_out[target_id] + __ldg(&bias_k[col_id]);

                v_out[target_id] = __ldg(&v_in[src_id]);
                v_out[target_id] = v_out[target_id] + __ldg(&bias_v[col_id]);
        */
        T q_val = __ldg(&q_in[src_id]) + __ldg(&bias_q[col_id]);
        q_out[target_id] = q_val + __ldg(&pos_bias_u[col_id]);

        k_out[target_id] = __ldg(&k_in[src_id]) + __ldg(&bias_k[col_id]);

        v_out[target_id] = __ldg(&v_in[src_id]) + __ldg(&bias_v[col_id]);

        p_buf[target_id] = __ldg(&P[src_id]);

        q_buf_bias_v[target_id] = q_val + __ldg(&pos_bias_v[col_id]);
    }
}

template<typename T>
__global__ void QKVPTranspose(T* q_out,
                              T* k_out,
                              T* v_out,
                              const T* __restrict q_in,
                              const T* __restrict k_in,
                              const T* __restrict v_in,
                              T* p_buf,
                              const T* P,
                              T* q_buf_bias_v,
                              const T* pos_bias_u,
                              const T* pos_bias_v,
                              const int batch_size,
                              const int seq_len,
                              const int head_num,
                              const int size_per_head)
{
    const int n = head_num * size_per_head;
    const int batch_id = blockIdx.x;
    const int word_id = blockIdx.y;
    const int row_id = batch_id * seq_len + word_id;
    for (int col_id = threadIdx.x; col_id < n; col_id += blockDim.x) {
        const int head_id = col_id / size_per_head;
        const int size_id = col_id % size_per_head;
        const int target_id = batch_id * (head_num * seq_len * size_per_head) + head_id * seq_len * size_per_head
                              + word_id * size_per_head + size_id;
        const int src_id = row_id * n + col_id;

        T q_val = __ldg(&q_in[src_id]);
        q_out[target_id] = q_val + __ldg(&pos_bias_u[col_id]);

        k_out[target_id] = __ldg(&k_in[src_id]);
        v_out[target_id] = __ldg(&v_in[src_id]);

        p_buf[target_id] = __ldg(&P[src_id]);
        q_buf_bias_v[target_id] = q_val + __ldg(&pos_bias_v[col_id]);
    }
}


template<typename T>
void invokeAddQKVPBiasTranspose(T* q_buf,
                                T* k_buf,
                                T* v_buf,
                                T* Q,
                                const T* bias_Q,
                                T* K,
                                const T* bias_K,
                                T* V,
                                const T* bias_V,
                                T* p_buf,
                                T* P,
                                T* q_buf_bias_v,
                                const T* pos_bias_u,
                                const T* pos_bias_v,
                                const int batch_size,
                                const int seq_len,
                                const int head_num,
                                const int size_per_head,
                                cudaStream_t stream)
{
    const int k = head_num * size_per_head;
    dim3 grid(batch_size, seq_len);
    bool is_add_bias = bias_Q != nullptr;
    if (sizeof(T) == 4 || k % 2 != 0) {
        dim3 block(min(k, 512));
        if (is_add_bias) {
            addQKVPBiasTranspose<T><<<grid, block, 0, stream>>>(q_buf,
                                                                k_buf,
                                                                v_buf,
                                                                Q,
                                                                bias_Q,
                                                                K,
                                                                bias_K,
                                                                V,
                                                                bias_V,
                                                                p_buf,
                                                                P,
                                                                q_buf_bias_v,
                                                                pos_bias_u,
                                                                pos_bias_v,
                                                                batch_size,
                                                                seq_len,
                                                                head_num,
                                                                size_per_head);
        }
        else {
            QKVPTranspose<T><<<grid, block, 0, stream>>>(q_buf,
                                                         k_buf,
                                                         v_buf,
                                                         Q,
                                                         K,
                                                         V,
                                                         p_buf,
                                                         P,
                                                         q_buf_bias_v,
                                                         pos_bias_u,
                                                         pos_bias_v,
                                                         batch_size,
                                                         seq_len,
                                                         head_num,
                                                         size_per_head);
        }
    }
    else {
        dim3 block(min(k / 2, 512));
        if (is_add_bias) {
            addQKVPBiasTranspose<half2><<<grid, block, 0, stream>>>((half2*)q_buf,
                                                                    (half2*)k_buf,
                                                                    (half2*)v_buf,
                                                                    (const half2*)Q,
                                                                    (const half2*)bias_Q,
                                                                    (const half2*)K,
                                                                    (const half2*)bias_K,
                                                                    (const half2*)V,
                                                                    (const half2*)bias_V,
                                                                    (half2*)p_buf,
                                                                    (const half2*)P,
                                                                    (half2*)q_buf_bias_v,
                                                                    (const half2*)pos_bias_u,
                                                                    (const half2*)pos_bias_v,
                                                                    batch_size,
                                                                    seq_len,
                                                                    head_num,
                                                                    size_per_head / 2);
        }
        else {
            QKVPTranspose<half2><<<grid, block, 0, stream>>>((half2*)q_buf,
                                                             (half2*)k_buf,
                                                             (half2*)v_buf,
                                                             (const half2*)Q,
                                                             (const half2*)K,
                                                             (const half2*)V,
                                                             (half2*)p_buf,
                                                             (const half2*)P,
                                                             (half2*)q_buf_bias_v,
                                                             (const half2*)pos_bias_u,
                                                             (const half2*)pos_bias_v,
                                                             batch_size,
                                                             seq_len,
                                                             head_num,
                                                             size_per_head / 2);
        }
    }
}

template void invokeAddQKVPBiasTranspose<float>(float* q_buf,
                                         float* k_buf,
                                         float* v_buf,
                                         float* Q,
                                         const float* bias_Q,
                                         float* K,
                                         const float* bias_K,
                                         float* V,
                                         const float* bias_V,
                                         float* p_buf,
                                         float* P,
                                         float* q_buf_bias_v,
                                         const float* pos_bias_u,
                                         const float* pos_bias_v,
                                         const int batch_size,
                                         const int seq_len,
                                         const int head_num,
                                         const int size_per_head,
                                         cudaStream_t stream);

template void invokeAddQKVPBiasTranspose<half>(half* q_buf,
                                         half* k_buf,
                                         half* v_buf,
                                         half* Q,
                                         const half* bias_Q,
                                         half* K,
                                         const half* bias_K,
                                         half* V,
                                         const half* bias_V,
                                         half* p_buf,
                                         half* P,
                                         half* q_buf_bias_v,
                                         const half* pos_bias_u,
                                         const half* pos_bias_v,
                                         const int batch_size,
                                         const int seq_len,
                                         const int head_num,
                                         const int size_per_head,
                                         cudaStream_t stream);


template void invokeAddMaskedSoftMax<float, float>(float* buffer,
                                     const float* buffer_src,
                                     const float* qp_buf,
                                     const float* attr_mask,
                                     const int batch_size,
                                     const int seq_len,
                                     const int head_num,
                                     const float scalar,
                                     cudaStream_t stream);

template void invokeAddMaskedSoftMax<half, float>(half* buffer,
                                     const float* buffer_src,
                                     const float* qp_buf,
                                     const half* attr_mask,
                                     const int batch_size,
                                     const int seq_len,
                                     const int head_num,
                                     const half scalar,
                                     cudaStream_t stream);

template void invokeAddMaskedSoftMax<half, half>(half* buffer,
                                     const half* buffer_src,
                                     const half* qp_buf,
                                     const half* attr_mask,
                                     const int batch_size,
                                     const int seq_len,
                                     const int head_num,
                                     const half scalar,
                                     cudaStream_t stream);


