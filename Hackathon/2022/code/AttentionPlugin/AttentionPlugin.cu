#include "AttentionPlugin.h"

/*
input[ 0]:  [float32/float16],  (b,time1,hidden_dim)
input[ 1]:  [float32/float16],  (b,time2,hidden_dim)
input[ 2]:  [float32/float16],  (b,time2,hidden_dim)
input[ 3]:  [float32/float16],  (b,time2,hidden_dim)
input[ 4]:  [int32],            (b,1,time1)

input[ 5]:  [float32/float16],  (head_num,size_per_head),   pos_bias_u
input[ 6]:  [float32/float16],  (head_num,size_per_head),   pos_bias_v
input[ 7]:  [float32/float16],  (hidden_dim,hidden_dim),    linear_q_weight
input[ 8]:  [float32/float16],  (hidden_dim,),              linear_q_bias
input[ 9]:  [float32/float16],  (hidden_dim,hidden_dim),    linear_k_weight
input[10]:  [float32/float16],  (hidden_dim,),              linear_k_bias
input[11]:  [float32/float16],  (hidden_dim,hidden_dim),    linear_v_weight
input[12]:  [float32/float16],  (hidden_dim,),              linear_v_bias
input[13]:  [float32/float16],  (hidden_dim,hidden_dim),    linear_out_weight
input[14]:  [float32/float16],  (hidden_dim,),              linear_out_bias
input[15]:  [float32/float16],  (hidden_dim,hidden_dim),    linear_pos_weight

output[0]:  [float32/float16],  (b,time2,hidden_dim)
*/

using namespace nvinfer1;
using namespace plugin;

PluginFieldCollection AttentionPluginCreator::fc_{};
std::vector<PluginField> AttentionPluginCreator::attr_;

template<typename T>
__global__ void stack(const T* A, T* C, const size_t batch_size)
{
    for(size_t i = 0; i < batch_size; i++)
    {
        size_t idx_A = threadIdx.x * gridDim.x + blockIdx.x;
        size_t idx_C = i * blockDim.x * gridDim.x + threadIdx.x * gridDim.x + blockIdx.x;
        C[idx_C] = A[idx_A];
    }
}


template <typename T>
int32_t AttentionPlugin<T>::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    m_.batch_size  = inputDesc[0].dims.d[0];
    m_.time1       = inputDesc[0].dims.d[1];
    m_.time2       = inputDesc[0].dims.d[1];

#if DEBUG_ENABLE
    printf("[AttentionPlugin::enqueue]m_:\n");
    printf("\tbatch_size = %d\n",m_.batch_size);
    printf("\tbatch_pos = %d\n",m_.batch_pos);
    printf("\ttime1 = %d\n",m_.time1);
    printf("\ttime2 = %d\n",m_.time2);
    printf("\thead_num = %d\n",m_.head_num);
    printf("\tsize_per_head = %d\n",m_.size_per_head);
    printf("\thidden_dim = %d\n",m_.hidden_dim);
    printf("\tbuf_size = %d\n",m_.buf_size);
    printf("\tpbuf_size = %d\n",m_.pbuf_size);
#endif

    // cudaEvent_t start, stop;
    // ck( cudaEventCreate( &start ) );
    // ck( cudaEventCreate( &stop ) );
    // ck( cudaEventRecord( start, 0 ) );

    T*      x_q                 = (T*)  inputs[0];
    T*      pos_emb             = (T*)  inputs[1];
    T*    mask                = (T*)inputs[2];
    T*  pos_bias_u_dev_         = (T*)  inputs[3];
    T*  pos_bias_v_dev_         = (T*)  inputs[4];
    T*  qkv_weight_buf_dev_    = (T*)  inputs[5];
    T*  linear_q_bias_dev_      = (T*)  inputs[6];
    T*  linear_k_bias_dev_      = (T*)  inputs[7];
    T*  linear_v_bias_dev_      = (T*)  inputs[8];
    T*  linear_out_weight_dev_  = (T*)  inputs[9];
    T*  linear_out_bias_dev_    = (T*)  inputs[10];
    T*  linear_pos_weight_dev_  = (T*)  inputs[11];
    T*  res                     = (T*)  outputs[0];

    auto* space     = (T*)workspace;
    int nElement    = 0;
    auto* qkvp_buf  = space + nElement;
    nElement       += 0;
    auto* q         = space + nElement;
    nElement       += m_.buf_size;
    auto* k         = space + nElement;
    nElement       += m_.buf_size;
    auto* v         = space + nElement;
    nElement       += m_.buf_size;
    auto* p         = space + nElement;
    nElement       += m_.pbuf_size;
    auto* qkvp_transpose_buf    = space + nElement;
    nElement                   += 0;
    auto* q_transpose           = space + nElement;
    nElement                   += m_.buf_size;
    auto* k_transpose           = space + nElement;
    nElement                   += m_.buf_size;
    auto* v_transpose           = space + nElement;
    nElement                   += m_.buf_size;
    auto* p_transpose           = space + nElement;
    nElement                   += m_.buf_size;
    auto* q_with_bias_u         = space + nElement;
    nElement                   += m_.buf_size;
    auto* q_with_bias_v         = space + nElement;
    nElement                   += m_.buf_size;
    auto* matrix_ac             = space + nElement;
    nElement                   += m_.batch_size * m_.head_num * m_.time1 * m_.time2;
    auto* matrix_bd             = space + nElement;
    nElement                   += m_.batch_size * m_.head_num * m_.time1 * m_.time2;
    auto* score                 = space + nElement;
    nElement                   += m_.batch_size * m_.head_num * m_.time1 * m_.time2;
    
    auto* x                     = space + nElement;
    nElement                   += m_.batch_size * m_.head_num * m_.time1 * m_.size_per_head;
    auto* x_transpose           = space + nElement;
    nElement                   += m_.batch_size * m_.head_num * m_.time1 * m_.size_per_head;
    T** dx                      = reinterpret_cast<T**>(space + nElement);
    
    ck(cudaGetLastError());

    BIG_PRINT_WEIGHT()
    BIG_PRINT(0)

    

    int cublas_index = 0;
    //q, k, v = self.forward_qkv(query, key, value)
    //p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
    cudaDataType_t d_type=CUDA_R_32F;
    cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    if(sizeof(T) == 2)
    {
        d_type = CUDA_R_16F;
    }
    float a = 1.0;
    float b = 0.0;


    const T* hx[] {x_q, x_q, x_q, qkv_weight_buf_dev_, qkv_weight_buf_dev_ + m_.hidden_dim * m_.hidden_dim, 
    qkv_weight_buf_dev_ + m_.hidden_dim * m_.hidden_dim *2, q, k, v};

    cudaMemcpyAsync((void*)dx, hx, sizeof(T*) * 9, cudaMemcpyHostToDevice, stream);
    
    cublasStatus_t s0 =  cublasGemmBatchedEx(cublasHandle_,
                                  CUBLAS_OP_N,
                                  CUBLAS_OP_N,
                                  m_.hidden_dim, m_.batch_size * m_.time1, m_.hidden_dim,
                                  &a,
                                  (const void* const*)(dx+3),
                                  d_type, m_.hidden_dim, 
                                  (const void* const*)dx, 
                                  d_type,
                                  m_.hidden_dim, 
                                  &b,                                
                                  (void* const*)(dx+6),
                                  d_type,
                                   m_.hidden_dim, 
                                  3,
                                  CUBLAS_COMPUTE_32F,
                                  CUBLAS_GEMM_DEFAULT);
    
    BIG_PRINT(1)

    cublasStatus_t s4 =  cublasGemmStridedBatchedEx(cublasHandle_,
                                  CUBLAS_OP_N,
                                  CUBLAS_OP_N,
                                  m_.hidden_dim, 1 * m_.time1, m_.hidden_dim,
                                  &a,
                                  (T*)linear_pos_weight_dev_,
                                  d_type,
                                  m_.hidden_dim, 
                                  (int64_t)(m_.hidden_dim * m_.hidden_dim),
                                  (T*)pos_emb,
                                  d_type,
                                  m_.hidden_dim, (int64_t)(1 * m_.time1 * m_.hidden_dim),
                                  &b,
                                  (T*)p,
                                  d_type, 
                                  m_.hidden_dim, (int64_t)(1 * m_.time1 * m_.hidden_dim),
                                  1,
                                  compute_type,
                                  CUBLAS_GEMM_DEFAULT);
    
    BIG_PRINT(2)

    invokeAddQKVPBiasTranspose<T>(q_with_bias_u, k_transpose, v_transpose, q, linear_q_bias_dev_, k,
                                 linear_k_bias_dev_, v, linear_v_bias_dev_, p_transpose,
                                 p, q_with_bias_v, pos_bias_u_dev_, pos_bias_v_dev_,
                                 m_.batch_size, m_.time1, m_.head_num, m_.size_per_head,
                                 stream);


    cublasStatus_t s1 =  cublasGemmStridedBatchedEx(cublasHandle_,
                                  CUBLAS_OP_T,
                                  CUBLAS_OP_N,
                                  m_.time1, m_.time1, m_.size_per_head,
                                  &a,
                                  (T*)k_transpose,
                                  d_type, 
                                  m_.size_per_head, 
                                  (int64_t)(m_.size_per_head * m_.time1),
                                  (T*)q_with_bias_u,
                                  d_type,
                                  m_.size_per_head, 
                                  (int64_t)(m_.size_per_head * m_.time1),
                                  &b,
                                  (T*)matrix_ac,
                                  d_type, 
                                  m_.time1, (int64_t)(m_.time1 * m_.time1),
                                  m_.batch_size * m_.head_num,
                                  compute_type,
                                  CUBLAS_GEMM_DEFAULT);
    BIG_PRINT(3)

    T** hp  = new T*[m_.batch_size * 4 * 3];
    for (int i = 0; i < m_.batch_size; i++){
        for (int j = 0; j < 4; j++){
            hp[i*4 + j] = p_transpose + j * m_.time1 * m_.size_per_head;
            hp[m_.batch_size * 4 + i*4 + j] = q_with_bias_v + m_.time1 * m_.size_per_head * (i*4 + j);
            hp[m_.batch_size * 4 * 2 + i*4 + j] = matrix_bd + (i*4 + j) * m_.time1 * m_.time1;
        }
    }

    cudaMemcpyAsync((void*)dx, hp, sizeof(T*) * m_.batch_size * 4 * 3, cudaMemcpyHostToDevice, stream);

    cublasStatus_t s2 =  cublasGemmBatchedEx(cublasHandle_,
                                  CUBLAS_OP_T,
                                  CUBLAS_OP_N,
                                  m_.time1, m_.time1, m_.size_per_head,
                                  &a,
                                  (const void* const*)(dx),
                                  d_type, 
                                  m_.size_per_head, 
                                  (const void* const*)(dx + m_.batch_size * 4 ),
                                  d_type,
                                  m_.size_per_head, 
                                  &b,
                                  (void* const*)(dx + m_.batch_size * 4 *2),
                                  d_type, 
                                  m_.time1,
                                  m_.batch_size * 4,
                                  CUBLAS_COMPUTE_32F,
                                  CUBLAS_GEMM_DEFAULT);

    BIG_PRINT(4)

    // //++++++++forward+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    invokeAddMaskedSoftMax<T, T>(
        score, matrix_ac, matrix_bd, mask, m_.batch_size, m_.time1, m_.head_num, (T)0.125f, stream);


    //@@@@ p_attn = self.dropout(attn) //@TODO:check this is not used
    //@@@@ x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
    /// p_attn [batch, head, time1, time2]
    // value (v) (batch, head,time1, d_k)
    // -> [batch, head, time1, d_k]
    //@TODO: add the variable to store the result

    cublasStatus_t s3 =  cublasGemmStridedBatchedEx(cublasHandle_,
                                  CUBLAS_OP_N,
                                  CUBLAS_OP_N,
                                  m_.size_per_head, m_.time1, m_.time1,
                                  &a,
                                  (T*)v_transpose,
                                  d_type, 
                                  m_.size_per_head, 
                                  (int64_t)(m_.size_per_head * m_.time1),
                                  (T*)score,
                                  d_type,
                                  m_.time1, 
                                  (int64_t)(m_.time1 * m_.time1),
                                  &b,
                                  (T*)x,
                                  d_type, 
                                  m_.size_per_head, (int64_t)(m_.time1 * m_.size_per_head),
                                  m_.batch_size * m_.head_num,
                                  compute_type,
                                  CUBLAS_GEMM_DEFAULT);

    int time1_per_block= floor(1024/m_.size_per_head);
    int time_loop_count = 10;
    dim3 grid_trans(m_.batch_size, m_.head_num, (m_.time1 - 1) / (time1_per_block*time_loop_count) + 1);
    dim3 block_trans(m_.size_per_head, time1_per_block);

    transpose102_res<<<grid_trans, block_trans, 0, stream >>>(
        res, x, x_transpose,linear_out_bias_dev_,
            m_.head_num * m_.time1 * m_.size_per_head, m_.time1 * m_.size_per_head,
            m_.head_num * m_.size_per_head, m_.size_per_head, m_.time1);

    // ck( cudaEventRecord( stop, 0 ) );
    // ck( cudaEventSynchronize( stop ) );
    // float elapsedTime;
    // ck( cudaEventElapsedTime( &elapsedTime,
    // start, stop ) );
    // printf( "attention time: %f ms\n", elapsedTime );
    // ck( cudaEventDestroy( start ) );
    // ck( cudaEventDestroy( stop ) );

    delete [] hp;

    return 0;
}

REGISTER_TENSORRT_PLUGIN(AttentionPluginCreator);

