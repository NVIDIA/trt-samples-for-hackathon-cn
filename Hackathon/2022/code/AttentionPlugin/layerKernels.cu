#include <cmath>
#include "layerKernels.h"

namespace cg = cooperative_groups;

/*************Device Function**************/
/*
   dim3 grid_trans_v(4*batch_size, 1, 1);
   dim3 block_trans_v(1024);


   const int toff0=head_num*time1*size_per_head;
   const int ti_off1=size_per_head*head_num;
   const int to_off1=time1*size_per_head;
   const int toff2=size_per_head;;

   transpose102_addBias<<<grid_trans_v, block_trans_v,0,stream>>>(qkvp_transpose_buf, q_with_bias_u, q_with_bias_v,
   qkvp_buf, layer_weight_device.qkv_bias_buf,layer_weight_device.pos_bias_u,layer_weight_device.pos_bias_v,
   batch_size,time1, toff0, ti_off1, to_off1, toff2);
 */

template<typename T>
__global__ void transpose102_addBias(T* dst, T* q_u, T* q_v, const T* src, const T* bias, const T* bias_u, const T* bias_v,
    const int batch_size,const int time1, const int off0,const int i_off1, const int o_off1, const int off2)
{
    int time1_per_block=1024/off2;
    int x[4]={0};
    x[0]=blockIdx.x;//batch_size*3+1
    x[2]=blockIdx.y;//num_head
    x[1]=blockIdx.z*time1_per_block+threadIdx.x/off2;//time1
    x[3]=threadIdx.x%off2;//size_per_head
    int i_qkvp=x[0]/batch_size;

    T tmp;
    int input_index=0;
    int out_index=0;
    int offset=x[2]*off2+x[3];

    if(x[1]<o_off1/off2){
        if(i_qkvp==0){
            input_index=x[0]*off0+x[1]*i_off1+offset;
            out_index=x[0]*off0+x[2]*o_off1+x[1]*off2+x[3];

            bias+=i_qkvp*i_off1;
            for(; x[1]<time1;x[1]=x[1]+32){
                tmp=src[input_index]+bias[offset];
                q_u[out_index]=tmp+bias_u[offset];
                q_v[out_index]=tmp+bias_v[offset];

                input_index+=i_off1<<5;
                out_index+=off2<<5;
            }
        }else{
            if(i_qkvp==1||i_qkvp==2){
                input_index=x[0]*off0+x[1]*i_off1+offset;
                out_index=x[0]*off0+x[2]*o_off1+x[1]*off2+x[3];
                bias+=i_qkvp*i_off1;
                for(; x[1]<time1;x[1]=x[1]+32){
                    dst[out_index]=src[input_index]+bias[offset];
                    input_index+=i_off1<<5;
                    out_index+=off2<<5;
                }
            }else if(i_qkvp==3){
                input_index=3*batch_size*off0+x[1]*i_off1+offset;
                out_index=x[0]*off0+x[2]*o_off1+x[1]*off2+x[3];

                for(; x[1]<time1;x[1]=x[1]+32){
                    dst[out_index]=src[input_index];

                    input_index+=i_off1<<5;
                    out_index+=off2<<5;
                }
            }//end if_else (i_qkvp==1||i_qkvp==2)
        }////end else (i_qkvp==0)
    }//end time1
}



template<typename T>
__global__ void add_ac_bd_mask(T* dst, const T* ac, const T* bd, const int * mask,
        const T sqrt_d_k,const int off, const int offset, const int time2)
{
    int i_batch=blockIdx.x;
    int batch_off=i_batch*off;
    ac+=batch_off;
    bd+=batch_off;
    dst+=batch_off;

    int index = blockIdx.y*blockDim.x+ threadIdx.x;
    int mask_index = i_batch*time2;
    bool ifMask=true;
    T v_score;
    for(; index<off; index=index+offset){
        ifMask= (mask[mask_index+ index%time2]==0)? true: false;
        if(ifMask)
        {
            if (sizeof(T) == 2){
                v_score = T(-60000.0f);
            }
            else
            {
                v_score = T(-1* pow(10,38));
            }
            
	    /*
            if(sizeof(T)==2){
                v_score  =__float2half(-float(INFINITY));
            }else{
                v_score = -float(INFINITY);
            }
	    */
        }
        else
        {
            v_score= (ac[index]+bd[index])/sqrt_d_k;
        }
        dst[index] = v_score;
    }
}

// attn = torch.softmax(scores, dim=-1)                 [batch, head, time1, time2]
// [0,0,0,1]+[0,0,0,2]+[0,0,0,3]+...[0,0,0,47]
// attn=attn.masked_fill(mask, 0.0)  # (batch, head, time1, time2)
//grid(batch*head, (time1-1)/16+1)
//block(16,32)
template<typename T,int NUM_PER_THREAD, int WARP_PER_BLOCK>
__global__ void softmax(T* dst, T* src, const int offset, const int time2)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    // Handle to tile in thread block
    cg::thread_block_tile<32> tile = cg::tiled_partition<32>(cta);

    int index = blockIdx.x*offset;
    int i_time = blockIdx.y*WARP_PER_BLOCK+(threadIdx.x/32);
    index +=i_time*time2;

    T elements_per_thread[NUM_PER_THREAD]={0};
    T sum=0;
    T mask_value = 0;
    T max=0;
    if (sizeof(T) == 2){
        max = T(-60000.0f);
        mask_value = T(-60000.0f);
    }
    else
    {
        max = T(-1* pow(10,38));
        mask_value = T(-1* pow(10,38));
    }

    int i=threadIdx.x%WARP_SIZE;
    int j=0;
    for(; i < time2; i= i + WARP_SIZE)
    {
        elements_per_thread[j] = src[index+i];
        if(elements_per_thread[j]>max)
        {
            max=elements_per_thread[j];
        }
        j++;
    }
    max = cg::reduce(tile, max, cg::greater<T>());

    i=threadIdx.x%WARP_SIZE;
    j=0;
    for(; i < time2; i= i + WARP_SIZE)
    {
        elements_per_thread[j] = __expf(elements_per_thread[j]-max);
        sum+=elements_per_thread[j];
        j++;
    }
    sum = cg::reduce(tile, sum, cg::plus<T>());

    i=threadIdx.x%WARP_SIZE;
    j=0;
    for(; i < time2; i= i + WARP_SIZE)
    {
        if(src[index + i] == mask_value)
        {
            dst[index+i]=0;
        }
        else{
            dst[index+i]=elements_per_thread[j]/sum;
        }
        j++;
    }
}

// x= x.transpose(1, 2)
//[batch, head, time1, d_k] -> [batch, time1, head,d_k]  012-> 102
template<typename T>
__global__ void transpose102_res(T* dst, T* src, T* res, T* bias,
        const int off0,const int i_off1, const int o_off1, const int off2, const int time1)
{
    int time1_offset= blockDim.y;
    int ibatch= blockIdx.x;
    int ihead = blockIdx.y;
    int itime = blockIdx.z*time1_offset + threadIdx.y;
    int id_k = threadIdx.x;
    int input_index=ibatch*off0+id_k;
    int out_index=input_index;
    input_index+=ihead*i_off1+itime*off2;
    out_index+=itime*o_off1+ihead*off2;

    time1_offset=time1_offset*gridDim.z;
    T b=bias[ihead*off2+id_k];
    for(; itime<time1;itime=itime + time1_offset){
        dst[out_index] = src[input_index];
        res[out_index] = b;
        input_index+=off2*time1_offset;
        out_index+=o_off1*time1_offset;
    }
}

#define SOFT_MAX(NUM_PER_THREAD, WARP_PER_BLOCK)    \
    softmax<T,NUM_PER_THREAD, WARP_PER_BLOCK>       \
    <<<grid_softmax, block_softmax, 0, stream>>>    \
    (dst, src, offset, time2);

template<typename T>
void invokeSoftmax(T* dst, T* src, const int batch_size, const int head_num,
     const int time1,const int time2, cudaStream_t stream)
{
    const int offset =time1*time2;

    if(time2 <= WARP_SIZE)
    {
        int WARP_PER_BLOCK=32;
        dim3 grid_softmax(batch_size*head_num, (time1-1)/WARP_PER_BLOCK+1);
        dim3 block_softmax(WARP_PER_BLOCK*WARP_SIZE);
        SOFT_MAX(1,32)
    }
    else if(time2 < WARP_SIZE*2)
    {
        int WARP_PER_BLOCK=32;
        dim3 grid_softmax(batch_size*head_num, (time1-1)/WARP_PER_BLOCK+1);
        dim3 block_softmax(WARP_PER_BLOCK*WARP_SIZE);
        SOFT_MAX(2,32)
    }
    else if(time2 < WARP_SIZE*3)
    {
        int WARP_PER_BLOCK=16;
        dim3 grid_softmax(batch_size*head_num, (time1-1)/WARP_PER_BLOCK+1);
        dim3 block_softmax(WARP_PER_BLOCK*WARP_SIZE);
        SOFT_MAX(3,16)
    }
    else if(time2 < WARP_SIZE*4)
    {
        int WARP_PER_BLOCK=16;
        dim3 grid_softmax(batch_size*head_num, (time1-1)/WARP_PER_BLOCK+1);
        dim3 block_softmax(WARP_PER_BLOCK*WARP_SIZE);
        SOFT_MAX(4,16)
    }
    else if(time2 < WARP_SIZE*5)
    {
        int WARP_PER_BLOCK=8;
        dim3 grid_softmax(batch_size*head_num, (time1-1)/WARP_PER_BLOCK+1);
        dim3 block_softmax(WARP_PER_BLOCK*WARP_SIZE);
        SOFT_MAX(5,8)
    }
    else if(time2 < WARP_SIZE*6)
    {
        int WARP_PER_BLOCK=8;
        dim3 grid_softmax(batch_size*head_num, (time1-1)/WARP_PER_BLOCK+1);
        dim3 block_softmax(WARP_PER_BLOCK*WARP_SIZE);
        SOFT_MAX(6,8)
    }
    else if(time2 < WARP_SIZE*7)
    {
        int WARP_PER_BLOCK=8;
        dim3 grid_softmax(batch_size*head_num, (time1-1)/WARP_PER_BLOCK+1);
        dim3 block_softmax(WARP_PER_BLOCK*WARP_SIZE);
        SOFT_MAX(7,8)
    }
    else if(time2 < WARP_SIZE*8)
    {
        int WARP_PER_BLOCK=4;
        dim3 grid_softmax(batch_size*head_num, (time1-1)/WARP_PER_BLOCK+1);
        dim3 block_softmax(WARP_PER_BLOCK*WARP_SIZE);
        SOFT_MAX(8,4)
    }
}

//The explicit instantiation part
template __global__ void transpose102_addBias<float>(float* dst, float* q_u, float* q_v,
        const float* src, const float* bias,const float* bias_u, const float* bias_v,
        const int batch_size, const int time1, const int off0, const int i_off1, const int o_off1, const int off2);

template __global__ void transpose102_addBias<__half>(__half* dst, __half* q_u, __half* q_v,
        const __half* src, const __half* bias,const __half* bias_u, const __half* bias_v,
        const int batch_size, const int time1, const int off0, const int i_off1, const int o_off1, const int off2);
template __global__  void add_ac_bd_mask<float>(float* dst, const float* ac, const float* bd,const int * mask,
        const float sqrt_d_k, const int off, const int offset, const int time2);

template __global__  void add_ac_bd_mask<__half>(__half* dst, const __half* ac, const __half* bd,const int * mask,
        const __half sqrt_d_k, const int off, const int offset, const int time2);
template void invokeSoftmax<float>(float* dst, float* src, const int batch_size, const int head_num,
     const int time1,const int time2, cudaStream_t stream);

template void invokeSoftmax<half>(half* dst, half* src, const int batch_size, const int head_num,
     const int time1,const int time2, cudaStream_t stream);
template __global__ void transpose102_res<float>(float* dst, float* src, float* res, float* bias,
        const int off0,const int i_off1, const int o_off1, const int off2, const int time1);

template __global__ void transpose102_res<__half>(__half* dst, __half* src, __half* res, __half* bias,
        const int off0,const int i_off1, const int o_off1, const int off2, const int time1);

