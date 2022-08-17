#pragma once

#include "utils.h"

template<typename T> 
__global__ void transpose102_addBias(T* dst, T* q_u, T* q_v, 
        const T* src, const T* bias, const T* bias_u, const T* bias_v,
        const int batch_size,const int time1, const int off0,const int i_off1, const int o_off1, const int off2);

template<typename T>
    __global__ void add_ac_bd_mask(T* dst, const T* ac, const T* bd,const int * mask,
    const T sqrt_d_k,const int off, const int offset, const int time2);

template<typename T>
__global__
void transpose102_res(T* dst, T* src, T* res, T* bias,
    const int off0,const int i_off1, const int o_off1, const int off2, const int time1);

template<typename T>
void invokeSoftmax(T* dst, T* src, const int batch_size, const int head_num,
    const int time1,const int time2, cudaStream_t stream);

