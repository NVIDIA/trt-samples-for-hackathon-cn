#pragma once
#include <getopt.h>
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <math.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <assert.h>
//#include"cnpy.h"
#include <nvml.h>
#include <string>
#include <iomanip>
#include "string.h"
#include "stdio.h"
#include <limits.h> 
#include <random>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <stdio.h>





/*************MACRO FUNCTION**************/
#define FINAL_MASK 0xffffffff
#define MAX_HIDDEN_DIM 1024
#define NUM_CUBLAS_FUNC 10
#define FULL_GEMM_LENGTH 23
#define VERIFY_NUM 7
#define WARP_SIZE 32
#define NUM_ITER 1
#define ERR_LIMIT 0.1
#define ERR_LIMIT_FP16 10
//#define SIZE_DICTION 32000
//#define DATA_TYPE __half
//#define DATA_TYPE float

enum cublasFunction{
    CUBLAS_STRIDE=0,
    CUBLAS_A_0=1,
    CUBLAS_B_0=2,
    CUBLAS_SINGLE=4,
    CUBLASLT_GEMM=5,
    CUBLASLT_BIAS=6,
};

enum RUN_MODE{
    FP16_PERF_TEST=0,
    FP16_CORRECTNESS_TEST=1,
    FP32_PERF_TEST=2,
    FP32_CORRECTNESS_TEST=3,
};

const char * const matmulTileName[] = {
    "UNDEF",
    "8x8",
    "8x16",
    "16x8"   ,
    "8x32"   ,
    "16x16"  ,
    "32x8"   ,
    "8x64"   ,
    "16x32"  ,
    "32x16"  ,
    "64x8"   ,
    "32x32"  ,
    "32x64"  ,
    "64x32"  ,
    "32x128" ,
    "64x64"  ,
    "128x32" ,
    "64x128" ,
    "128x64" ,
    "64x256" ,
    "128x128",
    "256x64" ,
    "64x512" ,
    "128x256",
    "256x128",
    "512x64" ,
};

/*************Error Handling**************/
bool check(cudaError_t e, int iLine, const char *szFile);
bool check(cublasStatus_t e, int iLine, const char *szFile);
#define ck(call) check(call, __LINE__, __FILE__)
#define PRINT_FUNC_NAME_() do{\
    std::cout << "[FT][CALL] " << __FUNCTION__ << " " << std::endl; \
} while (0)

/*************Time Handling**************/
class CudaTimer{
    private:
        cudaEvent_t event_start;
        cudaEvent_t event_stop;
        cudaStream_t stream;
        float time;
    public:
        CudaTimer(cudaStream_t stream=0);
        void start();
        float stop();
        ~CudaTimer();
};
/*************Useful functions***********************/
int blockNum(int size, int blockSize);
int next_pow2(int a);
template <typename T> int numPerThread();
template <typename T> void deviceMalloc(T** ptr, int size);
template <typename T> void deviceMemset(T* ptr, int value, int size);
template <typename T> cudaError_t  deviceFree(T* & ptr);
template <typename T> void deviceMemcpyHtoD(cudaStream_t stream, T* d_ptr,T* h_ptr, int size);
template <typename T> float castToFloat(T input);
template <typename T> void setRandom(T* data, int len);
/*********************Npz &Npy File Process functions***********************/

 
