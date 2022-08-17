#include "utils.h"
#define LOAD_FROM_FLOAT  

using namespace std;
namespace std {
  template <typename _CharT, typename _Traits>
  inline basic_ostream<_CharT, _Traits> &
  tab(basic_ostream<_CharT, _Traits> &__os) {
    return __os.put(__os.widen('\t'));
  }
}

std::string stringPadding(std::string original, size_t charCount)
{
    original.resize(charCount, ' ');
    return original;
}

/*************Error Handling**************/
bool check(cudaError_t e, int iLine, const char *szFile) {
    if (e != cudaSuccess) {
        cout << "CUDA runtime API error " << cudaGetErrorName(e)<<" with e= "<< e << " at line " << iLine << " in file " << szFile << endl;
        exit(0);
        return false;
    }
    return true;
}
const char* cublasGetErrorString(cublasStatus_t status)
{
    switch(status)
    {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE"; 
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH"; 
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED"; 
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR"; 
        case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
    }
    return "unknown error";
}

bool check(cublasStatus_t e, int iLine, const char *szFile) {
    if (e !=CUBLAS_STATUS_SUCCESS) {
        cout << "CUDA CUBLAS runtime API error " << cublasGetErrorString(e)<<" with e= "<< e  << " at line " << iLine << " in file " << szFile << endl;
        exit(0);
        return false;
    }
    return true;
}

/*************Time Handling**************/
CudaTimer::CudaTimer(cudaStream_t stream){
    this->stream=stream;
}

void CudaTimer::start(){
#ifdef CHECK
    ck(cudaEventCreate(&event_start));
    ck(cudaEventCreate(&event_stop));
    ck(cudaEventRecord(event_start, stream)); 
#else
    cudaEventCreate(&event_start);
    cudaEventCreate(&event_stop);
    cudaEventRecord(event_start, stream); 
#endif
}
float CudaTimer::stop(){
#ifdef CHECK
    ck(cudaEventRecord(event_stop,stream));
    ck(cudaEventSynchronize(event_stop));
    ck(cudaEventElapsedTime(&time, event_start, event_stop));
    ck(cudaEventDestroy(event_start));
    ck(cudaEventDestroy(event_stop));
#else
    cudaEventRecord(event_stop,stream);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&time, event_start, event_stop);
    cudaEventDestroy(event_start);
    cudaEventDestroy(event_stop);
#endif
    return time;
}
CudaTimer:: ~CudaTimer(){
}


/*************Useful functions***********************/
int blockNum(int size, int blockSize){
    int nblock= (size-1)/blockSize+1;
    return nblock;
}
int next_pow2(int a){
    int rval=32;
    if(a>32){
        while(rval<a) rval<<=1;
    }
    return rval;
}


template<typename T>
int numPerThread(){
    return sizeof(float)/sizeof(T); 
}

    template <typename T>
void deviceMalloc(T** ptr, int size)
{
    ck(cudaMalloc((void**)ptr, sizeof(T) * size));
}

    template <typename T>
void deviceMemset(T* ptr, int value, int size)
{
    ck(cudaMemset((void*)ptr,0, sizeof(T) * size));
}

    template <typename T>
cudaError_t  deviceFree(T* & ptr){
    cudaError_t res= cudaSuccess;
    if(ptr!=NULL){
        res= cudaFree(ptr);
        ptr=NULL;
    }
    return res;
}

    template <typename T>
void deviceMemcpyHtoD(cudaStream_t stream, T* d_ptr,T* h_ptr, int size)
{
   ck(cudaMemcpyAsync(d_ptr, h_ptr,size *sizeof(T),cudaMemcpyHostToDevice,stream));
}


    template <typename T>
float castToFloat(T input){
    float output=(T)(input);
    return output;
}

template<>
float castToFloat(__half input){
    float output=__half2float(input);
    return output;
}

template <typename T>
void setRandom(T* data, int len){
        std::random_device rd;
        std::mt19937 mt(rd());
        std::uniform_real_distribution<float> dist(1.0, 10.0);
        for(int i=0;i<len;i++){
            if(sizeof(T)==4){
                data[i]=static_cast<T>(dist(mt));
            }else{
                data[i]=__float2half_rn(static_cast<float>(dist(mt)));
            }
        }
}
/*********************Npz &Npy File Process functions***********************/

/*********************The explicit instantiation part***********************/
template int numPerThread<float>();
template int numPerThread<__half>();

template float castToFloat<float>(float input);
template float castToFloat<__half>(__half input);

template  void deviceMalloc<float>(float** ptr, int size);
template  void deviceMemset<float>(float* ptr, int value, int size);
template  cudaError_t  deviceFree<float>(float* & ptr);
template  void deviceMemcpyHtoD<float>(cudaStream_t stream, float* d_ptr,float* h_ptr, int size);


template  void deviceMalloc<int>(int** ptr, int size);
template  void deviceMemset<int>(int* ptr, int value, int size);
template  cudaError_t  deviceFree<int>(int* & ptr);
template  void deviceMemcpyHtoD<int>(cudaStream_t stream, int* d_ptr,int* h_ptr, int size);


template  void deviceMalloc<__half>(__half** ptr, int size);
template  void deviceMemset<__half>(__half* ptr, int value, int  size);
template  cudaError_t  deviceFree<__half>(__half* & ptr);
template  void deviceMemcpyHtoD<__half>(cudaStream_t stream, __half* d_ptr,__half* h_ptr, int size);

template  void deviceMalloc<bool>(bool** ptr, int size);
template  void deviceMemset<bool>(bool* ptr, int value, int  size);
template  void deviceMemcpyHtoD<bool>(cudaStream_t stream, bool* d_ptr,bool* h_ptr, int size);
template  cudaError_t  deviceFree<bool>(bool* & ptr);

template void setRandom<__half>(__half* data, int len);
template void setRandom<float>(float* data, int len);
template void setRandom<int>(int* data, int len);
template void setRandom<bool>(bool* data, int len);
