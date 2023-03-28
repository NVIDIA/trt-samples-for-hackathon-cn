#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>

using namespace nvinfer1;

class MyCalibrator : public IInt8EntropyCalibrator2
{
private:
    int                             nCalibration {0};
    int                             nBatchSize {0};
    std::map<std::string, Dims32>   shapeMap {};
    std::map<std::string, DataType> dataTypeMap {};
    int                             iBatch {0};
    std::map<std::string, size_t>   sizeMap {};
    std::map<std::string, void *>   bufferHMap {};
    std::map<std::string, void *>   bufferDMap {};
    std::string                     cacheFile {};

public:
    MyCalibrator(const int                             nCalibration,
                 const int                             nBatchSize,
                 const std::map<std::string, Dims32>   shapeMap,
                 const std::map<std::string, DataType> dataTypeMap,
                 const std::string                     cacheFile);
    ~MyCalibrator() noexcept;
    int32_t     getBatchSize() const noexcept;
    bool        getBatch(void *bindings[], char const *names[], int32_t nbBindings) noexcept;
    void const *readCalibrationCache(std::size_t &length) noexcept;
    void        writeCalibrationCache(void const *ptr, std::size_t length) noexcept;
};