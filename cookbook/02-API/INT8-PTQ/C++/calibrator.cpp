#include "calibrator.h"

#include "cookbookHelper.cuh"

using namespace nvinfer1;

MyCalibrator::MyCalibrator(const int                             nCalibration,
                           const int                             nBatchSize,
                           const std::map<std::string, Dims32>   shapeMap,
                           const std::map<std::string, DataType> dataTypeMap,
                           const std::string                     cacheFile):
    nCalibration(nCalibration),
    nBatchSize(nBatchSize),
    shapeMap(shapeMap),
    dataTypeMap(dataTypeMap),
    iBatch(0),
    cacheFile(cacheFile)
{
#ifdef DEBUG
    std::cout << "[MyCalibrator::MyCalibrator]" << std::endl;
#endif
    for (auto element : shapeMap)
    {
        std::string name {element.first};
        size_t      n = nBatchSize;
        for (int i = 1; i < element.second.nbDims; ++i)
        {
            n *= element.second.d[i];
        }
        sizeMap[name]    = sizeof(dataTypeToSize(this->dataTypeMap[name])) * n;
        bufferHMap[name] = (void *)new char[sizeMap[name]];
        bufferDMap[name] = nullptr;
        cudaMalloc((void **)&bufferDMap[name], sizeMap[name]);
    }
    return;
}

MyCalibrator::~MyCalibrator() noexcept
{
#ifdef DEBUG
    std::cout << "[MyCalibrator::~MyCalibrator]" << std::endl;
#endif
    for (auto element : bufferHMap)
    {
        delete[](char *) element.second;
    }

    for (auto element : bufferDMap)
    {
        cudaFree(element.second);
    }
    return;
}

int32_t MyCalibrator::getBatchSize() const noexcept
{
#ifdef DEBUG
    std::cout << "[MyCalibrator::getBatchSize]" << std::endl;
#endif
    return nBatchSize;
}

bool MyCalibrator::getBatch(void *bindings[], char const *names[], int32_t nbBindings) noexcept
{
#ifdef DEBUG
    std::cout << "[MyCalibrator::getBatch]" << std::endl;
#endif
    if (iBatch >= nCalibration)
    {
        return false;
    }

    for (int i = 0; i < nbBindings; ++i) // nbBindings is number of input tensors, not all input / output tensors
    {
        std::string name {names[i]};
        cudaMemcpy(bufferDMap[name], &bufferHMap[name], sizeMap[name], cudaMemcpyHostToDevice);
        bindings[i] = bufferDMap[name];
    }
    iBatch++;
    return true;
}

void const *MyCalibrator::readCalibrationCache(std::size_t &length) noexcept
{
#ifdef DEBUG
    std::cout << "[MyCalibrator::readCalibrationCache]" << std::endl;
#endif
    std::fstream f;
    f.open(cacheFile, std::fstream::in);
    if (f.fail())
    {
        std::cout << "Failed finding " << cacheFile << std::endl;
        return nullptr;
    }

    std::cout << "Succeeded finding " << cacheFile << std::endl;
    char *ptr = new char[length];
    if (f.is_open())
    {
        f >> ptr;
    }
    return ptr;
}

void MyCalibrator::writeCalibrationCache(void const *ptr, std::size_t length) noexcept
{
#ifdef DEBUG
    std::cout << "[MyCalibrator::writeCalibrationCache]" << std::endl;
#endif
    std::ofstream f(cacheFile, std::ios::binary);
    if (f.fail())
    {
        std::cout << "Failed opening " << cacheFile << " to write" << std::endl;
        return;
    }
    f.write(static_cast<char const *>(ptr), length);
    if (f.fail())
    {
        std::cout << "Failed saving " << cacheFile << std::endl;
        return;
    }
    f.close();
}