#include "calibrator.h"

using namespace nvinfer1;

MyCalibrator::MyCalibrator(const std::string& calibrationDataFile, const int nCalibration, const Dims32 dim, const std::string& cacheFile) :
    nCalibration(nCalibration), dim(dim), cacheFile(cacheFile), iBatch(0)
{
#ifdef DEBUG
    std::cout << "[MyCalibrator::MyCalibrator]" << std::endl;
#endif
    cnpy::npz_t    npzFile = cnpy::npz_load(calibrationDataFile);
    cnpy::NpyArray array = npzFile[std::string("calibrationData")];
    auto pDataTemp = array.data<float>();
    pData = (float*)malloc(array.num_bytes());
    memcpy(pData, pDataTemp, array.num_bytes());

    if (pData == nullptr)
    {
        std::cout << "Failed getting calibration data!" << std::endl;
        return;
    }

    nElement = 1;
    for (int i = 0; i < dim.nbDims; ++i)
    {
        nElement *= dim.d[i];
    }
    bufferSize = sizeof(float) * nElement;
    nBatch = array.num_bytes() / bufferSize;
    cudaMalloc((void**)&bufferD, bufferSize);

    return;
}

MyCalibrator::~MyCalibrator() noexcept
{
#ifdef DEBUG
    std::cout << "[MyCalibrator::~MyCalibrator]" << std::endl;
#endif
    if (bufferD != nullptr)
    {
        cudaFree(bufferD);
    }
    if (pData != nullptr) {
        free(pData);
    }
    return;
}

int32_t MyCalibrator::getBatchSize() const noexcept
{
#ifdef DEBUG
    std::cout << "[MyCalibrator::getBatchSize]" << std::endl;
#endif
    return dim.d[0];
}

bool MyCalibrator::getBatch(void* bindings[], char const* names[], int32_t nbBindings) noexcept
{
#ifdef DEBUG
    std::cout << "[MyCalibrator::getBatch]" << std::endl;
#endif
    if (iBatch < nBatch)
    {
        cudaMemcpy(bufferD, &pData[iBatch * nElement], bufferSize, cudaMemcpyHostToDevice);
        bindings[0] = bufferD;
        iBatch++;
        return true;
    }
    else
    {
        return false;
    }
}

void const* MyCalibrator::readCalibrationCache(std::size_t& length) noexcept
{
#ifdef DEBUG
    std::cout << "[MyCalibrator::readCalibrationCache]" << std::endl;
#endif
    std::fstream f;
    f.open(cacheFile, std::fstream::in);
    if (f.fail())
    {
        std::cout << "Failed finding cache file!" << std::endl;
        return nullptr;
    }
    char* ptr = new char[length];
    if (f.is_open())
    {
        f >> ptr;
    }
    return ptr;
}

void MyCalibrator::writeCalibrationCache(void const* ptr, std::size_t length) noexcept
{
#ifdef DEBUG
    std::cout << "[MyCalibrator::writeCalibrationCache]" << std::endl;
#endif
    std::ofstream f(cacheFile, std::ios::binary);
    if (f.fail())
    {
        std::cout << "Failed opening cache file to write!" << std::endl;
        return;
    }
    f.write(static_cast<char const*>(ptr), length);
    if (f.fail())
    {
        std::cout << "Failed saving cache file!" << std::endl;
        return;
    }
    f.close();
}
