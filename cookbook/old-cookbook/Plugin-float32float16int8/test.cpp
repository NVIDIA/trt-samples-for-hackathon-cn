#include <iostream>
#include <fstream>
#include <vector>
#include <iterator>
#include <string>
#include <iomanip>
#include <cstdlib>
#include <unistd.h>
#include <cuda_runtime_api.h>
#include "ScalarAdditionPlugin.h"

using namespace nvinfer1;

#define ck(call) check(call, __LINE__, __FILE__)

const int bIn = 1, cIn = 3, hIn = 4, wIn = 5;
const int calibCount = 10;
const std::string cacheFile = "./calibration.cache";
const float addend = 0.5f;

inline bool check(cudaError_t e, int iLine, const char *szFile)
{
    if (e != cudaSuccess)
    {
        std::cout << "CUDA runtime API error " << cudaGetErrorName(e) << " at line " << iLine << " in file " << szFile << std::endl;
        return false;
    }
    return true;
}

class Logger : public nvinfer1::ILogger
{
public:
    Logger(Severity severity = Severity::kINFO) : reportableSeverity(severity) {}

    void log(Severity severity, const char* msg) override
    {
        if (severity > reportableSeverity)
            return;

        switch (severity)
        {
        case Severity::kINTERNAL_ERROR: std::cerr << "INTERNAL_ERROR: ";break;
        case Severity::kERROR:          std::cerr << "ERROR: ";         break;
        case Severity::kWARNING:        std::cerr << "WARNING: ";       break;
        case Severity::kINFO:           std::cerr << "INFO: ";          break;
        default:                        std::cerr << "UNKNOWN: ";       break;
        }
        std::cerr << msg << std::endl;
    }

    Severity reportableSeverity;
};
static Logger gLogger(ILogger::Severity::kINFO);

void printFloat(const std::vector<float> &v, Dims4 dimOut)
{
    std::cout << "B|C|H" << std::endl;
    for (int b = 0; b < dimOut.d[0]; b++)
    {
        for (int c = 0; c < dimOut.d[1]; c++)
        {
            for (int h = 0; h < dimOut.d[2]; h++)
            {
                std::cout << b << "|" << c << "|" << h << ": ";
                for (int w = 0; w < dimOut.d[3]; w++)
                    std::cout << std::fixed << std::setprecision(4) << 
                                 v[((b * dimOut.d[1] + c) * dimOut.d[2] + h)*dimOut.d[3] + w] << 
                                 " ";
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
    }
}

void printInt8(const std::vector<float> &v, Dims4 dimOut)
{
    std::cout << "B|C|H" << std::endl;
    for (int b = 0; b < dimOut.d[0]; b++)
    {
        for (int c = 0; c < dimOut.d[1]; c++)
        {
            for (int h = 0; h < dimOut.d[2]; h++)
            {
                std::cout << b << "|" << c << "|" << h << ": ";
                for (int w = 0; w < dimOut.d[3]; w++)
                    std::cout << std::fixed << std::setprecision(0) << 
                                 (int)*((char*)v.data() + ((b * dimOut.d[1] + c) * dimOut.d[2] + h)*dimOut.d[3] + w) << 
                                 " ";
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
    }
}

void printDim(Dims const &dim)
{
    for (int i = 0; i < dim.nbDims; i++)
        std::cout << dim.d[i] << " ";
    std::cout << std::endl;
}

void fill(std::vector<float> &v)
{
    for (int i = 0; i < v.size(); i++)
        v[i] = (float)i / v.size();
}

class MyCalibrator: public IInt8EntropyCalibrator2
{
private:
    int               calibCount;
    const Dims4       shape;
    const int         dataSize;    
    void              *dIn;
    const std::string cacheFile;
    std::vector<char> cache;

public:
    MyCalibrator(int calibCount, Dims4 inputShape, const std::string& cacheFile):
        calibCount(calibCount), shape(inputShape),dataSize(shape.d[1]*shape.d[2]*shape.d[3]),
        dIn(nullptr),cacheFile(cacheFile)
    {
        ck(cudaMalloc(&dIn, dataSize*sizeof(float)));          
    }

    ~MyCalibrator()
    {
        ck(cudaFree(dIn));
    }

    int getBatchSize() const override
    {
        return shape.d[0];
    }

    bool getBatch(void* bindings[], const char* names[], int nbBindings) override
    {
        calibCount--;
        if (calibCount<=0)
            return false;

        std::vector<float> data(dataSize);
        for(int i=0;i<data.size();i++)
            data[i] = rand()/float(RAND_MAX);
        ck(cudaMemcpy(dIn, data.data(), dataSize*sizeof(float),cudaMemcpyHostToDevice));
        bindings[0] = dIn;
        return true;
    }

    const void* readCalibrationCache(size_t& length) override
    {
        cache.clear();
        std::ifstream fIn(cacheFile, std::ios::binary);
        fIn >> std::noskipws;
        if (fIn.good())
            std::copy(std::istream_iterator<char>(fIn), std::istream_iterator<char>(), std::back_inserter(cache));
        length = cache.size();
        return length ? cache.data() : nullptr;
    }

    void writeCalibrationCache(const void* cache, size_t length) override
    {
        std::ofstream fOut(cacheFile, std::ios::binary);
        fOut.write(reinterpret_cast<const char*>(cache), length);
    }
};

ICudaEngine* loadEngine(const std::string& trtFile)
{
    std::ifstream engineFile(trtFile, std::ios::binary);
    long int fsize = 0;
    engineFile.seekg(0, engineFile.end);
    fsize = engineFile.tellg();
    engineFile.seekg(0, engineFile.beg);
    std::vector<char> engineData(fsize);
    engineFile.read(engineData.data(), fsize);

    IRuntime *runtime{createInferRuntime(gLogger)}; 
    ICudaEngine* engine = runtime->deserializeCudaEngine(engineData.data(), fsize, nullptr);
    runtime->destroy();
    return engine;
}

bool saveEngine(const ICudaEngine *engine, const std::string& fileName)
{
    std::ofstream engineFile(fileName, std::ios::binary);
    if (!engineFile)
    {
        std::cout << "Failed opening file to write" << std::endl;
        return false;
    }
    IHostMemory *serializedEngine{engine->serialize()};
    if (serializedEngine == nullptr)
    {
        std::cout << "Failed serializaing engine" << std::endl;
        return false;
    }
    engineFile.write(static_cast<char*>(serializedEngine->data()), serializedEngine->size());
    serializedEngine->destroy();
    return !engineFile.fail();
}

ICudaEngine* buildEngine(Dims3 dimIn, const std::string& dataType)
{

    IBuilder* builder = createInferBuilder(gLogger);
    builder->setMaxBatchSize(16);
    builder->setMaxWorkspaceSize(4 << 30);
    builder->setFp16Mode(dataType == std::string("float16"));
    builder->setInt8Mode(dataType == std::string("int8"));
    builder->setStrictTypeConstraints(true);
    MyCalibrator myCalib = MyCalibrator(calibCount, Dims4{1,dimIn.d[0],dimIn.d[1],dimIn.d[2]}, cacheFile);
	builder->setInt8Calibrator(&myCalib);
    INetworkDefinition* network = builder->createNetworkV2(1<<int(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));

    ITensor *inputTensor = network->addInput("input", DataType::kFLOAT, dimIn);

    Weights w{DataType::kFLOAT, &addend, 1};
    ScalarAdditionPlugin *plugin = new ScalarAdditionPlugin(w);

    ITensor *tensorList[] = {inputTensor};
    IPluginV2Layer *pluginLayer = network->addPluginV2(tensorList, 1, *plugin);
    
    network->markOutput(*pluginLayer->getOutput(0));
    network->getOutput(0)->setType((dataType == std::string("int8"))?(DataType::kINT8):(DataType::kFLOAT));
    network->getOutput(0)->setAllowedFormats(1U << int(TensorFormat::kLINEAR));

    ICudaEngine* engine = builder->buildCudaEngine(*network);
    network->destroy();
    builder->destroy();    
    return engine;
}

void run(const std::string& dataType)
{
    const int nIn = cIn * hIn * wIn;
    ICudaEngine *engine = nullptr;
    std::string trtFile = "./engine-" + dataType + ".trt";    
    if(access(trtFile.c_str(),F_OK) == 0)
    {
        engine = loadEngine(trtFile);        
        if(engine == nullptr) 
        {   
            std::cout << "failed laoding engine!" << std::endl;
            return;
        }
        std::cout << "succeeded loading engine!" << std::endl;
    }
    else
    {
        engine = buildEngine({cIn,hIn,wIn}, dataType);
        if(engine == nullptr)
        {
            std::cout << "Failed building engine!" << std::endl;
            return;
        }
        std::cout << "succeeded building engine!" << std::endl;
        saveEngine(engine, trtFile);
    }

    IExecutionContext* context = engine->createExecutionContext();

    cudaStream_t stream;
    ck(cudaStreamCreate(&stream));
    
    std::vector<float> input1_h(nIn), output1_h(nIn);
    fill(input1_h);

    std::vector<void*> binding = {nullptr, nullptr};
    ck(cudaMalloc(&binding[0], input1_h.size() * sizeof(engine->getBindingDataType(0))));
    ck(cudaMalloc(&binding[1], output1_h.size() * sizeof(engine->getBindingDataType(1))));

    ck(cudaMemcpyAsync(binding[0], input1_h.data(), input1_h.size() * sizeof(engine->getBindingDataType(0)), cudaMemcpyHostToDevice, stream));
    context->executeV2(binding.data());
    ck(cudaMemcpyAsync(output1_h.data(), binding[1], output1_h.size() * sizeof(engine->getBindingDataType(1)), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    if(dataType == std::string("int8"))
		printInt8(output1_h,{bIn,cIn,hIn,wIn});
    else
		printFloat(output1_h,{bIn,cIn,hIn,wIn});

    context->destroy();
    cudaStreamDestroy(stream);
    ck(cudaFree(binding[0]));
    ck(cudaFree(binding[1]));
}

int main()
{
    srand(97);
    int iDevice = 0;
    ck(cudaSetDevice(iDevice));
    cudaDeviceProp prop = {};
    ck(cudaGetDeviceProperties(&prop, iDevice));
    std::cout << "Using " << prop.name << std::endl;
    run(std::string("float32"));
    run(std::string("float16"));
    run(std::string("int8"));
    return 0;
}
