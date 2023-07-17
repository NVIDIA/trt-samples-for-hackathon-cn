#include "NvInfer.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <cuda.h>
#include <cuda_fp16.h>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

using namespace nvinfer1;

extern "C" nvinfer1::IPluginCreator *const *getPluginCreators(int32_t &nbCreators);

inline void caughtError(const std::exception &e)
{
    std::cout << e.what() << std::endl;
}

// Write values into buffer
template<typename T>
void write(char *&buffer, const T &val)
{
    std::memcpy(buffer, &val, sizeof(T));
    buffer += sizeof(T);
}

// Read values from buffer
template<typename T>
T read(const char *&buffer)
{
    T val {};
    std::memcpy(&val, buffer, sizeof(T));
    buffer += sizeof(T);
    return val;
}

template<typename Dtype>
struct CudaBind
{
    size_t mSize;
    Dtype *mPtr;

    CudaBind(size_t size)
    {
        mSize = size;
        assert(!cudaMalloc((void **)&mPtr, sizeof(Dtype) * mSize));
    }

    ~CudaBind()
    {
        if (mPtr != nullptr)
        {
            assert(!cudaFree(mPtr));
            mPtr = nullptr;
        }
    }
};

inline int64_t volume(Dims const &dims)
{
    return std::accumulate(dims.d, dims.d + dims.nbDims, int64_t {1}, std::multiplies<int64_t> {});
}

std::ostream &operator<<(std::ostream &o, TensorFormat fmt)
{
    switch (fmt)
    {
    case TensorFormat::kCDHW32: o << "CDHW32"; break;
    case TensorFormat::kCHW16: o << "CHW16"; break;
    case TensorFormat::kCHW2: o << "CHW2"; break;
    case TensorFormat::kCHW32: o << "CHW32"; break;
    case TensorFormat::kCHW4: o << "CHW4"; break;
    case TensorFormat::kDHWC8: o << "DHWC8"; break;
    case TensorFormat::kDLA_HWC4: o << "DLA_HWC4"; break;
    case TensorFormat::kDLA_LINEAR: o << "DLA_LINEAR"; break;
    case TensorFormat::kHWC16: o << "HWC16"; break;
    case TensorFormat::kHWC8: o << "HWC8"; break;
    case TensorFormat::kHWC: o << "HWC"; break;
    case TensorFormat::kLINEAR: o << "LINEAR"; break;
    }
    // coverity[autosar_cpp14_a8_4_9_violation : FALSE] http://nvbugs/200649380
    return o;
}

std::string getTypeString(nvinfer1::DataType type)
{
    switch (type)
    {
    case nvinfer1::DataType::kFLOAT: return "FLOAT";
    case nvinfer1::DataType::kHALF: return "HALF";
    case nvinfer1::DataType::kINT8: return "INT8";
    case nvinfer1::DataType::kUINT8: return "UINT8";
    case nvinfer1::DataType::kINT32: return "INT32";
    case nvinfer1::DataType::kBOOL: return "BOOL";
    default: return "UNKNOWN TYPE";
    }
}

template<typename T>
__global__ void circ_pad_kernel(T const *X, int const *all_pads, int const *orig_dims, T *Y, int const *Y_shape, int Y_len)
{
    int index  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < Y_len; i += stride)
    {
        int i3 = i % Y_shape[3];
        int i2 = (i / Y_shape[3]) % Y_shape[2];
        int i1 = (i / Y_shape[3] / Y_shape[2]) % Y_shape[1];
        int i0 = i / Y_shape[3] / Y_shape[2] / Y_shape[1];

        int j0 = (i0 - all_pads[0] + orig_dims[0]) % orig_dims[0];
        int j1 = (i1 - all_pads[2] + orig_dims[1]) % orig_dims[1];
        int j2 = (i2 - all_pads[4] + orig_dims[2]) % orig_dims[2];
        int j3 = (i3 - all_pads[6] + orig_dims[3]) % orig_dims[3];

        Y[i] = X[orig_dims[3] * orig_dims[2] * orig_dims[1] * j0 + orig_dims[3] * orig_dims[2] * j1 + orig_dims[3] * j2 + j3];
    }
}

class CircPadPlugin : public nvinfer1::IPluginV2DynamicExt
{
public:
    CircPadPlugin() = default;

    CircPadPlugin(std::vector<int32_t> pads):
        mPads(pads)
    {
    }

    CircPadPlugin(CircPadPlugin const &p) = default;

    CircPadPlugin(void const *serialData, size_t length)
    {
        assert(serialData != nullptr);

        char const *d = static_cast<char const *>(serialData);
        char const *a = d;

        int32_t padsSize = read<int32_t>(d);
        mPads.resize(padsSize);
        for (int i = 0; i < padsSize; ++i)
        {
            mPads[i] = read<int32_t>(d);
        }

        assert(d == a + length);
    }

    int32_t getNbOutputs() const noexcept override
    {
        return 1;
    }

    bool supportsFormatCombination(int32_t pos, PluginTensorDesc const *inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
    {
        PluginTensorDesc const &desc = inOut[pos];
        if (desc.format != TensorFormat::kLINEAR)
        {
            return false;
        }

        // first input should be float16 or float32
        if (pos == 0)
        {
            return (inOut[pos].type == nvinfer1::DataType::kFLOAT || inOut[pos].type == nvinfer1::DataType::kHALF);
        }

        // output should have the same type as the input
        if (pos == 1)
        {
            return (inOut[pos].type == inOut[0].type);
        }

        return false;
    }

    void configureWithFormat(nvinfer1::Dims const *, int32_t, nvinfer1::Dims const *, int32_t, nvinfer1::DataType type, nvinfer1::PluginFormat floatFormat, int32_t) noexcept override
    {
    }

    int32_t initialize() noexcept override
    {
        return 0;
    }

    void terminate() noexcept override
    {
        allPadsPtr.reset();
        origDimsPtr.reset();
        outDimsPtr.reset();
    }

    int32_t enqueue(PluginTensorDesc const *inputDesc, PluginTensorDesc const *outputDesc, void const *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept
    {
        auto inpDType = inputDesc[0].type;

        int32_t const blockSize = 256;
        int32_t const numBlocks = (volume(outputDesc[0].dims) + blockSize - 1) / blockSize;

        if (inpDType == DataType::kFLOAT)
        {
            circ_pad_kernel<float><<<numBlocks, blockSize, 0, stream>>>(static_cast<float const *>(inputs[0]), allPadsPtr->mPtr, origDimsPtr->mPtr, static_cast<float *>(outputs[0]), outDimsPtr->mPtr, volume(outputDesc[0].dims));
        }
        else if (inpDType == DataType::kHALF)
        {
            circ_pad_kernel<half><<<numBlocks, blockSize, 0, stream>>>(static_cast<half const *>(inputs[0]), allPadsPtr->mPtr, origDimsPtr->mPtr, static_cast<half *>(outputs[0]), outDimsPtr->mPtr, volume(outputDesc[0].dims));
        }
        else
        {
            assert(false && "inpDType not valid");
        }
        return 0;
    }

    size_t getSerializationSize() const noexcept override
    {
        return (mPads.size() + 1) * sizeof(int32_t);
    }

    void serialize(void *buffer) const noexcept override
    {
        assert(buffer != nullptr);
        char *d = static_cast<char *>(buffer);
        char *a = d;
        write(d, static_cast<int32_t>(mPads.size()));
        for (int i = 0; i < mPads.size(); ++i)
        {
            write(d, mPads[i]);
        }
        assert(d == a + getSerializationSize());
    }

    char const *getPluginType() const noexcept override
    {
        return "CircPadPlugin";
    }

    char const *getPluginVersion() const noexcept override
    {
        return "1";
    }

    nvinfer1::IPluginV2DynamicExt *clone() const noexcept override
    {
        return new CircPadPlugin(*this);
    }

    void destroy() noexcept override
    {
        delete this;
    }

    void setPluginNamespace(char const *libNamespace) noexcept override
    {
        mNamespace = libNamespace;
    }

    char const *getPluginNamespace() const noexcept override
    {
        return mNamespace.c_str();
    }

    DataType getOutputDataType(int index, const nvinfer1::DataType *inputTypes, int nbInputs) const noexcept
    {
        return inputTypes[0];
    }

    DimsExprs getOutputDimensions(int32_t outputIndex, DimsExprs const *inputs, int32_t nbInputs, IExprBuilder &exprBuilder) noexcept
    {
        nvinfer1::DimsExprs outDims {inputs[0]};
        int32_t             nbOutDims = inputs[0].nbDims;

        for (int32_t i = 0; i < mPads.size() / 2; ++i)
        {
            outDims.d[nbOutDims - i - 1] = exprBuilder.operation(
                nvinfer1::DimensionOperation::kSUM,
                *inputs[0].d[nbOutDims - i - 1],
                *exprBuilder.constant(mPads[i * 2] + mPads[i * 2 + 1]));
        }

        return outDims;
    }

    void configurePlugin(DynamicPluginTensorDesc const *in, int32_t nbInputs, DynamicPluginTensorDesc const *out, int32_t nbOutputs) noexcept
    {
        mN = in[0].desc.dims.nbDims;

        std::vector<int32_t> allPads(mN * 2);
        std::vector<int32_t> origDims(mN);
        std::vector<int32_t> outDims(mN);

        for (int32_t i = 0; i < mN; ++i)
        {
            origDims[i] = in[0].desc.dims.d[i];
            outDims[i]  = in[0].desc.dims.d[i];
        }

        for (int32_t i = 0; i < mPads.size() / 2; ++i)
        {
            outDims[mN - i - 1] += mPads[i * 2] + mPads[i * 2 + 1];
            allPads[mN * 2 - 2 * i - 2] = mPads[i * 2];
            allPads[mN * 2 - 2 * i - 1] = mPads[i * 2 + 1];
        }

        allPadsPtr  = std::make_shared<CudaBind<int32_t>>(mN * 2);
        origDimsPtr = std::make_shared<CudaBind<int32_t>>(mN);
        outDimsPtr  = std::make_shared<CudaBind<int32_t>>(mN);

        assert(!cudaMemcpy(allPadsPtr->mPtr, &allPads.front(), allPads.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
        assert(!cudaMemcpy(origDimsPtr->mPtr, &origDims.front(), origDims.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
        assert(!cudaMemcpy(outDimsPtr->mPtr, &outDims.front(), outDims.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    }

    size_t getWorkspaceSize(PluginTensorDesc const *inputs, int32_t nbInputs, PluginTensorDesc const *outputs, int32_t nbOutputs) const noexcept
    {
        return 0;
    }

private:
    std::vector<int32_t>               mPads {};
    int32_t                            mN {};
    std::shared_ptr<CudaBind<int32_t>> allPadsPtr {};
    std::shared_ptr<CudaBind<int32_t>> origDimsPtr {};
    std::shared_ptr<CudaBind<int32_t>> outDimsPtr {};
    std::string                        mNamespace;
};

class CircPadPluginCreator : public nvinfer1::IPluginCreator
{
public:
    CircPadPluginCreator()
    {
        mPluginAttributes.clear();
        mPluginAttributes.emplace_back(PluginField("pads", nullptr, PluginFieldType::kINT32, 1));
        mFC.nbFields = mPluginAttributes.size();
        mFC.fields   = mPluginAttributes.data();
    }

    const char *getPluginName() const noexcept
    {
        return "CircPadPlugin";
    }

    const char *getPluginVersion() const noexcept
    {
        return "1";
    }

    const PluginFieldCollection *getFieldNames() noexcept
    {
        return &mFC;
    }

    IPluginV2 *createPlugin(const char *name, const PluginFieldCollection *fc) noexcept
    {
        try
        {
            std::vector<int32_t> pads;

            for (int32_t i = 0; i < fc->nbFields; i++)
            {
                std::string field_name(fc->fields[i].name);
                if (field_name.compare("pads") == 0)
                {
                    pads.resize(fc->fields[i].length);
                    auto const *padsPtr = static_cast<int32_t const *>(fc->fields[i].data);
                    std::copy_n(padsPtr, fc->fields[i].length, pads.data());
                }
            }

            return new CircPadPlugin(pads);
        }
        catch (const std::exception &e)
        {
            caughtError(e);
        }
        return nullptr;
    }

    IPluginV2 *deserializePlugin(
        const char *name,
        const void *serialData,
        size_t      serialLength) noexcept
    {
        try
        {
            return new CircPadPlugin(serialData, serialLength);
        }
        catch (const std::exception &e)
        {
            caughtError(e);
        }
        return nullptr;
    }

    void setPluginNamespace(const char *libNamespace) noexcept
    {
        mNamespace = libNamespace;
    }

    const char *getPluginNamespace() const noexcept
    {
        return mNamespace.c_str();
    }

private:
    nvinfer1::PluginFieldCollection    mFC;
    std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string                        mNamespace;
};

nvinfer1::IPluginCreator *const *getPluginCreators(int32_t &nbCreators)
{
    nbCreators                                                    = 1;
    static CircPadPluginCreator                    creator        = CircPadPluginCreator();
    static std::vector<nvinfer1::IPluginCreator *> pluginCreators = {&creator};
    return pluginCreators.data();
}

REGISTER_TENSORRT_PLUGIN(CircPadPluginCreator);
