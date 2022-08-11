#include "upsample.h"

using namespace nvinfer1;
// using namespace std;

static __device__ float Bilinear(const float *pSrc, const int nSrcPitch, const int nSrcWidth, const int nSrcHeight, float x, float y)
{
    x = min(max(x - 0.5f, 0.0f), nSrcWidth - 1.0f);
    y = min(max(y - 0.5f, 0.0f), nSrcHeight - 1.0f);

    int x_low = (int)x;
    int y_low = (int)y;

    float lx = x - x_low;
    float ly = y - y_low;
    float hx = 1.0f - lx;
    float hy = 1.0f - ly;

    int x_high = min(x_low + 1, nSrcWidth - 1);
    int y_high = min(y_low + 1, nSrcHeight - 1);

    float v1 = *(float *)((uint8_t *)pSrc + y_low * nSrcPitch + x_low * sizeof(float));
    float v2 = *(float *)((uint8_t *)pSrc + y_low * nSrcPitch + x_high * sizeof(float));
    float v3 = *(float *)((uint8_t *)pSrc + y_high * nSrcPitch + x_low * sizeof(float));
    float v4 = *(float *)((uint8_t *)pSrc + y_high * nSrcPitch + x_high * sizeof(float));
    float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

    return w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;
}

static inline __device__ float Nearest(const float *pSrc, const int nSrcPitch, float x, float y)
{
    return *(float *)((uint8_t *)pSrc + int(y) * nSrcPitch + int(x) * sizeof(float));
}

static __global__ void Scale_fp32(bool bNearest, int nImage, float *pSrc, int nSrcPitch, int nSrcWidth, int nSrcHeight, float *pDst, int nDstPitch, int nDstWidth, int nDstHeight)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nDstWidth || y >= nDstHeight)
    {
        return;
    }
    float fxScale = 1.0f * nSrcWidth / nDstWidth, fyScale = 1.0f * nSrcHeight / nDstHeight;
    for (int i = 0; i < nImage; i++)
    {
        *(float *)((uint8_t *)pDst + nDstPitch * y + x * sizeof(float)) =
            bNearest ? Nearest(pSrc, nSrcPitch, (x + 0.5f) * fxScale, (y + 0.5f) * fyScale) : Bilinear(pSrc, nSrcPitch, nSrcWidth, nSrcHeight, (x + 0.5f) * fxScale, (y + 0.5f) * fyScale);
        pSrc = (float *)((uint8_t *)pSrc + nSrcPitch * nSrcHeight);
        pDst = (float *)((uint8_t *)pDst + nDstPitch * nDstHeight);
    }
}

UpsamplePlugin::UpsamplePlugin(int nScaleFactor, bool bNearest):
    nScaleFactor(nScaleFactor), bNearest(bNearest) {}

Dims UpsamplePlugin::getOutputDimensions(int index, const Dims *pInputDim, int nInputDim)
{
    assert(index == 0 && nInputDim == 1 && pInputDim[0].nbDims == 3);
    return Dims3(pInputDim[0].d[0] /*C*/, pInputDim[0].d[1] * nScaleFactor, pInputDim[0].d[2] * nScaleFactor);
}

bool UpsamplePlugin::supportsFormat(DataType type, PluginFormat format) const
{
    return type == DataType::kFLOAT && format == PluginFormat::kNCHW;
}

void UpsamplePlugin::configureWithFormat(const Dims *pInputDim, int nInputDim, const Dims *pOutputDim, int nOutputDim, DataType dataType, PluginFormat pluginFormat, int maxBatchSize)
{
    assert(nInputDim == 1 && dataType == DataType::kFLOAT && pluginFormat == PluginFormat::kNCHW);
    nChannel   = pInputDim[0].d[0];
    nSrcHeight = pInputDim[0].d[1];
    nSrcWidth  = pInputDim[0].d[2];
}

int UpsamplePlugin::enqueue(int nBatch, const void *const *inputs, void **outputs, void *workspace, cudaStream_t stream)
{
    // cout << "NCHW: " << nBatch << "x" << nChannel << "x" << nSrcHeight << "x" << nSrcWidth
    //        << "; nScaleFactor: " << nScaleFactor << "; bNearest: " << bNearest << endl;
    float *dpSrc = (float *)inputs[0], *dpDst = (float *)outputs[0];
    int    nDstWidth = nSrcWidth * nScaleFactor, nDstHeight = nSrcHeight * nScaleFactor;
    Scale_fp32<<<dim3((nDstWidth + 15) / 16, (nDstHeight + 15) / 16), dim3(16, 16), 0, stream>>>(
        bNearest, nBatch * nChannel, dpSrc, nSrcWidth * sizeof(float), nSrcWidth, nSrcHeight, dpDst, nDstWidth * sizeof(float), nDstWidth, nDstHeight);
    return 0;
}
