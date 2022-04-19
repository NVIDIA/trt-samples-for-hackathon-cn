/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <vector>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <sys/stat.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "Logger.h"

inline bool check(CUresult e, int iLine, const char *szFile) {
    if (e != CUDA_SUCCESS) {
        const char *szErrName = NULL;
        cuGetErrorName(e, &szErrName);
        LOG(FATAL) << "CUDA driver API error " << szErrName << " at line " << iLine << " in file " << szFile;
        return false;
    }
    return true;
}

inline bool check(cudaError_t e, int iLine, const char *szFile) {
    if (e != cudaSuccess) {
        LOG(ERROR) << "CUDA runtime API error " << cudaGetErrorName(e) << " at line " << iLine << " in file " << szFile;
        return false;
    }
    return true;
}

inline bool check(bool bSuccess, int iLine, const char *szFile) {
    if (!bSuccess) {
        LOG(ERROR) << "Error at line " << iLine << " in file " << szFile;
        return false;
    }
    return true;
}

#define ck(call) check(call, __LINE__, __FILE__)

class BufferedFileReader {
public:
    BufferedFileReader(const char *szFileName, bool bPartial = false) {
        struct stat st;
        if (stat(szFileName, &st) != 0) {
            LOG(WARNING) << "File " << szFileName << " does not exist.";
            return;
        }
        
        nSize = st.st_size;
        while (true) {
            try {
                pBuf = new uint8_t[nSize + 1];
                if (nSize != st.st_size) {
                    LOG(WARNING) << "File is too large - only " << std::setprecision(4) << 100.0 * nSize / (uint32_t)st.st_size << "% is loaded"; 
                }
                break;
            } catch(std::bad_alloc&) {
                if (!bPartial) {
                    LOG(ERROR) << "Failed to allocate memory in BufferedReader";
                    return;
                }
                nSize = (uint32_t)(nSize * 0.9);
            }
        }
        
        FILE *fp = fopen(szFileName, "rb");
        size_t nRead = fread(pBuf, 1, nSize, fp);
        pBuf[nSize] = 0;
        fclose(fp);

        if (nRead != nSize) {
        	LOG(ERROR) << "nRead != nSize";
        }
    }
    ~BufferedFileReader() {
        if (pBuf) {
            delete[] pBuf;
        }
    }
    bool GetBuffer(uint8_t **ppBuf, uint32_t *pnSize) {
        if (!pBuf) {
            return false;
        }

        if (ppBuf) *ppBuf = pBuf;
        if (pnSize) *pnSize = nSize;
        return true;
    }

private:
    uint8_t *pBuf = NULL;
    uint32_t nSize = 0;
};

class StopWatch {
public:
    void Start() {
        t0 = std::chrono::high_resolution_clock::now();
    }
    double Stop() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch() - t0.time_since_epoch()).count() / 1.0e9;
    }

private:
    std::chrono::high_resolution_clock::time_point t0;
};

inline std::string to_string(nvinfer1::Dims const &dim) {
    std::ostringstream oss;
    oss << "(";
    for (int i = 0; i < dim.nbDims; i++) {
        oss << dim.d[i] << ", ";
    }
    oss << ")";
    return oss.str();
}

template<class T>
void print(T *v, int yMax, int xMax, int yStart = 0, int yStop = INT_MAX, int xStart = 0, int xStop = INT_MAX) {
	yStart = std::max(yStart, 0);
	yStop = std::min(yStop, yMax);
	xStart = std::max(xStart, 0);
	xStop = std::min(xStop, xMax);
	if (yStart != 0) std::cout << "..." << std::endl;
    for (int y = yStart; y < yStop; y++) {
    	if (xStart != 0) std::cout << "... ";
        for (int x = xStart; x < xStop; x++) {
        	if (sizeof(T) == 1) {
        		std::cout << std::setw(4) << (int)(int8_t)v[y * xMax + x] << " ";
        	} else {
        		std::cout << std::setw(6) << std::fixed << std::setprecision(3) << v[y * xMax + x] << " ";
        	}
        }
        if (xStop != xMax) std::cout << "...";
        std::cout << std::endl;
    }
    if (yStop != yMax) std::cout << "..." << std::endl;
    std::cout << std::endl;
}

template<class T>
inline void fill(T *v, int nSize, float factor) {
    for (int i = 0; i < nSize; i++) {
        v[i] = (T)((2.0f * (rand() - 1) / RAND_MAX - 1.0f) * factor);
    }
}

template<class T>
inline void fill(std::vector<T> &v, float factor) {
    fill(v.data(), v.size(), factor);
}

struct BuildEngineParam {
    int nMaxBatchSize;
    int nChannel, nHeight, nWidth;
    std::size_t nMaxWorkspaceSize;
    bool bFp16, bInt8, bRefit;
};

class Calibrator : public nvinfer1::IInt8Calibrator {
public:
	Calibrator(int nBatchSize, BuildEngineParam *pParam, const char *szCacheFileName) : nBatchSize(nBatchSize), 
		nValue(nBatchSize * pParam->nChannel * pParam->nHeight * pParam->nWidth), strCacheFileName(szCacheFileName)
	{
		pInput = new float[nValue];
		ck(cudaMalloc(&dpInput, nValue * sizeof(float)));
	}
	virtual ~Calibrator() {
		if (pInput) delete[] pInput;
		if (dpInput) ck(cudaFree(dpInput));
		if (pReader) delete pReader;
	}
    virtual int getBatchSize() const noexcept override {
    	return nBatchSize;
    }
    virtual bool getBatch(void* adpInput[], const char* aszInputName[], int nInput) noexcept override {
    	if (iRound++ >= nTotalRound) {
    		return false;
    	}
    	//fill(pInput, nValue, 1.0f);
    	for (int i=0;i<nValue;++i)
    	{
    	    pInput[i] = -1.0f + 2/nValue * i;
    	}
    	ck(cudaMemcpy(dpInput, pInput, nValue * sizeof(float), cudaMemcpyHostToDevice));
    	adpInput[0] = dpInput;
    	return true;
    }
    virtual const void* readCalibrationCache(std::size_t& length) noexcept override {
    	pReader = new BufferedFileReader(strCacheFileName.c_str());
    	uint8_t *pData = nullptr;
    	uint32_t size = 0;
    	pReader->GetBuffer(&pData, &size);
    	
    	length = size;
    	return pData;
    }
    virtual void writeCalibrationCache(const void* ptr, std::size_t length) noexcept override {
        std::ofstream of(strCacheFileName, std::ios_base::binary);
        of.write((char *)ptr, length);
    }
    virtual nvinfer1::CalibrationAlgoType getAlgorithm() noexcept override {
    	return nvinfer1::CalibrationAlgoType::kENTROPY_CALIBRATION;
    }
    
private:
    const int nValue, nBatchSize;
    float *pInput = nullptr, *dpInput = nullptr;
    int iRound = 0, nTotalRound = 1;
    BufferedFileReader *pReader = nullptr;
    const std::string strCacheFileName;
};

typedef cudaError_t (*MallocProcType)(void **, size_t);
typedef cudaError_t (*FreeProcType)(void *);
template<MallocProcType MallocProc, FreeProcType FreeProc>
class Buffer {
public:
    Buffer(int nBufSize = 0) {
        Resize(nBufSize);
    }
    ~Buffer() {
        Resize(0);
    }
    void Resize(size_t nBufSize) {
        if (pBuf) {
            ck(FreeProc(pBuf));
            pBuf = nullptr;
        }
        if (nBufSize) {
            ck(MallocProc((void **)&pBuf, nBufSize));
        }
        this->nBufSize = nBufSize;
    }
    std::vector<void *> GetBufferPointers(std::vector<size_t> const &vSize) {
        size_t nSum = 0;
        for (size_t size : vSize) {
            nSum += (size / nAlignment + 1) * nAlignment;
        }
        if (nBufSize < nSum) {
            Resize(nSum);
        }

        std::vector<void *> ret;
        nSum = 0;
        for (size_t size : vSize) {
            ret.push_back(pBuf + nSum);
            nSum += (size / nAlignment + 1) * nAlignment;
        }
        return ret;
    }
    void *GetBuf() {
        return pBuf;
    }
    size_t GetBufSize() {
        return nBufSize;
    }
protected:
    size_t nBufSize = 0;
    uint8_t *pBuf = nullptr;
    static const size_t nAlignment = 128;
};
typedef Buffer<cudaMalloc, cudaFree> DeviceBuffer;
typedef Buffer<cudaMallocHost, cudaFreeHost> HostBuffer;
