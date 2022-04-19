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

#include "TrtLite.h"
#include "Utils.h"

simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger(simplelogger::TRACE);

static IHostMemory *BuildNetworkProc(IBuilder *builder, void *pData) {
    BuildEngineParam *pParam = (BuildEngineParam *)pData;
    unique_ptr<INetworkDefinition> network(builder->createNetworkV2(0));
    ITensor *tensor = network->addInput("input0", DataType::kFLOAT, Dims3{pParam->nChannel, pParam->nHeight, pParam->nWidth});
    vector<float> vKernel(pParam->nChannel * pParam->nChannel * 3 * 3);
    vector<float> vBias(pParam->nChannel);
    fill(vKernel, 1.0f);
    std::fill(vBias.begin(), vBias.end(), 0);
    IConvolutionLayer *conv = network->addConvolutionNd(*tensor, pParam->nChannel, DimsHW(3, 3), 
        Weights{DataType::kFLOAT, vKernel.data(), (int64_t)vKernel.size()}, 
        Weights{DataType::kFLOAT, vBias.data(), (int64_t)vBias.size()}
    );
    conv->setPaddingMode(PaddingMode::kSAME_LOWER);
    tensor = conv->getOutput(0);
    network->markOutput(*tensor);
    
    unique_ptr<IBuilderConfig> config(builder->createBuilderConfig());
    if (pParam->bFp16) {
        config->setFlag(BuilderFlag::kFP16);
    }
    Calibrator calib(1, pParam, "int8_cache.AppInt8");
    if (pParam->bInt8) {
        config->setFlag(BuilderFlag::kINT8);
        config->setInt8Calibrator(&calib);
    }
    if (pParam->bRefit) {
        config->setFlag(BuilderFlag::kREFIT);
    }
    builder->setMaxBatchSize(pParam->nMaxBatchSize);

    return builder->buildSerializedNetwork(*network, *config);
}

static IHostMemory *BuildNetworkProc_DynamicShape(IBuilder *builder, void *pData) {
    BuildEngineParam *pParam = (BuildEngineParam *)pData;
    unique_ptr<INetworkDefinition> network(builder->createNetworkV2(1));
    const char* szInputName = "input0";
    ITensor *tensor = network->addInput(szInputName, DataType::kFLOAT, Dims4{-1, pParam->nChannel, pParam->nHeight, pParam->nWidth});
    vector<float> vKernel(pParam->nChannel * pParam->nChannel * 3 * 3);
    vector<float> vBias(pParam->nChannel);
    fill(vKernel, 1.0f);
    std::fill(vBias.begin(), vBias.end(), 0);
    IConvolutionLayer *conv = network->addConvolutionNd(*tensor, pParam->nChannel, DimsHW(3, 3),
        Weights{DataType::kFLOAT, vKernel.data(), (int64_t)vKernel.size()},
        Weights{DataType::kFLOAT, vBias.data(), (int64_t)vBias.size()}
    );
    conv->setPaddingMode(PaddingMode::kSAME_LOWER);
    tensor = conv->getOutput(0);
    network->markOutput(*tensor);

    unique_ptr<IBuilderConfig> config(builder->createBuilderConfig());
    if (pParam->bFp16) {
        config->setFlag(BuilderFlag::kFP16);
    }
    /*
     * To run INT8 calibration for a network with dynamic shapes, calibration optimization profile must be set. See setCalibrationProfile()
     * If calibration optimization profile is not set, the first network optimization profile will be used as a calibration optimization profile. 
     * Calibration is performed using kOPT values of the profile. 
     */
    int nOptBatchSize = pParam->nMaxBatchSize; 
    Calibrator calib(nOptBatchSize, pParam, "int8_cache.AppInt8");
    if (pParam->bInt8) {
        config->setFlag(BuilderFlag::kINT8);
        config->setInt8Calibrator(&calib);
    }
    if (pParam->bRefit) {
        config->setFlag(BuilderFlag::kREFIT);
    }
    
    IOptimizationProfile *profile = builder->createOptimizationProfile();
    profile->setDimensions(szInputName, OptProfileSelector::kMIN, Dims4(1, pParam->nChannel, pParam->nHeight, pParam->nWidth));
    profile->setDimensions(szInputName, OptProfileSelector::kOPT, Dims4(nOptBatchSize, pParam->nChannel, pParam->nHeight, pParam->nWidth));
    profile->setDimensions(szInputName, OptProfileSelector::kMAX, Dims4(pParam->nMaxBatchSize, pParam->nChannel, pParam->nHeight, pParam->nWidth));
    config->addOptimizationProfile(profile);

    return builder->buildSerializedNetwork(*network, *config);
}

int main(int argc, char** argv) {
    int iDevice = 0;
    if (argc >= 2) iDevice = atoi(argv[1]);
    ck(cudaSetDevice(iDevice));
    cudaDeviceProp prop = {};
    ck(cudaGetDeviceProperties(&prop, iDevice));
    cout << "Using " << prop.name << endl;

    BuildEngineParam param = {2048, 4, 4, 8};
    param.bInt8 = !(argc >= 3 && atoi(argv[2]) == 0);
    int nBatch = 1;
    map<int, Dims> i2shape;
    i2shape.insert(make_pair(0, Dims4(nBatch, param.nChannel, param.nHeight, param.nWidth)));

    int iProc = argc >= 4 ? atoi(argv[3]) : 0;
    if (iProc < 0 || iProc > 2) {
        iProc = 0;
    }
    BuildNetworkProcType aProc[] = {BuildNetworkProc, BuildNetworkProc_DynamicShape};
    auto trt = unique_ptr<TrtLite>(TrtLiteCreator::Create(aProc[iProc], &param));
    trt->PrintInfo();
    
    vector<void *> vpBuf, vdpBuf;
    vector<IOInfo> vInfo;
    if (trt->GetEngine()->hasImplicitBatchDimension()) {
        vInfo = trt->ConfigIO(nBatch);
    } else {
        vInfo = trt->ConfigIO(i2shape);
    }
    for (auto info : vInfo) {
        cout << info.to_string() << endl;
        
        void *pBuf = nullptr;
        pBuf = new uint8_t[info.GetNumBytes()];
        vpBuf.push_back(pBuf);

        void *dpBuf = nullptr;
        ck(cudaMalloc(&dpBuf, info.GetNumBytes()));
        vdpBuf.push_back(dpBuf);

        if (info.bInput) {
            fill((float *)pBuf, info.GetNumBytes() / sizeof(float), 1.0f);
            ck(cudaMemcpy(dpBuf, pBuf, info.GetNumBytes(), cudaMemcpyHostToDevice));
        }
    }
    if (trt->GetEngine()->hasImplicitBatchDimension()) {
        trt->Execute(nBatch, vdpBuf);
    } else {
        trt->Execute(i2shape, vdpBuf);
    }
    for (int i = 0; i < vInfo.size(); i++) {
        auto &info = vInfo[i];
        if (info.bInput) {
            continue;
        }
        ck(cudaMemcpy(vpBuf[i], vdpBuf[i], info.GetNumBytes(), cudaMemcpyDeviceToHost));
    }

    print((float *)vpBuf[1], nBatch * param.nChannel * param.nHeight, param.nWidth);    
    
    for (int i = 0; i < vInfo.size(); i++) {
        delete[] (uint8_t *)vpBuf[i];
        ck(cudaFree(vdpBuf[i]));
    }

    return 0;
}
