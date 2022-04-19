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

#include "Utils.h"
#include "TrtLite.h"

simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger(simplelogger::TRACE);

IHostMemory *BuildNetworkProc(IBuilder *builder, void *pData) {
    BuildEngineParam *pParam = (BuildEngineParam *)pData;
    unique_ptr<INetworkDefinition> network(builder->createNetworkV2(1));
    const char* szInputName = "input0";
    ITensor *input = network->addInput(szInputName, DataType::kFLOAT, Dims4(-1, pParam->nChannel, -1, -1));
    ITensor *tensor = network->addUnary(*input, UnaryOperation::kNEG)->getOutput(0);
    network->markOutput(*tensor);

    unique_ptr<IBuilderConfig> config(builder->createBuilderConfig());
    
    IOptimizationProfile *profile0 = builder->createOptimizationProfile();
    profile0->setDimensions(szInputName, OptProfileSelector::kMIN, Dims4(1, pParam->nChannel, 1, 1));
    profile0->setDimensions(szInputName, OptProfileSelector::kOPT, Dims4(pParam->nMaxBatchSize, pParam->nChannel, 1024, 1024));
    profile0->setDimensions(szInputName, OptProfileSelector::kMAX, Dims4(pParam->nMaxBatchSize, pParam->nChannel, 1024, 1024));
    config->addOptimizationProfile(profile0);

    IOptimizationProfile *profile1 = builder->createOptimizationProfile();
    profile1->setDimensions(szInputName, OptProfileSelector::kMIN, Dims4(1, pParam->nChannel, 1, 1));
    profile1->setDimensions(szInputName, OptProfileSelector::kOPT, Dims4(pParam->nMaxBatchSize, pParam->nChannel, 1024, 1024));
    profile1->setDimensions(szInputName, OptProfileSelector::kMAX, Dims4(pParam->nMaxBatchSize, pParam->nChannel, 1024, 1024));
    config->addOptimizationProfile(profile1);

    return builder->buildSerializedNetwork(*network, *config);
}

int main(int argc, char** argv) {
    int iDevice = 0;
    if (argc >= 2) iDevice = atoi(argv[1]);
    ck(cudaSetDevice(iDevice));
    cudaDeviceProp prop = {};
    ck(cudaGetDeviceProperties(&prop, iDevice));
    cout << "Using " << prop.name << endl;

    BuildEngineParam param = {128, 1};
    int nBatch = 128, nHeight = 1024, nWidth = 1024;
    float *hInput = nullptr, *hOutput = nullptr;
    int nFloat = nBatch * param.nChannel * nHeight * nWidth, 
        nByte = nFloat * sizeof(float);
    ck(cudaMallocHost((void **)&hInput, nByte));
    ck(cudaMallocHost((void **)&hOutput, nByte));
    fill(hInput, nFloat, 1.0f);
    
    vector<void *> vdpBuf0(2);
    ck(cudaMalloc(&vdpBuf0[0], nByte));
    ck(cudaMalloc(&vdpBuf0[1], nByte));

    vector<void *> vdpBuf1(2);
    ck(cudaMalloc(&vdpBuf1[0], nByte));
    ck(cudaMalloc(&vdpBuf1[1], nByte));

    cudaStream_t stm0, stm1;
    ck(cudaStreamCreate(&stm0));
    ck(cudaStreamCreate(&stm1));
    
    auto trt = unique_ptr<TrtLite>(TrtLiteCreator::Create(BuildNetworkProc, &param));
    trt->PrintInfo();
    auto trtClone = unique_ptr<TrtLite>(trt->Clone());

    int iScaleFactor[] = {1, 2, 4, 8, 16, 8, 4, 2, 1};
    for (int i = 0; i < sizeof(iScaleFactor) / sizeof(iScaleFactor[0]); i++) {
        int n = nBatch * param.nChannel * nHeight / iScaleFactor[i] * nWidth / iScaleFactor[i];
        ck(cudaMemcpyAsync(vdpBuf0[0], hInput, n, cudaMemcpyHostToDevice, stm0));
        ck(cudaMemcpyAsync(vdpBuf1[0], hInput, n, cudaMemcpyHostToDevice, stm1));
        map<int, Dims> i2shape;
        Dims shape = {4, {nBatch, param.nChannel, nHeight / iScaleFactor[i], nWidth / iScaleFactor[i]}};
        i2shape.insert(make_pair(0, shape));
        trt->Execute(i2shape, vdpBuf0, stm0);
        trtClone->Execute(i2shape, vdpBuf1, stm1);
        cout << "#" << i << ": " << to_string(shape) << endl;
        ck(cudaMemcpyAsync(hOutput, vdpBuf0[1], n, cudaMemcpyDeviceToHost, stm0));
        ck(cudaMemcpyAsync(hOutput, vdpBuf1[1], n, cudaMemcpyDeviceToHost, stm1));
    }

    ck(cudaStreamSynchronize(stm0));
    ck(cudaStreamSynchronize(stm1));
    print(hOutput, nBatch * param.nChannel * nHeight, nWidth, 0, 4, 0, 4);
    
    for (void *dpData: vdpBuf0) {
        ck(cudaFree(dpData));
    }
    for (void *dpData: vdpBuf1) {
        ck(cudaFree(dpData));
    }
    ck(cudaFreeHost(hInput));
    ck(cudaFreeHost(hOutput));
    
    return 0;
}
