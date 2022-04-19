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
    IOptimizationProfile *profile = builder->createOptimizationProfile();
    profile->setDimensions(szInputName, OptProfileSelector::kMIN, Dims4(1, pParam->nChannel, 1, 1));
    profile->setDimensions(szInputName, OptProfileSelector::kOPT, Dims4(pParam->nMaxBatchSize, pParam->nChannel, 1024, 1024));
    profile->setDimensions(szInputName, OptProfileSelector::kMAX, Dims4(pParam->nMaxBatchSize, pParam->nChannel, 1024, 1024));
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

    BuildEngineParam param = {128, 1};
    int nBatch = 128, nHeight = 1024, nWidth = 1024;
    float *hInput = nullptr, *hOutput = nullptr;
    int nFloat = nBatch * param.nChannel * nHeight * nWidth, 
        nByte = nFloat * sizeof(float);
    ck(cudaMallocHost((void **)&hInput, nByte));
    ck(cudaMallocHost((void **)&hOutput, nByte));
    fill(hInput, nFloat, 1.0f);
    
    vector<void *> vdpBuf(2);
    ck(cudaMalloc(&vdpBuf[0], nByte));
    ck(cudaMalloc(&vdpBuf[1], nByte));

    cudaStream_t stm;
    ck(cudaStreamCreate(&stm));
    
    auto trt = unique_ptr<TrtLite>(TrtLiteCreator::Create(BuildNetworkProc, &param));
    int iScaleFactor[] = {1, 2, 4, 8, 16, 8, 4, 2, 1};
    for (int i = 0; i < sizeof(iScaleFactor) / sizeof(iScaleFactor[0]); i++) {
        int n = nBatch * param.nChannel * nHeight / iScaleFactor[i] * nWidth / iScaleFactor[i];
        ck(cudaMemcpyAsync(vdpBuf[0], hInput, n, cudaMemcpyHostToDevice, stm));
        map<int, Dims> i2shape;
        Dims shape = {4, {nBatch, param.nChannel, nHeight / iScaleFactor[i], nWidth / iScaleFactor[i]}};
        i2shape.insert(make_pair(0, shape));
        trt->Execute(i2shape, vdpBuf, stm);
        cout << "#" << i << ": " << to_string(shape) << endl;
        ck(cudaMemcpyAsync(hOutput, vdpBuf[1], n, cudaMemcpyDeviceToHost, stm));
    }
    
    ck(cudaStreamSynchronize(stm));
    print(hOutput, nBatch * param.nChannel * nHeight, nWidth, 0, 4, 0, 4);
    
    for (void *dpData: vdpBuf) {
        ck(cudaFree(dpData));
    }
    ck(cudaFreeHost(hInput));
    ck(cudaFreeHost(hOutput));
    
    return 0;
}
