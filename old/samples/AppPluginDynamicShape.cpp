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
#include "../plugins/AddPluginDyn.h"

simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger(simplelogger::TRACE);

int nBatch = 1;

IHostMemory *BuildNetworkProc(IBuilder *builder, void *pData) {
    BuildEngineParam *pParam = (BuildEngineParam *)pData;
    
    pParam->nMaxWorkspaceSize = (unsigned int)4<<30;
    
    unique_ptr<INetworkDefinition> network(builder->createNetworkV2(1));
    const char *szInputName = "input0";
    ITensor *tensor = network->addInput(szInputName, DataType::kFLOAT, Dims4{-1, pParam->nChannel, -1, -1});
    float valueToAdd = 0.1f;
    AddPluginDyn addPlugin(Weights{DataType::kFLOAT, &valueToAdd, 1});
    ITensor *aInputTensor[] = {tensor};
    tensor = network->addPluginV2(aInputTensor, 1, addPlugin)->getOutput(0);
    if (pParam->bInt8) {
        // to output int8 from the plugin, it must be set explicitly, otherwise it won't be selected
        tensor->setType(DataType::kINT8);
        // convert to float (if not, output type is int8 and you need to deal with the buffer accordingly)
        tensor = network->addIdentity(*tensor)->getOutput(0);
        tensor->setType(DataType::kFLOAT);
    }
    network->markOutput(*tensor);

    unique_ptr<IBuilderConfig> config(builder->createBuilderConfig());
    if (pParam->bFp16) {
        config->setFlag(BuilderFlag::kFP16);
    }
    BuildEngineParam optParam = {pParam->nMaxBatchSize, pParam->nChannel, 1024, 1024};
    Calibrator calib(16, &optParam, "int8_cache.AppPluginDynamicShape");
    if (pParam->bInt8) {
        config->setFlag(BuilderFlag::kINT8);
        config->setInt8Calibrator(&calib);
    }

    IOptimizationProfile *profile = builder->createOptimizationProfile();
    profile->setDimensions(szInputName, OptProfileSelector::kMIN, Dims4(1, pParam->nChannel, 1, 1));
    profile->setDimensions(szInputName, OptProfileSelector::kOPT, Dims4(optParam.nMaxBatchSize, optParam.nChannel, optParam.nHeight, optParam.nWidth));
    profile->setDimensions(szInputName, OptProfileSelector::kMAX, Dims4(optParam.nMaxBatchSize, optParam.nChannel, optParam.nHeight, optParam.nWidth));
    config->addOptimizationProfile(profile);

    return builder->buildSerializedNetwork(*network, *config);
}

int main(int argc, char** argv) {
    ck(cudaSetDevice(0));

    int nBatchSize = 1, nChannel = 4, nHeight = 1, nWidth = 8;
    BuildEngineParam param = {16, 4};
    param.bInt8 = true;//!(argc >= 3 && atoi(argv[2]) == 0);
    param.bFp16 = false;//!(argc >= 4 && atoi(argv[3]) == 0);
    map<int, Dims> i2shape;
    i2shape.insert(make_pair(0, Dims4(1,4,1,8)));

    auto trt = unique_ptr<TrtLite>(TrtLiteCreator::Create(BuildNetworkProc, &param));
        
    trt->PrintInfo();
    vector<void *> vpBuf, vdpBuf;
    vector<IOInfo> vInfo;
    vInfo = trt->ConfigIO(i2shape);
    for (auto info : vInfo) {
        cout << info.to_string() << endl;
        
        void *pBuf = nullptr;
        pBuf = new uint8_t[info.GetNumBytes()];
        vpBuf.push_back(pBuf);

        void *dpBuf = nullptr;
        ck(cudaMalloc(&dpBuf, info.GetNumBytes()));
        vdpBuf.push_back(dpBuf);

        if (info.bInput) {
            //fill((float *)pBuf, info.GetNumBytes() / sizeof(float), 1.0f);
            float * in = (float *)pBuf;
            for (int i=0; i < info.GetNumBytes() / sizeof(float); ++i)
                in[i] = -1.0f + (float)i/32*2;
            ck(cudaMemcpy(dpBuf, pBuf, info.GetNumBytes(), cudaMemcpyHostToDevice));
        }
    }
    trt->Execute(i2shape, vdpBuf);
    for (int i = 0; i < vInfo.size(); i++) {
        auto &info = vInfo[i];
        if (info.bInput) {
            continue;
        }
        ck(cudaMemcpy(vpBuf[i], vdpBuf[i], info.GetNumBytes(), cudaMemcpyDeviceToHost));
    }

    print((float *)vpBuf[0], vInfo[1].dim.d[0] * vInfo[1].dim.d[1] * vInfo[1].dim.d[2], 8);
    print((float *)vpBuf[1], vInfo[1].dim.d[0] * vInfo[1].dim.d[1] * vInfo[1].dim.d[2], 8);
    
    for (int i = 0; i < vInfo.size(); i++) {
        delete[] (uint8_t *)vpBuf[i];
        ck(cudaFree(vdpBuf[i]));
    }
    // trt->Save("out.trt");

    return 0;
}
