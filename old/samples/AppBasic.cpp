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

IHostMemory *BuildNetworkProc(IBuilder *builder, void *pData) {
    BuildEngineParam *pParam = (BuildEngineParam *)pData;
    unique_ptr<INetworkDefinition> network(builder->createNetworkV2(0));
    ITensor *tensor = network->addInput("input0", DataType::kFLOAT, Dims3{pParam->nChannel, pParam->nHeight, pParam->nWidth});
    float kernel[] = {
        0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
    };
    float bias[] = {
        0.0f,
    };
    for (int i = 0; i < 2; i++) {
        IConvolutionLayer *conv = network->addConvolutionNd(*tensor, pParam->nChannel, DimsHW(3, 3), 
            Weights{DataType::kFLOAT, kernel, sizeof(kernel) / sizeof(kernel[0])}, 
            Weights{DataType::kFLOAT, bias, sizeof(bias) / sizeof(bias[0])}
        );
        conv->setPaddingMode(PaddingMode::kSAME_LOWER);
        tensor = conv->getOutput(0);
    }
    network->markOutput(*tensor);

    unique_ptr<IBuilderConfig> config(builder->createBuilderConfig());
    if (pParam->bFp16) {
        config->setFlag(BuilderFlag::kFP16);
    }
    if (pParam->bRefit) {
        config->setFlag(BuilderFlag::kREFIT);
    }
    builder->setMaxBatchSize(pParam->nMaxBatchSize);

    return builder->buildSerializedNetwork(*network, *config);
}

int main(int argc, char** argv) {
    int iDevice = 0;
    if (argc >= 2) iDevice = atoi(argv[1]);
    ck(cudaSetDevice(iDevice));
    cudaDeviceProp prop = {};
    ck(cudaGetDeviceProperties(&prop, iDevice));
    cout << "Using " << prop.name << endl;

    BuildEngineParam param = {16, 1, 2, 8};
    param.bFp16 = true;
    param.bRefit = true;
    int nBatch = 2;

    auto trt = unique_ptr<TrtLite>(TrtLiteCreator::Create(BuildNetworkProc, &param));
    trt->PrintInfo();
    
    vector<void *> vpBuf, vdpBuf;
    vector<IOInfo> vInfo = trt->ConfigIO(nBatch);
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
    trt->Execute(nBatch, vdpBuf);
    for (int i = 0; i < vInfo.size(); i++) {
        auto &info = vInfo[i];
        if (info.bInput) {
            continue;
        }
        ck(cudaMemcpy(vpBuf[i], vdpBuf[i], info.GetNumBytes(), cudaMemcpyDeviceToHost));
    }

    print((float *)vpBuf[1], nBatch * param.nChannel * param.nHeight, param.nWidth);    
    const char *szPath = "out.trt";
    trt->Save(szPath);
    
    for (int i = 0; i < vInfo.size(); i++) {
        delete[] (uint8_t *)vpBuf[i];
        ck(cudaFree(vdpBuf[i]));
    }
    
    return 0;
}
