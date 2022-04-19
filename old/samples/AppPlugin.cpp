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
#include "../plugins/AddPlugin.h"

simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger(simplelogger::TRACE);

struct BuildEngineParamExt : BuildEngineParam {
    int nBatchSize;
};

IHostMemory *BuildNetworkProc(IBuilder *builder, void *pData) {
    BuildEngineParamExt *pParam = (BuildEngineParamExt *)pData;
    bool bExplicitBatch = pParam->nBatchSize != 0;
    unique_ptr<INetworkDefinition> network(builder->createNetworkV2(bExplicitBatch ? 1 : 0));
    ITensor *tensor;
    if (bExplicitBatch) {
        tensor = network->addInput("input0", DataType::kFLOAT, Dims4{pParam->nBatchSize, pParam->nChannel, pParam->nHeight, pParam->nWidth});
    } else {
        tensor = network->addInput("input0", DataType::kFLOAT, Dims3{pParam->nChannel, pParam->nHeight, pParam->nWidth});
    }
    float valueToAdd = 0.1f;
    AddPlugin addPlugin(Weights{DataType::kFLOAT, &valueToAdd, 1});
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
    Calibrator calib(1, pParam, "int8_cache.AppPlugin");
    if (pParam->bInt8) {
        config->setFlag(BuilderFlag::kINT8);
        config->setInt8Calibrator(&calib);
    }
    if (pParam->bRefit) {
        config->setFlag(BuilderFlag::kREFIT);
    }
    if (!bExplicitBatch) {
        builder->setMaxBatchSize(pParam->nMaxBatchSize);
    }

    return builder->buildSerializedNetwork(*network, *config);
}

int main(int argc, char** argv) {
    int iDevice = 0;
    if (argc >= 2) iDevice = atoi(argv[1]);
    ck(cudaSetDevice(iDevice));
    cudaDeviceProp prop = {};
    ck(cudaGetDeviceProperties(&prop, iDevice));
    cout << "Using " << prop.name << endl;

    bool bExplitBatch = false;
    int nBatch = 2;
    // to use int8, dim C must be multiple of 4
    BuildEngineParamExt param = {{16, 4, 1, 8}, bExplitBatch ? nBatch : 0};
    param.bInt8 = !(argc >= 3 && atoi(argv[2]) == 0);
    param.bFp16 = !(argc >= 4 && atoi(argv[3]) == 0);
    map<int, Dims> i2shape;
    i2shape.insert(make_pair(0, Dims4(nBatch, param.nChannel, param.nHeight, param.nWidth)));

    auto trt = unique_ptr<TrtLite>(TrtLiteCreator::Create(BuildNetworkProc, &param));
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
    trt->Save("out.trt");
    
    return 0;
}
