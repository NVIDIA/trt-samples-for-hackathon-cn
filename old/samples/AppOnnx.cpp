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

static void ConfigBuilderProc(IBuilderConfig *config, vector<IOptimizationProfile *> vProfile, void *pData) {
    BuildEngineParam *pParam = (BuildEngineParam *)pData;

    if (pParam->bFp16) {
        config->setFlag(BuilderFlag::kFP16);
    }
    
    vProfile[0]->setDimensions("input", OptProfileSelector::kMIN, Dims4(1, pParam->nChannel, pParam->nHeight, pParam->nWidth));
    vProfile[0]->setDimensions("input", OptProfileSelector::kOPT, Dims4(pParam->nMaxBatchSize, pParam->nChannel, pParam->nHeight, pParam->nWidth));
    vProfile[0]->setDimensions("input", OptProfileSelector::kMAX, Dims4(pParam->nMaxBatchSize, pParam->nChannel, pParam->nHeight, pParam->nWidth));
    config->addOptimizationProfile(vProfile[0]);
}

int main(int argc, char** argv) {
    int iDevice = 0;
    if (argc >= 2) iDevice = atoi(argv[1]);
    ck(cudaSetDevice(iDevice));
    cudaDeviceProp prop = {};
    ck(cudaGetDeviceProperties(&prop, iDevice));
    cout << "Using " << prop.name << endl;

    BuildEngineParam param = {1, 3, 128, 128};
    map<int, Dims> i2shape;
    i2shape.insert(make_pair(0, Dims4(param.nMaxBatchSize, param.nChannel, param.nHeight, param.nWidth)));

    int iCase = argc >= 3 ? atoi(argv[2]) : 0;
    auto trt = unique_ptr<TrtLite>(iCase ? TrtLiteCreator::CreateFromOnnx("../python/resnet50.dynamic_shape.onnx", ConfigBuilderProc, 1, &param) 
        : TrtLiteCreator::CreateFromOnnx("../python/resnet50.onnx"));
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
            fill((float *)pBuf, info.GetNumBytes() / sizeof(float), 1.0f);
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

    //print((float *)vpBuf[1], param.nMaxBatchSize * param.nChannel * param.nHeight, param.nWidth);    
    
    for (int i = 0; i < vInfo.size(); i++) {
        delete[] (uint8_t *)vpBuf[i];
        ck(cudaFree(vdpBuf[i]));
    }

    return 0;
}
