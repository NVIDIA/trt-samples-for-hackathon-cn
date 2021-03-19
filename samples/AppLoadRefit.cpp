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

#include <dlfcn.h>
#include <cuda_fp16.h>
#include "TrtLite.h"
#include "Utils.h"

simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger(simplelogger::TRACE);

void DoRefit(TrtLite *trt) {
    float kernel[] = {
        0.0f, 0.0f, 0.0f,
        0.0f, -1.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
    };
    float bias[] = {
        0.0f,
    };
    vector<tuple<const char *, WeightsRole, Weights>> tab = {
        make_tuple("(Unnamed Layer* 0) [Convolution]", WeightsRole::kKERNEL, Weights{DataType::kFLOAT, kernel, sizeof(kernel) / sizeof(kernel[0])}),
        make_tuple("(Unnamed Layer* 0) [Convolution]", WeightsRole::kBIAS, Weights{DataType::kFLOAT, bias, sizeof(bias) / sizeof(bias[0])}),
    };
    trt->Refit(tab);
    
    /*
     // For fp16 only. Ensure bForceOneFp16Layer == true
     make && ./AppBasic &> build.log
     grep -E '\[Fully Connected\]|\[Convolution\]|\[Constant\]' build.log | grep Float | grep -Ev 'Layer\(Reformat\)|Layer\(Shuffle\)|Adding reformat layer' > fp32layer.txt
     grep -E '\[Fully Connected\]|\[Convolution\]|\[Constant\]' build.log | grep Half | grep -Ev 'Layer\(Reformat\)|Layer\(Shuffle\)|Adding reformat layer' > fp16layer.txt
     */
    BufferedFileReader fp32reader("fp32layer.txt");
    const char *szFp32Layer = nullptr;
    fp32reader.GetBuffer((uint8_t **)&szFp32Layer, nullptr);
    BufferedFileReader fp16reader("fp16layer.txt");
    const char *szFp16Layer = nullptr;
    fp16reader.GetBuffer((uint8_t **)&szFp16Layer, nullptr);
    vector<half *> vp;
    vector<tuple<const char *, WeightsRole, Weights>> tabFp16;
    if (szFp32Layer && szFp16Layer) {
        for (auto t : tab) {
            cout << get<0>(t) << " ";
            if (strstr(szFp32Layer, get<0>(t)) && !strstr(szFp16Layer, get<0>(t))) {
                cout << "fp32";
            } else if (!strstr(szFp32Layer, get<0>(t)) && strstr(szFp16Layer, get<0>(t))) {
                cout << "fp16";
            } else if (strstr(szFp32Layer, get<0>(t)) && strstr(szFp16Layer, get<0>(t))) {
                cout << "both";
            } else {
                cout << "null";
            }
            cout << endl;
            
            Weights &w = get<2>(t);
            if (w.type == DataType::kFLOAT && !strstr(szFp32Layer, get<0>(t)) && strstr(szFp16Layer, get<0>(t))) {
                __half *p = new __half[w.count];
                vp.push_back(p);
                for (int i = 0; i < w.count; i++) {
                    p[i] = __half(((float *)w.values)[i]);
                }
                w.type = DataType::kHALF;
                w.values = p;
                cout << "Convert to fp16: " << get<0>(t) << endl;
            }
            tabFp16.push_back(t);
        }
        trt->Refit(tabFp16);
    }
}

int main(int argc, char** argv) {    
    int iDevice = 0;
    if (argc >= 2) iDevice = atoi(argv[1]);
    ck(cudaSetDevice(iDevice));
    cudaDeviceProp prop = {};
    ck(cudaGetDeviceProperties(&prop, iDevice));
    cout << "Using " << prop.name << endl;

    dlopen("./AppPlugin", RTLD_LAZY);
    // you have to adjust (n,c,h,w) to something like (2,4,1,8) load the engine saved by AppPluginDynamicShape
    //dlopen("./AppPluginDynamicShape", RTLD_LAZY);

    auto trt = unique_ptr<TrtLite>(TrtLiteCreator::Create("out.trt"));
    trt->PrintInfo();

    if (trt->GetEngine()->isRefittable()) {
        LOG(INFO) << "Engine is refittable. Refitting...";
        DoRefit(trt.get());
    } else {
        LOG(INFO) << "Engine isn't refittable. Refit is skipped.";
    }
    
    const int nBatch = 2, nChannel = 1, nHeight = 2, nWidth = 8;
    map<int, Dims> i2shape;
    i2shape.insert(make_pair(0, Dims{4, {nBatch, nChannel, nHeight, nWidth}}));

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

    print((float *)vpBuf[1], nBatch * nChannel * nHeight, nWidth);    
    
    for (int i = 0; i < vInfo.size(); i++) {
        delete[] (uint8_t *)vpBuf[i];
        ck(cudaFree(vdpBuf[i]));
    }

    return 0;
}
