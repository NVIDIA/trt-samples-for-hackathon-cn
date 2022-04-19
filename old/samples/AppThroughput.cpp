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

#include <iostream>
#include <thread>
#include <sstream>
#include <dlfcn.h>
#include <cuda_fp16.h>
#include "TrtLite.h"
#include "Utils.h"

simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger(simplelogger::TRACE);

void ThreadProc(const char *szEnginePath, bool bCreateCudaContext, int iThread, const int nThread, const int nStream, const int nRound, 
        char &cReady, const bool &bGo) {
    if (bCreateCudaContext) {
    // may use with MPS
        CUdevice dev;
        ck(cuDeviceGet(&dev, 0));
        CUcontext ctx;
        ck(cuCtxCreate(&ctx, 0, dev));
    }

    if (nStream <= 0) {
        cout << "Error: nStream = " << nStream << endl;
        return;
    }

    vector<cudaStream_t> vStm(nStream);
    vector<TrtLite *> vTrt(nStream);
    vector<tuple<vector<void *>, vector<void *>>> vDualBuf(nStream);
    
    vector<IOInfo> vInfo;
    map<int, Dims> i2shape;
    const int nBatchForImplicit = 1;
    vTrt[0] = TrtLiteCreator::Create(szEnginePath);
    ICudaEngine *engine = vTrt[0]->GetEngine();
    if (engine->hasImplicitBatchDimension()) {
        vInfo = vTrt[0]->ConfigIO(nBatchForImplicit);
    } else {
        for (int i = 0; i < engine->getNbBindings() / engine->getNbOptimizationProfiles() && engine->bindingIsInput(i); i++) {
            i2shape[i] = engine->getProfileDimensions(i, 0, OptProfileSelector::kOPT);
        }
        vInfo = vTrt[0]->ConfigIO(i2shape);
    }

    for (int i = 0; i < nStream; i++) {
        ck(cudaStreamCreate(&vStm[i]));
        if (i > 0) {
            vTrt[i] = TrtLiteCreator::Create(szEnginePath);
        }

        vector<void *> vpBuf, vdpBuf;
        for (auto info : vInfo) {
            void *pBuf = nullptr;
            ck(cudaMallocHost(&pBuf, info.GetNumBytes()));
            vpBuf.push_back(pBuf);

            void *dpBuf = nullptr;
            ck(cudaMalloc(&dpBuf, info.GetNumBytes()));
            vdpBuf.push_back(dpBuf);
        }
        vDualBuf[i] = {vpBuf, vdpBuf};
    }

    cReady = true;
    while (!bGo) {
        std::this_thread::sleep_for(1ms);
    }

    for (int k = iThread, iStream = 0; k < nRound; k += nThread, iStream = (iStream + 1) % nStream) {
        vector<void *> vpBuf, vdpBuf;
        tie(vpBuf, vdpBuf) = vDualBuf[iStream];
        
        for (int i = 0; i < vInfo.size(); i++) {
            auto &info = vInfo[i];
            if (info.bInput) {
                ck(cudaMemcpyAsync(vdpBuf[i], vpBuf[i], info.GetNumBytes(), cudaMemcpyHostToDevice, vStm[iStream]));
            }
        }

        if (vTrt[iStream]->GetEngine()->hasImplicitBatchDimension()) {
            vTrt[iStream]->Execute(nBatchForImplicit, vdpBuf);
        } else {
            vTrt[iStream]->Execute(i2shape, vdpBuf);
        }

        for (int i = 0; i < vInfo.size(); i++) {
            auto &info = vInfo[i];
            if (!info.bInput) {
                ck(cudaMemcpyAsync(vpBuf[i], vdpBuf[i], info.GetNumBytes(), cudaMemcpyDeviceToHost, vStm[iStream]));
            }
        }
    }
    for (int i = 0; i < nStream; i++) {
        ck(cudaStreamSynchronize(vStm[i]));
        ck(cudaStreamDestroy(vStm[i]));
        delete vTrt[i];
        for (int j = 0; j < vInfo.size(); j++) {
            ck(cudaFreeHost(get<0>(vDualBuf[i])[j]));
            ck(cudaFree(get<1>(vDualBuf[i])[j]));
        }
    }
}

int main(int argc, char** argv) {
    cudaDeviceProp prop = {};
    ck(cudaGetDeviceProperties(&prop, 0));
    cout << "Using " << prop.name << endl;

    int nThread = 1, nStream = 1, nRound = 200;
    bool bCreateCudaContext = false;
    const char *szEnginePath = "../python/resnet50.trt";
    if (argc >= 2) szEnginePath = argv[1];
    if (argc >= 3 && atoi(argv[2]) >= 1) nThread = atoi(argv[2]);
    if (argc >= 4 && atoi(argv[3]) >= 1) nStream = atoi(argv[3]);
    if (argc >= 5 && atoi(argv[4]) >= 1) nRound = atoi(argv[4]);
    if (argc >= 6) bCreateCudaContext = atoi(argv[5]);
    cout << "thread (engine): " << nThread << ", stream per thread: " << nStream 
        << ", round: " << nRound << ", standalone CUDA context: " << (bCreateCudaContext ? "yes" : "no") << endl;

    vector<thread *>vpThread(nThread);
    vector<char> vcReady(nThread);
    bool bGo = false;
    
    for (int iThread = 0; iThread < nThread; iThread++) {
        vpThread[iThread] = new thread(ThreadProc, szEnginePath, bCreateCudaContext, iThread, nThread, nStream, nRound, ref(vcReady[iThread]), ref(bGo));
    }

    while (true) {
        bool bAllReady = true;
        for (int i = 0; i < nThread; i++) {
            if (!vcReady[i]) {
                bAllReady = false;
                break;
            }
        }
        if (bAllReady) {
            bGo = true;
            break;
        }
        std::this_thread::sleep_for(1ms);
    }

    StopWatch w;
    w.Start();
    for (int i = 0; i < nThread; i++) {
        vpThread[i]->join();
    }
    double t = w.Stop();
    cout << endl << "Time: " << t << "s, QPS=" << nRound / t << endl;

    for (int i = 0; i < nThread; i++) { 
        delete vpThread[i];
    }

    return 0;
}
