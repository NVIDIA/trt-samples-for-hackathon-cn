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

#include <vector>
#include <map>
#include <memory>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <string>
#include "Utils.h"

using namespace nvinfer1;
using namespace std;

extern simplelogger::Logger *logger;

class TrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        static simplelogger::LogLevel map[] = {
            simplelogger::FATAL, simplelogger::ERROR, simplelogger::WARNING, simplelogger::INFO, simplelogger::TRACE
        };
        simplelogger::LogTransaction(logger, map[(int)severity], __FILE__, __LINE__, __FUNCTION__).GetStream() << msg;
    }
};

typedef IHostMemory *(*BuildNetworkProcType)(IBuilder *builder, void *pData);
typedef void (*ConfigBuilderProcType)(IBuilderConfig *config, vector<IOptimizationProfile *> vProfile, void *pData);

struct IOInfo {
    string name;
    bool bInput;
    nvinfer1::Dims dim;
    nvinfer1::DataType dataType;

    string GetDimString() {
        return ::to_string(dim);
    }
    string GetDataTypeString() {
        static string aTypeName[] = {"float", "half", "int8", "int32", "bool"};
        return aTypeName[(int)dataType];
    }
    int GetNumBytes() {
        static int aSize[] = {4, 2, 1, 4, 1};
        size_t nSize = aSize[(int)dataType];
        for (int i = 0; i < dim.nbDims; i++) {
            if (dim.d[i] < 0) {
                nSize = -1;
                break;
            }
            nSize *= dim.d[i];
        }
        return nSize;
    }
    string to_string() {
        ostringstream oss;
        oss << setw(6) << (bInput ? "input" : "output") 
            << " | " << setw(5) << GetDataTypeString() 
            << " | " << GetDimString() 
            << " | " << "size=" << GetNumBytes()
            << " | " << name;
        return oss.str();
    }
};

class TrtLite {
public:
    virtual ~TrtLite() {
        if (context) {
            delete context;
        }
    }
    TrtLite *Clone() {
        if (!engine->hasImplicitBatchDimension() && *pnProfile == engine->getNbOptimizationProfiles()) {
            LOG(ERROR) << "Insufficient profiles for creating more contexts";
            return nullptr;
        }
        TrtLite *p = new TrtLite(*this);
        p->iProfile = (*pnProfile)++;
        p->context = nullptr;
        return p;
    }
    ICudaEngine *GetEngine() {
        return engine.get();
    }
    void Execute(int nBatch, vector<void *> &vdpBuf, cudaStream_t stm = 0, cudaEvent_t* evtInputConsumed = nullptr) {
        if (!engine->hasImplicitBatchDimension() && nBatch > 1) {
            LOG(WARNING) << "Engine was built with explicit batch but is executed with batch size != 1. Results may be incorrect.";
            return;
        }
        if (engine->getNbBindings() != vdpBuf.size()) {
            LOG(ERROR) << "Number of bindings conflicts with input and output";
            return;
        }
        if (!context) {
            context = engine->createExecutionContext();
            if (!context) {
                LOG(ERROR) << "createExecutionContext() failed";
                return;
            }
        }
        ck(context->enqueue(nBatch, vdpBuf.data(), stm, evtInputConsumed));
    }
    void Execute(map<int, Dims> i2shape, vector<void *> &vdpBuf, cudaStream_t stm = 0, cudaEvent_t* evtInputConsumed = nullptr) {
        if (engine->hasImplicitBatchDimension()) {
            LOG(ERROR) << "Engine was built with static-shaped input";
            return;
        }
        const int nBuf = engine->getNbBindings() / engine->getNbOptimizationProfiles();
        if (nBuf != vdpBuf.size()) {
            LOG(ERROR) << "Number of bindings conflicts with input and output";
            return;
        }
        if (!context) {
            context = engine->createExecutionContext();
            context->setOptimizationProfileAsync(iProfile, stm);
            if (!context) {
                LOG(ERROR) << "createExecutionContext() failed";
                return;
            }
        }
        // i2shape may have different size with nBuf, because some shapes don't need it or have been set
        for (auto &it : i2shape) {
            context->setBindingDimensions(it.first + nBuf * iProfile, it.second);
        }
        vector<void *> vdpBufEnqueue(engine->getNbBindings());
        std::copy(vdpBuf.begin(), vdpBuf.end(), vdpBufEnqueue.begin() + nBuf * iProfile);
        ck(context->enqueueV2(vdpBufEnqueue.data(), stm, evtInputConsumed));
    }
    void Refit(vector<tuple<const char *, WeightsRole, Weights>> vtWeight) {
        IRefitter *refitter = createInferRefitter(*engine, trtLogger);
        if (!refitter) {
            LOG(ERROR) << "createInferRefitter() failed";
            return;
        }
        for (auto tWeight : vtWeight) {
            if (!refitter->setWeights(get<0>(tWeight), get<1>(tWeight),  get<2>(tWeight))) {
                LOG(ERROR) << "refitter->setWeights() failed for " << get<0>(tWeight);
            }
        }
        if (!refitter->refitCudaEngine()) {
            LOG(ERROR) << "refitter->refitCudaEngine() failed";
        }
        delete refitter;
    }
    void Save(const char *szPath) {
        ofstream of(szPath, ios_base::binary);
        if (!of.good()) {
            LOG(ERROR) << "Failed to open " << szPath;
            return;
        }
        unique_ptr<IHostMemory>m(engine->serialize());
        of.write((char *)m->data(), m->size());
    }

    vector<IOInfo> ConfigIO(int nBatchSize) {
        vector<IOInfo> vInfo;
        if (!engine->hasImplicitBatchDimension()) {
            LOG(ERROR) << "Engine must be built with implicit batch size (and static shape)";
            return vInfo;
        }
        for (int i = 0; i < engine->getNbBindings(); i++) {
            vInfo.push_back({string(engine->getBindingName(i)), engine->bindingIsInput(i), 
                MakeDim(nBatchSize, engine->getBindingDimensions(i)), engine->getBindingDataType(i)});
        }
        return vInfo;
    }
    vector<IOInfo> ConfigIO(map<int, Dims> i2shape) {
        vector<IOInfo> vInfo;
        if (!engine) {
            LOG(ERROR) << "No engine";
            return vInfo;
        }
        if (engine->hasImplicitBatchDimension()) {
            LOG(ERROR) << "Engine must be built with explicit batch size (to enable dynamic shape)";
            return vInfo;
        }
        if (!context) {
            context = engine->createExecutionContext();
            context->setOptimizationProfileAsync(iProfile, 0);
            if (!context) {
                LOG(ERROR) << "createExecutionContext() failed";
                return vInfo;
            }
        }
        const int nBuf = engine->getNbBindings() / engine->getNbOptimizationProfiles();
        for (auto &it : i2shape) {
            context->setBindingDimensions(it.first + nBuf * iProfile, it.second);
        }
        if (!context->allInputDimensionsSpecified()) {
            LOG(ERROR) << "Not all binding shape are specified";
            return vInfo;
        }
        for (int i = nBuf * iProfile; i < nBuf * iProfile + nBuf; i++) {
            vInfo.push_back({string(engine->getBindingName(i)), engine->bindingIsInput(i), 
                context->getBindingDimensions(i), engine->getBindingDataType(i)});
        }
        return vInfo;
    }

    void PrintInfo() {
        cout << "nbBindings: " << engine->getNbBindings() << endl;
        // Only IO information at the engine level is included: -1 for dynamic shape dims
        for (size_t i = 0; i < engine->getNbBindings(); i++) {
            cout << "#" << i << ": " << IOInfo{string(engine->getBindingName(i)), engine->bindingIsInput(i),
                engine->getBindingDimensions(i), engine->getBindingDataType(i)}.to_string() << endl;
        }
        cout << "Refittable: " << (engine->isRefittable() ? "Yes" : "No") << endl;
        if (engine->isRefittable()) {
            cout << "Refittable weights: " << endl;
            IRefitter *refitter = createInferRefitter(*engine, trtLogger);
            if (!refitter) {
                LOG(ERROR) << "createInferRefitter() failed";
                return;
            }
            const int n = refitter->getAll(0, nullptr, nullptr);
            std::vector<const char*> vName(n);
            std::vector<WeightsRole> vRole(n);
            refitter->getAll(n, vName.data(), vRole.data());
            const char *aszRole[] = {"kernel", "bias", "shift", "scale", "constant"};
            for (int i = 0; i < vName.size(); i++) {
                cout << vName[i] << ", " << aszRole[(int)vRole[i]] << endl;
            }
            delete refitter;
        }
    }
    
private:
    TrtLite(TrtLogger trtLogger, ICudaEngine *engine) : trtLogger(trtLogger), 
            engine(shared_ptr<ICudaEngine>(engine)), pnProfile(make_shared<int>()){
        *pnProfile = 1;
    }
    TrtLite(TrtLite const &trt) = default;

    static size_t GetBytesOfBinding(int iBinding, ICudaEngine *engine, IExecutionContext *context = nullptr) {
        size_t aValueSize[] = {4, 2, 1, 4, 1};
        size_t nSize = aValueSize[(int)engine->getBindingDataType(iBinding)];
        const Dims &dims = context ? context->getBindingDimensions(iBinding) : engine->getBindingDimensions(iBinding);
        for (int i = 0; i < dims.nbDims; i++) {
            nSize *= dims.d[i];
        }
        return nSize;
    }
    static nvinfer1::Dims MakeDim(int nBatchSize, nvinfer1::Dims dim) {
        nvinfer1::Dims ret(dim);
        for (int i = ret.nbDims; i > 0; i--) {
            ret.d[i] = ret.d[i - 1];
        }
        ret.d[0] = nBatchSize;
        ret.nbDims++;
        return ret;
    }

    shared_ptr<int> pnProfile;
    shared_ptr<ICudaEngine> engine;
    IExecutionContext *context = nullptr;
    int iProfile = 0;
    /*It seems Builder's logger will be passed to Engine, but there's no API to extract it out.
      Refitter is created from Engine, however, its logger should be passed explicitly but must be the same as Engine's.
      So it's a good idea to keep one logger for all and its life cycle should be at least as longer as Engine's.
      Besides, multi-threading on TRT 7.2 may yields mixed log message.*/
    TrtLogger trtLogger;

    friend class TrtLiteCreator;
};

class TrtLiteCreator {
public:
    static TrtLite* Create(BuildNetworkProcType BuildNetworkProc, void *pData) {
        unique_ptr<IBuilder> builder(createInferBuilder(trtLogger));
        unique_ptr<IHostMemory> plan(BuildNetworkProc(builder.get(), pData));
        unique_ptr<IRuntime> runtime{createInferRuntime(trtLogger)};
        ICudaEngine *engine = runtime->deserializeCudaEngine(plan->data(), plan->size());
        if (!engine) {
            LOG(ERROR) << "No engine created";
            return nullptr;
        }
        return new TrtLite(trtLogger, engine);
    }
    static TrtLite* Create(const char *szEnginePath) {
        BufferedFileReader reader(szEnginePath);
        uint8_t *pBuf = nullptr;
        uint32_t nSize = 0;
        reader.GetBuffer(&pBuf, &nSize);
        if (!nSize) {
            return nullptr;
        }

        IRuntime *runtime = createInferRuntime(trtLogger);
        ICudaEngine *engine = runtime->deserializeCudaEngine(pBuf, nSize);
        delete runtime;        
        if (!engine) {
            LOG(ERROR) << "No engine created";
            return nullptr;
        }
        return new TrtLite(trtLogger, engine);
    }
    static TrtLite* CreateFromOnnx(const char *szOnnxPath, ConfigBuilderProcType ConfigBuilderProc = nullptr, int nProfile = 0, void *pData = nullptr) {
        BufferedFileReader reader(szOnnxPath);
        uint8_t *pBuf = nullptr;
        uint32_t nSize = 0;
        reader.GetBuffer(&pBuf, &nSize);
        if (!nSize) {
            return nullptr;
        }

        unique_ptr<IBuilder> builder(createInferBuilder(trtLogger));
        unique_ptr<INetworkDefinition> network(builder->createNetworkV2(1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));
        nvonnxparser::IParser *parser = nvonnxparser::createParser(*network, trtLogger);
        if (!parser->parse(pBuf, nSize)) {
            return nullptr;
        }

        for (int i = 0; i < network->getNbInputs(); i++) {
            ITensor *tensor = network->getInput(i);
            cout << "#" << i << ": " << IOInfo{string(tensor->getName()), true,
                tensor->getDimensions(), tensor->getType()}.to_string() << endl;
        }

        unique_ptr<IBuilderConfig> config(builder->createBuilderConfig());
        if (ConfigBuilderProc) {
            vector<IOptimizationProfile *> vProfile;
            for (int i = 0; i < nProfile; i++) {
                IOptimizationProfile *profile = builder->createOptimizationProfile();
                vProfile.push_back(profile);
            }
            ConfigBuilderProc(config.get(), vProfile, pData);
        }
        unique_ptr<IHostMemory> plan(builder->buildSerializedNetwork(*network, *config));
        unique_ptr<IRuntime> runtime{createInferRuntime(trtLogger)};
        ICudaEngine *engine = runtime->deserializeCudaEngine(plan->data(), plan->size());

        if (!engine) {
            LOG(ERROR) << "No engine created";
            return nullptr;
        }
        return new TrtLite(trtLogger, engine);
    }
    inline static TrtLogger trtLogger;
};
