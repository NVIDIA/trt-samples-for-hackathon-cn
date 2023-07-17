
#include <NvInfer.h>
#include <array>
#include <iostream>

using namespace nvinfer1;
class Logger : public ILogger
{
public:
    void log(ILogger::Severity severity, const char *msg) noexcept override
    {
        std::cerr << "LOG: " << msg << "\n";
    }
};

static std::array<float, 4> WeightData {1, 2, 3, 4};
int                         main()
{
    Logger           logger;
    constexpr size_t MAX_WSS {1024UL * 1024UL * 1024UL * 2UL};
    auto             builder = createInferBuilder(logger);
    builder->setMaxBatchSize(10);
    NetworkDefinitionCreationFlags flags = 0;
    flags                                = 1U << static_cast<uint32_t>(
                NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

    auto     network = builder->createNetworkV2(flags);
    Dims2    shape(2, 2);
    DataType dt    = DataType::kFLOAT;
    auto     input = network->addInput("input", dt, shape);

    //auto d1 = network->addIdentity(*input);

    auto input2 = network->addInput("input2", dt, shape);
    // Weights weight{.type = dt, .values = WeightData.data(), .count = 4};
    // auto input2 = network->addConstant(shape, weight)->getOutput(0);

    auto bin =
        network->addElementWise(*input, *input2, ElementWiseOperation::kEQUAL)
            ->getOutput(0);
    bin->setType(DataType::kBOOL);
    // auto r = network->addIdentity(*bin->getOutput(0));

    network->markOutput(*bin);
    auto builder_cfg = builder->createBuilderConfig();
    builder_cfg->setMaxWorkspaceSize(MAX_WSS);
    auto engine = builder->buildEngineWithConfig(*network, *builder_cfg);
}

/*
#include <NvInfer.h>
#include <array>
#include <iostream>

using namespace nvinfer1;
class Logger : public ILogger {
 public:
  void log(ILogger::Severity severity, const char* msg) noexcept override {
    std::cerr << "LOG: " << msg << "\n";
  }
};

static std::array<float, 4> WeightData{1, 2, 3, 4};
int main() {
  Logger logger;
  constexpr size_t MAX_WSS{1024UL * 1024UL * 1024UL * 2UL};
  auto builder = createInferBuilder(logger);
  builder->setMaxBatchSize(10);
  NetworkDefinitionCreationFlags flags = 0;
   flags = 1U << static_cast<uint32_t>(
              NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

  auto network = builder->createNetworkV2(flags);
  Dims2 shape(2, 2);
  DataType dt = DataType::kFLOAT;
  auto input = network->addInput("input", dt, shape);

  //auto d1 = network->addIdentity(*input);

  auto input2 = network->addInput("input2", dt, shape);

  //auto d2 = network->addIdentity(*input2);

  // Weights weight{.type = dt, .values = WeightData.data(), .count = 4};
  //auto input2 = network->addConstant(shape, weight)->getOutput(0);

  //auto bin = network->addElementWise(*input, *input2, ElementWiseOperation::kEQUAL)->getOutput(0);
  auto bin = network->addElementWise(*input1, *input2, ElementWiseOperation::kEQUAL)->getOutput(0);
  bin->setType(DataType::kBOOL);
  // auto r = network->addIdentity(*bin->getOutput(0));

  network->markOutput(*bin);
  auto builder_cfg = builder->createBuilderConfig();
  builder_cfg->setMaxWorkspaceSize(MAX_WSS);
  auto engine = builder->buildEngineWithConfig(*network, *builder_cfg);
}
*/
