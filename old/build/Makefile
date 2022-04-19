SHELL=/bin/bash
CUDA_PATH = /usr/local/cuda
TRT_PATH = /usr/local/tensorrt8
GENCODE = -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_86,code=compute_86

GCC = g++
NVCC = $(CUDA_PATH)/bin/nvcc
CCFLAGS = -g -DNDEBUG -std=c++17
INCLUDES := -I../include -isystem $(CUDA_PATH)/include -isystem $(TRT_PATH)/include
LDFLAGS := -L$(CUDA_PATH)/lib64 -L$(TRT_PATH)/lib -Wl,-rpath=$(CUDA_PATH)/lib64 -L/usr/lib/wsl/lib
LDFLAGS += -lnvinfer -lnvonnxparser -lcudart -ldl -lpthread -lcuda

OBJ_DIR = obj
OBJ = $(shell find $(OBJ_DIR) -name *.o 2>/dev/null)
DEP = $(OBJ:.o=.d)

BIN = AppBasic AppLoadRefit AppDynamicShape AppInt8 AppPlugin AppPluginDynamicShape AppMultiContext AppThroughput AppOnnx AppShapeLayer
SO = AddPlugin.so GridSamplerPlugin.so ScatterND.so GatherND.so OnehotPlugin.so AddPluginDyn.so

all: $(BIN) $(SO)

AppBasic: $(addprefix $(OBJ_DIR)/, AppBasic.o)
AppLoadRefit: $(addprefix $(OBJ_DIR)/, AppLoadRefit.o)
AppDynamicShape: $(addprefix $(OBJ_DIR)/, AppDynamicShape.o)
AppInt8: $(addprefix $(OBJ_DIR)/, AppInt8.o)
AppPlugin: $(addprefix $(OBJ_DIR)/, AppPlugin.o AddPlugin.o)
AppPluginDynamicShape: $(addprefix $(OBJ_DIR)/, AppPluginDynamicShape.o AddPluginDyn.o)
AppMultiContext: $(addprefix $(OBJ_DIR)/, AppMultiContext.o)
AppThroughput: $(addprefix $(OBJ_DIR)/, AppThroughput.o)
AppOnnx: $(addprefix $(OBJ_DIR)/, AppOnnx.o)
AppShapeLayer: $(addprefix $(OBJ_DIR)/, AppShapeLayer.o)

AddPlugin.so: $(addprefix $(OBJ_DIR)/, AddPlugin.o)
AddPluginDyn.so: $(addprefix $(OBJ_DIR)/, AddPluginDyn.o)
GridSamplerPlugin.so: $(addprefix $(OBJ_DIR)/, GridSamplerPlugin.o)
ScatterND.so: $(addprefix $(OBJ_DIR)/, ScatterND.o)
GatherND.so: $(addprefix $(OBJ_DIR)/, GatherND.o)
OnehotPlugin.so: $(addprefix $(OBJ_DIR)/, OnehotPlugin.o)

-include $(DEP)

clean:
	rm -rf $(BIN) $(SO) $(DEP) $(OBJ_DIR) out.trt int8_cache.*
	rm -rf ../python/resnet50.{onnx,trt} ../python/resnet50.dynamic_shape.{onnx,trt} ../python/{custom.onnx,custom_surgeon.onnx} ../python/out.trt ../python/__pycache__

VPATH = ../include:../samples:../plugins

$(OBJ_DIR)/%.o: %.cpp
	@mkdir -p $(@D)
	$(GCC) $(CCFLAGS) -fPIC -MD -MP $(INCLUDES) -o $@ -c $<

$(OBJ_DIR)/%.o: %.cu
	@mkdir -p $(@D)
	$(NVCC) $(CCFLAGS) -M -MT $@ $(INCLUDES) -o $(@:.o=.d) $<
	$(NVCC) $(CCFLAGS) $(INCLUDES) -Xcompiler -fPIC $(GENCODE) -o $@ -c $<

$(SO):
	$(GCC) $(CCFLAGS) -shared -o $@ $+ $(LDFLAGS)

$(BIN):
	$(GCC) $(CCFLAGS) -o $@ $+ $(LDFLAGS)

$(CUDA_BIN):
	$(NVCC) $(NVCCFLAGS) -o $@ $+ $(LDFLAGS)
