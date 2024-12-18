SHELL           = /bin/bash

# Set path
CUDA_PATH       = /usr/local/cuda
#TRT_INC_PATH    = /usr/include/x86_64-linux-gnu
#TRT_LIB_PATH    = /usr/lib/x86_64-linux-gnu
TRT_INC_PATH    = /usr/local/tensorrt/include
TRT_LIB_PATH    = /usr/local/tensorrt/lib

# Choose GPU arhitecture
GENCODE         = -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_89,code=sm_89 -gencode=arch=compute_90,code=sm_90
GENCODE         = -gencode=arch=compute_89,code=sm_89

# Enable Debug or not
DEBUG_MACRO     = -DDEBUG
DEBUG_MACRO     = -UDEBUG

NVCC            = $(CUDA_PATH)/bin/nvcc
WARNING_MACRO   = -w
CCFLAG          = -std=c++17 -O3 -Xcompiler -fPIC -use_fast_math $(GENCODE)
SOFLAG          = -shared
INCLUDE         = -I. -I$(CUDA_PATH)/include -I$(TRT_INC_PATH) -I$(TRT_COOKBOOK_PATH)/include
LDFLAG          = -L$(CUDA_PATH)/lib64 -lcudart -L$(TRT_LIB_PATH) -lnvinfer

COMMAND_O 		= $(NVCC) $(CCFLAG) $(INCLUDE) $(DEBUG_MACRO) -MMD -MF $@.d -o $@ -c $<
COMMAND_EXE     = $(NVCC) $(LDFLAG) -o $@ $^
COMMAND_SO      = $(NVCC) $(LDFLAG) $(SOFLAG) -o $@ $^

-include $(shell find . -name '*.d')

cookbookHelper.o: $(TRT_COOKBOOK_PATH)/include/cookbookHelper.cu
	$(COMMAND_O)

calibrator.o: $(TRT_COOKBOOK_PATH)/include/calibrator.cpp
	$(COMMAND_O)

cnpy.o: $(TRT_COOKBOOK_PATH)/include/cnpy.cpp
	$(COMMAND_O)
