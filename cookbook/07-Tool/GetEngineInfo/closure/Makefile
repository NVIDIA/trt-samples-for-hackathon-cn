CUDA_PATH       = /usr/local/cuda
NVCC            = $(CUDA_PATH)/bin/nvcc
TRT_INC_PATH    = /usr/include/x86_64-linux-gnu
TRT_LIB_PATH    = /usr/lib/x86_64-linux-gnu
GENCODE         = -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_89,code=sm_89 -gencode=arch=compute_90,code=sm_90
DEBUG_MACRO     = -UDEBUG
WARNING_MACRO   = -w
CUFLAG          = -std=c++17 -O3 $(DEBUG_MACRO) -Xcompiler -fPIC $(GENCODE)
CCFLAG          = -std=c++17 -O3 $(DEBUG_MACRO) -Xcompiler -fPIC -use_fast_math
SOFLAG          = -shared
INCLUDE         = -I. -I$(CUDA_PATH)/include -I$(TRT_INC_PATH)
LDFLAG          = -L$(CUDA_PATH)/lib64 -lcudart -L$(TRT_LIB_PATH) -lnvinfer

SOURCE_CPP  = $(shell find . -name '*.cpp' 2>/dev/null)
OBJ         = $(shell find . -name *.o 2>/dev/null)
DEP         = $(OBJ:.o=.d)
TARGET_EXE  = $(SOURCE_CPP:.cpp=.exe)

-include $(DEP)

all: $(TARGET_EXE)

%.exe: %.o
	$(NVCC) $(CCFLAG) $(LDFLAG) -o $@ $+

%.o: %.cpp
	$(NVCC) $(CCFLAG) $(INCLUDE) -M -MT $@ -o $(@:.o=.d) $<
	$(NVCC) $(CCFLAG) $(INCLUDE) -Xcompiler -fPIC -o $@ -c $<

.PHONY: test
test:
	make clean
	make
	$(TARGET_EXE) > result.log

.PHONY: clean
clean:
	rm -rf ./*.d ./*.o ./*.so ./*.exe ./*.trt
