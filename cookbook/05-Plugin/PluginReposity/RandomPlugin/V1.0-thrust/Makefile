CUDA_PATH       = /usr/local/cuda
TRT_PATH        = /usr/local/tensorrt8/
NVCC            = $(CUDA_PATH)/bin/nvcc
SM              = 86


CUFLAG          = -w -std=c++14 -O3 -gencode=arch=compute_$(SM),code=sm_$(SM) -DNDEBUG -DDEBUG_ENABLE=0 -DUSE_FP_6=$(USE_FP16) -Xcompiler -fPIC
CPPFLAG         = -w -std=c++14 -O3 -gencode=arch=compute_$(SM),code=sm_$(SM) -use_fast_math
SOFLAG          = $(CUFLAG) -shared
INCLUDE         = -I. -I$(CUDA_PATH)/include -I$(TRT_PATH)/include
LDFLAG          = -L$(CUDA_PATH)/lib64 -lcudart -lcublas -lcublasLt -lcurand -L$(TRT_PATH)/lib -lnvinfer -lz

SRC_CU          = $(shell find ./ -name '*.cu')

all: RandomPlugin.so

%.o: %.cu
	$(NVCC) $(CUFLAG) $(INCLUDE) -o $@ -c $<

RandomPlugin.so: $(SRC_CU:.cu=.o)
	$(NVCC) $(SOFLAG) $(LDFLAG) -o $@ $^

.PHONY: clean
clean:
	rm -rf ./*.so ./*.o ./*.d ./*.trt
