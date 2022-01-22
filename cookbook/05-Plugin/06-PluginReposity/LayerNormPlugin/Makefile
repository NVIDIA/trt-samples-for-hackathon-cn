CUDA_PATH       = /usr/local/cuda
TRT_PATH        = /usr/lib/x86_64-linux-gnu
NVCC            = $(CUDA_PATH)/bin/nvcc
SM              = 61
USE_FP16        = 0
CUFLAG          = -w -std=c++14 -O3 -gencode=arch=compute_$(SM),code=sm_$(SM) -DNDEBUG -DDEBUG_ENABLE=0 -DUSE_FP_6=$(USE_FP16) -Xcompiler -fPIC
CPPFLAG         = -w -std=c++14 -O3 -gencode=arch=compute_$(SM),code=sm_$(SM) -use_fast_math
SOFLAG          = $(CUFLAG) -shared
INCLUDE         = -I. -I$(CUDA_PATH)/include
LDFLAG          = -L$(CUDA_PATH)/lib64 -lcudart -lcublas -lcublasLt -L$(TRT_PATH)/lib -lnvinfer -lz
SRC_CU          = $(shell find ./ -name '*.cu')

all: LayerNorm.so

%.o: %.cu
	$(NVCC) $(CUFLAG) $(INCLUDE) -o $@ -c $<

LayerNorm.so: $(SRC_CU:.cu=.o)
	$(NVCC) $(SOFLAG) $(LDFLAG) -o $@ $^

.PHONY: clean
clean:
	rm -rf ./*.so ./*.o ./*.d ./*.trt

.PHONY: t
t:
	clear
	python testLayerNormPlugin.py

