CUDA_PATH       = /usr/local/cuda
#TRT_PATH        = /usr/lib/x86_64-linux-gnu
TRT_PATH        = /usr/local/tensorrt8.0-cuda11.3
NVCC            = $(CUDA_PATH)/bin/nvcc
SM              = 61
                # 61 for GTX1070, 75 for T4,80 for A30
CUFLAG          = -w -std=c++14 -O3 -gencode=arch=compute_$(SM),code=sm_$(SM) -DNDEBUG -DDEBUG_ENABLE=0 -DUSE_FP_6=$(USE_FP16) -Xcompiler -fPIC
CPPFLAG         = -w -std=c++14 -O3 -gencode=arch=compute_$(SM),code=sm_$(SM) -use_fast_math
SOFLAG          = $(CUFLAG) -shared
INCLUDE         = -I. -I$(CUDA_PATH)/include -I$(TRT_PATH)/include
LDFLAG          = -L$(CUDA_PATH)/lib64 -lcudart -lcublas -lcublasLt -L$(TRT_PATH)/lib -lnvinfer -lz

SRC_ATTENTION   = $(shell find ./ -name '*.cu')

all: MaskPlugin.so

MaskPlugin.so: $(SRC_ATTENTION:.cu=.o)
	$(NVCC) $(SOFLAG) $(LDFLAG) -o $@ $^

%.o: %.cu
	$(NVCC) $(CUFLAG) $(INCLUDE) -o $@ -c $<

.PHONY: clean
clean:
	rm -rf ./*.so ./*.o ./*.d ./*.trt

.PHONY: t
t:
	clear
	python testMaskPlugin.py

