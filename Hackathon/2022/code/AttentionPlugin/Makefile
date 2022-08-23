CUDA_PATH   = /usr/local/cuda
NVCC        = $(CUDA_PATH)/bin/nvcc
TRT_PATH    = /usr/lib/x86_64-linux-gnu
GENCODE     = -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_86,code=sm_86
DEBUG_MACRO = -UDEBUG
CUFLAG      = -w -std=c++14 -O3 $(DEBUG_MACRO) -Xcompiler -fPIC $(GENCODE)
CCFLAG      = -w -std=c++14 -O3 $(DEBUG_MACRO) -Xcompiler -fPIC -use_fast_math
SOFLAG      = -shared
INCLUDE     = -I. -I$(CUDA_PATH)/include -I$(TRT_PATH)/include
LDFLAG      = -L$(CUDA_PATH)/lib64 -lcudart -lcublas -lcublasLt -L$(TRT_PATH)/lib -lnvinfer -L/usr/lib/x86_64-linux-gnu -lz

SRC_ATTENTION   = $(shell find ./ -name '*.cu')

all: Attention.so

%.o: %.cu
	$(NVCC) $(CUFLAG) $(INCLUDE) -o $@ -c $<

Attention.so: $(SRC_ATTENTION:.cu=.o)
	$(NVCC) $(SOFLAG) $(LDFLAG) -o $@ $^

.PHONY: clean
clean:
	rm -rf ./*.so ./*/*.so ./GemmFile/runEnc  ./*/*.o ./*/*.trt #./*.trt



