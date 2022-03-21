CUDA_PATH   = /usr/local/cuda
NVCC        = $(CUDA_PATH)/bin/nvcc
SM          = 61
CCFLAG      = -std=c++11 -DNDEBUG -O3 -arch=sm_$(SM)
INCLUDE     = -I. -I$(CUDA_PATH)/include -I/usr/include/x86_64-linux-gnu
LDFLAG      = -L$(CUDA_PATH)/lib64 -lcudart
SOURCE_CU       = $(shell find . -name '*.cu')
DEP             = $(SOURCE:.cu=.d)

-include $(DEP)

all: $(SOURCE_CU:%.cu=%.exe)

%.exe: %.cu
	$(NVCC) $(CCFLAG) $(LDFLAG) $(INCLUDE) -o $@ $<

.PHONY: test
test:
	make clean
	make
	$(SOURCE_CU:%.cu=%.exe)

.PHONY: clean
clean:
	rm -rf ./*.d ./*.o ./*.exe

