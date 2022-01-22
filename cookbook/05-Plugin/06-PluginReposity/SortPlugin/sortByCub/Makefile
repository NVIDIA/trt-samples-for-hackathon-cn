CUDA_PATH       = /usr/local/cuda
NVCC            = $(CUDA_PATH)/bin/nvcc
SM              = 61
CCFLAG          = -w -std=c++14 -O3
INCLUDE         = -I. -I/usr/local/cuda/include -I/usr/include/x86_64-linux-gnu -I../cub/cub/device
LDFLAG          = -L/usr/local/cuda/lib64 -lcudart -L/usr/local/cuda/targets/x86_64-linux/lib/ -lcublas -L/usr/lib/x86_64-linux-gnu -lnvinfer
SOURCE_CU       = $(shell find . -name '*.cu')
DEP             = $(SOURCE:.cu=.d)

-include $(DEP)

all: $(SOURCE_CU:%.cu=%.exe)

%.exe: %.cu
	$(NVCC) $(CCFLAG) -arch=sm_$(SM) $(INCLUDE) -o $@ $<

.PHONY: test
test:
	./$(SOURCE_CU:%.cu=%.exe)

.PHONY: clean
clean:
	rm -rf ./*.d ./*.o ./*.exe

