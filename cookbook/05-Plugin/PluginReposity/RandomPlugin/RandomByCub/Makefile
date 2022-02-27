CUDA_PATH       = /usr/local/cuda
NVCC            = $(CUDA_PATH)/bin/nvcc
SM              = 61
CCFLAG          = -w -std=c++14 -O3
SOFLAG          = $(CCFLAG) -shared
INCLUDE         = -I. -I/usr/local/cuda/include -I/usr/include/x86_64-linux-gnu
LDFLAG          = -L/usr/local/cuda/lib64 -lcudart -lcurand -L/usr/local/cuda-11.1/targets/x86_64-linux/lib/ -lcublas -L/usr/lib/x86_64-linux-gnu -lnvinfer
SOURCE_CU       = $(shell find . -name '*.cu')
DEP             = $(SOURCE:.cu=.d)

-include $(DEP)

all: $(SOURCE_CU:%.cu=%.exe)

%.exe: %.cu
	$(NVCC) $(CCFLAG) $(LDFLAG) $(INCLUDE) -o $@ $<

.PHONY: test
test:
	./distribution30.exe
	./distribution192.exe

.PHONY: clean
clean:
	rm -rf ./*.d ./*.o ./*.exe

