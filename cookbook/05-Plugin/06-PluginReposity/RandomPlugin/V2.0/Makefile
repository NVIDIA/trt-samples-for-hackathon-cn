CUDA_PATH       = /usr/local/cuda
NVCC            = $(CUDA_PATH)/bin/nvcc
SM              = 61
CCFLAG          = -w -std=c++14 -O3
SOFLAG          = $(CCFLAG) -shared
INCLUDE         = -I. -I/usr/local/cuda/include -I/usr/include/x86_64-linux-gnu 
LDFLAG          = -L/usr/local/cuda/lib64 -lcudart -lcurand -L/usr/local/cuda-11.1/targets/x86_64-linux/lib/ -lcublas -L/usr/lib/x86_64-linux-gnu -lnvinfer
SOURCE_CU       = $(shell find . -name '*.cu')
TEST_PY         = $(shell find . -name '*.py')
DEP             = $(SOURCE:.cu=.d)

-include $(DEP)

all: $(SOURCE_CU:%.cu=%.so)

%.so: %.o
	$(NVCC) $(SOFLAG) $(LDFLAG) -o $@ $^

%.o: %.cu
	$(NVCC) $(CCFLAG) -Xcompiler -fPIC -arch=sm_$(SM) $(INCLUDE) -o $@ -c $<

.PHONY: test
test:
	python $(TEST_PY)

.PHONY: clean
clean:
	rm -rf ./*.so ./*.d ./*.o ./*.trt

