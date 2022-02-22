CUDA_PATH       = /usr/local/cuda
NVCC            = $(CUDA_PATH)/bin/nvcc
SM              = 61
CCFLAG          = -w -std=c++14 -O3 -DNDEBUG -DDEBUG_ENABLE=0 -DBUILD_UTEST=0
SOFLAG          = $(CCFLAG) -shared
INCLUDE         = -I. -I/usr/local/cuda/include -I/usr/include/x86_64-linux-gnu 
LDFLAG          = -L/usr/local/cuda/lib64 -lcudart -lcublas -L/usr/lib/x86_64-linux-gnu -lnvinfer
SOURCE_CU       = $(shell find . -name '*.cu')
TEST_PYTHON     = $(shell find . -name 'test*.py')
DEP             = $(SOURCE:.cu=.d)

-include $(DEP)

all: $(SOURCE_CU:%.cu=%.so)

%.o: %.cu
	$(NVCC) $(CCFLAG) -Xcompiler -fPIC -arch=sm_$(SM) $(INCLUDE) -o $@ -c $<

%.o: %.cpp
	g++ -g $(CCFLAG) -fPIC $(INCLUDE) -o $@ -c $<

%.so: %.o
	$(NVCC) $(SOFLAG) $(LDFLAG) -o $@ $^

.PHONY: test
test:
	python $(TEST_PYTHON)

.PHONY: clean
clean:
	rm -rf ./*.so ./*.d ./*.o ./*.trt

