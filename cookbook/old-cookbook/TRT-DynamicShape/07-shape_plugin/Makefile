CUDA_PATH       = /usr/local/cuda
TENSORRT_PATH   = /usr/local/tensorrt7
NVCC            = $(CUDA_PATH)/bin/nvcc
#GCC             = g++

CCFLAG          = -std=c++11 -DNDEBUG 
SOFLAG          = $(CCFLAG) -shared
INCLUDE         = -I. -I$(CUDA_PATH)/include -I$(TENSORRT_PATH)/include
LDFLAG          = -L$(CUDA_PATH)/lib64 -lcudart -L$(TENSORRT_PATH)/lib -lnvinfer

SOURCE_CU       = $(shell find . -name '*.cu')
TARGET_SO       = $(SOURCE_CU:%.cu=%.so) 
DEP_SO          = $(SOURCE_CU:%.cu=%.d)
TEST            = $(shell find . -name '*.py')

-include $(DEP_SO)

all: $(TARGET_SO)
$(TARGET_SO) : $(SOURCE_CU:%.cu=%.o)

%.o: %.cu
	$(NVCC) $(CCFLAG) -m64 -Xcompiler -fPIC -arch=sm_61 $(INCLUDE) -o $@ -c $<

%.so: %.o
	$(NVCC) $(SOFLAG) $(LDFLAG) -o $@ $^

.PHONY: test
test:
	python $(TEST)

.PHONY: clean
clean:
	rm -rf ./*.d ./*.o ./*.so ./*.trt ./*.cache ./*.exe 

