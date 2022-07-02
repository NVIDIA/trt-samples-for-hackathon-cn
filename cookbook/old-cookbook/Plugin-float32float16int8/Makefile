CUDA_PATH       = /usr/local/cuda-10.2
TENSORRT_PATH   = /usr/local/tensorrt7.0-cuda10.2
NVCC            = $(CUDA_PATH)/bin/nvcc
GCC             = g++

CCFLAG          = -std=c++11 -DNDEBUG 
SOFLAG          = $(CCFLAG) -shared
INCLUDE         = -I. -I$(CUDA_PATH)/include -I$(TENSORRT_PATH)/include
LDFLAG          = -L$(CUDA_PATH)/lib64 -lcudart -L$(TENSORRT_PATH)/lib -lnvinfer

SOURCE_CU       = $(shell find . -name '*.cu')
TARGET_SO       = $(SOURCE_CU:%.cu=%.so) 
DEP_SO          = $(SOURCE_CU:%.cu=%.d)
SOURCE_CPP      = $(shell find . -name '*.cpp')
TARGET_EXE      = $(SOURCE_CPP:%.cpp=%.exe)
TEST            = $(shell find . -name '*.py')

#-include $(DEP_SO)

all: $(TARGET_SO) $(TARGET_EXE)
$(TARGET_SO) : $(SOURCE_CU:%.cu=%.o)
$(TARGET_EXE): $(SOURCE_CPP:%.cpp=%.o) $(SOURCE_CU:%.cu=%.o)

%.o: %.cpp
#	$(NVCC) $(CCFLAG) -M -MT $@ $(INCLUDE) -o $(@:.o=.d) $<
	$(GCC) $(CCFLAG) -fPIC -MD -MP $(INCLUDE) -o $@ -c $<

%.o: %.cu
#	$(NVCC) $(CCFLAG) -M -MT $@ $(INCLUDE) -o $(@:.o=.d) $<
	$(NVCC) $(CCFLAG) -m64 -Xcompiler -fPIC -arch=sm_61 $(INCLUDE) -o $@ -c $<

%.so: %.o
	$(NVCC) $(SOFLAG) $(LDFLAG) -o $@ $^

%.exe: %.o
	$(NVCC) $(CCFLAG) $(LDFLAG) -o $@ $^

.PHONY: test
test:
	python $(TEST)
	$(TARGET_EXE)

.PHONY: clean
clean:
	rm -rf ./*.d ./*.o ./*.so ./*.trt ./*.cache ./*.exe 

