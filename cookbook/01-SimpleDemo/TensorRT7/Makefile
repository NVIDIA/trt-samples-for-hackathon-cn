CUDA_PATH       = /usr/local/cuda
SM              = 61
TRT_PATH        = -L/usr/lib/x86_64-linux-gnu
NVCC            = $(CUDA_PATH)/bin/nvcc
CCFLAG          = -std=c++11 -DNDEBUG -arch=sm_$(SM)
INCLUDE         = -I. -I$(CUDA_PATH)/include -I$(TENSORRT_PATH)/include
LDFLAG          = -L$(CUDA_PATH)/lib64 -lcudart -L$(TENSORRT_PATH)/lib -lnvinfer

SOURCE_CPP      = $(shell find . -name '*.cpp')
TARGET_EXE      = $(SOURCE_CPP:%.cpp=%.exe)

all: $(TARGET_EXE)

%.exe: %.cpp
	$(NVCC) $(CCFLAG) $(INCLUDE) $(LDFLAG) -o $@ $^

.PHONY: test
test:
	make clean
	make
	python TRT7.py
	rm ./*.plan
	$(TARGET_EXE)

.PHONY: clean
clean:
	rm -rf ./*.d ./*.o ./*.so ./*.plan ./*.exe

