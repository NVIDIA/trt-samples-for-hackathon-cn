# TensorRT 11.2 added IRefitterObserver. The system install may still be older, so this example
# builds against an unpacked TensorRT >= 11.2 tarball pointed at by TRT_OBSERVER_PATH.
TRT_OBSERVER_PATH ?= /work/trt/TensorRT-11.3.0.93
CUDA_PATH         ?= /usr/local/cuda

all: main.exe

main.exe: main.cpp
	g++ -std=c++17 -O2 -w \
		-I$(TRT_OBSERVER_PATH)/include -I$(CUDA_PATH)/include \
		$< \
		-L$(TRT_OBSERVER_PATH)/lib -lnvinfer -lnvonnxparser \
		-L$(CUDA_PATH)/lib64 -lcudart \
		-Wl,-rpath,$(TRT_OBSERVER_PATH)/lib \
		-o $@

.PHONY: build
build:
	make clean; make all -j

.PHONY: clean
clean:
	rm -rf *.d *.o *.exe
