CUDA_PATH := /usr/local/cuda
TRT_SRC := /home/scratch.skarunaratne_sw/trt
TRT_LIB := /home/scratch.skarunaratne_sw/Downloads/TensorRT-8.6.0.9/TensorRT-8.6.0.9/lib

circ_pad_plugin.so: circ_pad_plugin.o
	nvcc -shared -o circ_pad_plugin.so circ_pad_plugin.o -L $(TRT_LIB) -lnvinfer

circ_pad_plugin.o: circ_pad_plugin.cu
	nvcc circ_pad_plugin.cu -I $(CUDA_PATH)/include -I $(TRT_SRC)/include -L $(TRT_LIB) --compiler-options -fPIC -c -o circ_pad_plugin.o  -lnvinfer

clean:
	rm circ_pad_plugin.o circ_pad_plugin.so
