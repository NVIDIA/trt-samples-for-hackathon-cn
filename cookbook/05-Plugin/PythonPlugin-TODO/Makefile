include ../../include/Makefile.inc

circ_pad_plugin.so: circ_pad_plugin.o
	nvcc -shared -o circ_pad_plugin.so -L$(TRT_LIB_PATH) -lnvinfer circ_pad_plugin.o

circ_pad_plugin.o: circ_pad_plugin.cu
	nvcc -Xcompiler -fPIC -I $(CUDA_PATH)/include -I$(TRT_INC_PATH)/include -L$(TRT_LIB_PATH) -lnvinfer -c -o circ_pad_plugin.o circ_pad_plugin.cu

clean:
	rm circ_pad_plugin.o circ_pad_plugin.so
