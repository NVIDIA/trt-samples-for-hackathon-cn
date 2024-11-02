include ../../include/Makefile.inc

MORE_LIB = -L$(CUDA_PATH)/lib64 -lcudart -lnvrtc -lnvptxcompiler_static -L$(TRT_LIB_PATH) -lnvinfer_static -lcudnn -L$(TRT_LIB_PATH)/stubs/

all: main.exe

main.exe: main.o cookbookHelper.o
	$(COMMAND_EXE) $(MORE_LIB)

main.o: main.cpp
	$(COMMAND_O)

.PHONY: test
test:
	make clean; make all -j; rm -rf *.trt
	./main.exe > log-main.exe.log

.PHONY: clean
clean:
	rm -rf *.d *.exe *.o *.so *.exe *.trt
