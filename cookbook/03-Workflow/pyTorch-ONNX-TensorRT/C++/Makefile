include $(TRT_COOKBOOK_PATH)/include/Makefile.inc

MORE_LIB = -lnvonnxparser -lz

all: main.exe

main.exe: main.o cookbookHelper.o cnpy.o calibrator.o
	$(COMMAND_EXE) $(MORE_LIB)

main.o: main.cpp
	$(COMMAND_O)

.PHONY: build
build:
	make clean; make all -j

.PHONY: clean
clean:
	rm -rf *.d *.exe *.Int8Cache*.o *.trt
