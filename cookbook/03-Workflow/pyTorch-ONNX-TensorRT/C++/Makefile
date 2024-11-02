include $(TRT_COOKBOOK_PATH)/include/Makefile.inc

MORE_LIB = -lnvonnxparser -lz

all: main.exe

main.exe: main.o cookbookHelper.o cnpy.o calibrator.o
	$(COMMAND_EXE) $(MORE_LIB)

main.o: main.cpp
	$(COMMAND_O)

.PHONY: test
test:
	make clean; make all -j; rm -rf *.trt *.*.Int8Cache
	./main.exe > log-main.exe.log

.PHONY: clean
clean:
	rm -rf *.d *.o *.exe *.trt *.Int8Cache
