include $(TRT_COOKBOOK_PATH)/include/Makefile.inc

all: main.exe

main.exe: main.o cookbookHelper.o
	$(COMMAND_EXE)

main.o: main.cpp
	$(COMMAND_O)

.PHONY: build
build:
	make clean; make all -j; rm -rf *.trt

.PHONY: clean
clean:
	rm -rf *.d *.exe *.o *.trt
