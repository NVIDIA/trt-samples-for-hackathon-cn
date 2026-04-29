TRT_COOKBOOK_PATH ?= $(shell d="$(abspath $(dir $(lastword $(MAKEFILE_LIST))))"; while [ "$$d" != "/" ] && [ ! -f "$$d/include/Makefile.inc" ]; do d=$$(dirname "$$d"); done; echo "$$d")
include $(TRT_COOKBOOK_PATH)/include/Makefile.inc

all: main.exe

main.exe: main.o cookbookHelper.o cnpy.o
	$(COMMAND_EXE) -lz

main.o: main.cpp
	$(COMMAND_O)

.PHONY: build
build:
	make clean; make all -j

.PHONY: clean
clean:
	rm -rf *.d *.o *.so *.exe
