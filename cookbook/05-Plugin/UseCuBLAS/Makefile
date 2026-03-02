include $(TRT_COOKBOOK_PATH)/include/Makefile.inc

MORE_LIB    = -lcublas
PLUGIN_CU 	= $(shell find . -name '*.cu')
PLUGIN_O	= $(PLUGIN_CU:.cu=.o)
PLUGIN_SO	= $(PLUGIN_CU:.cu=.so)

all: $(PLUGIN_SO) useCuBLASAlone.exe

%.so: %.o cookbookHelper.o
	$(COMMAND_SO) $(MORE_LIB)

$(PLUGIN_O): $(PLUGIN_CU)
	$(COMMAND_O)

useCuBLASAlone.exe: useCuBLASAlone.o
	$(COMMAND_EXE) $(MORE_LIB)

useCuBLASAlone.o: useCuBLASAlone.cpp
	$(COMMAND_O)

.PHONY: test
test:
	make clean; make all -j; rm -rf *.trt
	./useCuBLASAlone.exe > log-useCuBLASAlone.exe.log

.PHONY: clean
clean:
	rm -rf *.d *.o *.so *.exe *.trt
