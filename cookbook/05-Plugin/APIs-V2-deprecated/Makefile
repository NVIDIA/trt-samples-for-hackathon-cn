include $(TRT_COOKBOOK_PATH)/include/Makefile.inc

PLUGIN_CU 	= $(shell find . -name '*.cu')
PLUGIN_O	= $(PLUGIN_CU:.cu=.o)
PLUGIN_SO	= $(PLUGIN_CU:.cu=.so)

all: $(PLUGIN_SO) main.exe

%.so: %.o cookbookHelper.o
	$(COMMAND_SO)

$(PLUGIN_O): $(PLUGIN_CU)
	$(COMMAND_O)

main.exe: main.o cookbookHelper.o $(PLUGIN_O)
	$(COMMAND_EXE)

main.o: main.cpp
	$(COMMAND_O)

.PHONY: test
test:
	make clean; make all -j; rm -rf *.trt
	./main.exe > log-main.exe.log

.PHONY: clean
clean:
	rm -rf *.d *.o *.so *.exe *.trt
