.DEFAULT_GOAL := all

include $(TRT_COOKBOOK_PATH)/include/Makefile.inc

PLUGIN_CU  := $(shell find . -name '*.cu')
PLUGIN_O   := $(PLUGIN_CU:.cu=.o)
PLUGIN_SO  := $(filter-out ./cookbookHelper.o, $(PLUGIN_O))
PLUGIN_SO  := $(PLUGIN_SO:.o=.so)

.PHONY: all
all: $(PLUGIN_SO) main.exe

$(PLUGIN_SO): %.so: %.o cookbookHelper.o
	$(COMMAND_SO)

%.o: %.cu
	$(COMMAND_O)

main.exe: main.o cookbookHelper.o
	$(COMMAND_EXE)

main.o: main.cpp
	$(COMMAND_O)

.PHONY: build
build:
	make clean; make all -j

.PHONY: clean
clean:
	rm -rf *.d *.o *.so *.exe
