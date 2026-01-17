.DEFAULT_GOAL := all

include $(TRT_COOKBOOK_PATH)/include/Makefile.inc

PLUGIN_CU  := $(shell find . -name '*.cu')
PLUGIN_O   := $(PLUGIN_CU:.cu=.o)
PLUGIN_SO  := $(filter-out ./cookbookHelper.o, $(PLUGIN_O))
PLUGIN_SO  := $(PLUGIN_SO:.o=.so)

.PHONY: all
all: $(PLUGIN_SO)

$(PLUGIN_SO): %.so: %.o cookbookHelper.o
	$(COMMAND_SO)

%.o: %.cu
	$(COMMAND_O)



.PHONY: test
test:
	make clean; make -j; rm -rf *.trt; python3 main.py

.PHONY: clean
clean:
	rm -rf *.d *.o *.so *.exe *.trt
