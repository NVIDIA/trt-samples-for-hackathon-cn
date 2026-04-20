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

.PHONY: build
build:
	make clean
	pushd ../BasicExample               && make build && cp AddScalarPlugin.so ../APIs/AddScalarPluginV3.so
	pushd ../BasicExample-V2-deprecated && make build && cp AddScalarPlugin.so ../APIs/AddScalarPluginV2.so

.PHONY: clean
clean:
	rm -rf *.d *.o *.so *.exe *.trt
