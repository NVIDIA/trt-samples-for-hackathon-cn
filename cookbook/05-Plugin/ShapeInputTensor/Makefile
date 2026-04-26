include $(TRT_COOKBOOK_PATH)/include/Makefile.inc

PLUGIN_CU 	= $(shell find . -name '*.cu')
PLUGIN_O	= $(PLUGIN_CU:.cu=.o)
PLUGIN_SO	= $(PLUGIN_CU:.cu=.so)

all: $(PLUGIN_SO)

%.so: %.o cookbookHelper.o
	$(COMMAND_SO)

$(PLUGIN_O): $(PLUGIN_CU)
	$(COMMAND_O)

.PHONY: build
build:
	make clean; make all -j

.PHONY: clean
clean:
	rm -rf *.d *.o *.so *.exe
