include ../../../include/Makefile.inc

SOURCE_CU   = $(shell find . -name '*.cu' 2>/dev/null)
OBJ         = $(shell find . -name *.o 2>/dev/null)
DEP         = $(OBJ:.o=.d)
TARGET_SO   = $(SOURCE_CU:.cu=.so)

-include $(DEP)

all: $(TARGET_SO)

%.so: %.o
	$(NVCC) $(SOFLAG) $(LDFLAG) -o $@ $+

%.o: %.cu
	$(NVCC) $(CUFLAG) $(INCLUDE) -M -MT $@ -o $(@:.o=.d) $<
	$(NVCC) $(CUFLAG) $(INCLUDE) -o $@ -c $<

.PHONY: test
test:
	make clean
	make -j2
	python3 testLayerNormPluginCUBV1.py
	python3 testLayerNormPluginCUBV2.py
	python3 testLayerNormPluginCUBV3.py
	python3 testLayerNormPluginCUBV4.py
	python3 testLayerNormPluginOneFlow.py
	
.PHONY: clean
clean:
	rm -rf ./*.d ./*.o ./*.so ./*.exe ./*.plan

