include $(TRT_COOKBOOK_PATH)/include/Makefile.inc

OBJ         = $(shell find . -name *.o 2>/dev/null)
DEP         = $(OBJ:.o=.d)

-include $(DEP)

all: AddScalarPlugin.so

AddScalarPlugin.so: AddScalarPlugin.o cookbookHelper.o
	$(NVCC) $(SOFLAG) $(LDFLAG) -o $@ $+

AddScalarPlugin.o: AddScalarPlugin.cu
	$(NVCC) $(CUFLAG) $(INCLUDE) -M -MT $@ -o $(@:.o=.d) $<
	$(NVCC) $(CUFLAG) $(INCLUDE) -o $@ -c $<

cookbookHelper.o: $(TRT_COOKBOOK_PATH)/include/cookbookHelper.cu
	$(NVCC) $(CUFLAG) $(INCLUDE) -M -MT $@ -o $(@:.o=.d) $<
	$(NVCC) $(CUFLAG) $(INCLUDE) -o $@ -c $<

.PHONY: test
test:
	make clean
	make -j
	python3 main.py

.PHONY: clean
clean:
	rm -rf *.d *.o *.so *.trt
