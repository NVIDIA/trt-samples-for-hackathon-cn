include $(TRT_COOKBOOK_PATH)/include/Makefile.inc

LDFLAG      += -lnvonnxparser

OBJ         = $(shell find . -name *.o 2>/dev/null)
DEP         = $(SOURCE_CPP:.cpp=.d)

-include $(DEP)

all: main.exe

main.exe: main.o calibrator.o cookbookHelper.o cnpy.o
	$(NVCC) $(LDFLAG) -lz -o $@ $+

main.o: main.cpp
	$(NVCC) $(CCFLAG) $(INCLUDE) -M -MT $@ -o $(@:.o=.d) $<
	$(NVCC) $(CCFLAG) $(INCLUDE) -Xcompiler -fPIC -o $@ -c $<

calibrator.o: $(TRT_COOKBOOK_PATH)/include/calibrator.cpp
	$(NVCC) $(CCFLAG) $(INCLUDE) -Xcompiler -fPIC -o $@ -c $<

cookbookHelper.o: $(TRT_COOKBOOK_PATH)/include/cookbookHelper.cu
	$(NVCC) $(CUFLAG) $(INCLUDE) -M -MT $@ -o $(@:.o=.d) $<
	$(NVCC) $(CUFLAG) $(INCLUDE) -o $@ -c $<

cnpy.o: $(TRT_COOKBOOK_PATH)/include/cnpy.cpp
	$(NVCC) $(CCFLAG) $(INCLUDE) -Xcompiler -fPIC -o $@ -c $<

.PHONY: test
test:
	make clean
	make -j3
	./main.exe > result-C++.log

.PHONY: clean
clean:
	rm -rf *.d *.exe *.o *.trt *.Int8Cache
