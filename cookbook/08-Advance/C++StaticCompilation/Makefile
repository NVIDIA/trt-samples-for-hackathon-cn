include ../../include/Makefile.inc

LDFLAG      = -L$(CUDA_PATH)/lib64 -lcudart -lnvrtc -lnvptxcompiler_static -L$(TRT_LIB_PATH) -lnvinfer_static -lcudnn -L$(TRT_LIB_PATH)/stubs/ -lcublas_static_stub_trt -lcublasLt_static_stub_trt

OBJ         = $(shell find . -name *.o 2>/dev/null)
DEP         = $(OBJ:.o=.d)

-include $(DEP)

all: main.exe

main.exe: main.o cookbookHelper.o
	$(NVCC) $(CCFLAG) $(LDFLAG) -o $@ $+

cookbookHelper.o: $(TRT_COOKBOOK_PATH)/include/cookbookHelper.cu
	$(NVCC) $(CCFLAG) $(INCLUDE) -M -MT $@ -o $(@:.o=.d) $<
	$(NVCC) $(CCFLAG) $(INCLUDE) -Xcompiler -fPIC -o $@ -c $<

main.o: main.cpp
	$(NVCC) $(CCFLAG) $(INCLUDE) -M -MT $@ -o $(@:.o=.d) $<
	$(NVCC) $(CCFLAG) $(INCLUDE) -Xcompiler -fPIC -o $@ -c $<

.PHONY: test
test:
	make clean
	make -j
	./main.exe

.PHONY: clean
clean:
	rm -rf *.d *.exe *.o *.so *.trt
