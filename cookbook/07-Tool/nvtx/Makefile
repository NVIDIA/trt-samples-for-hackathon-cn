include $(TRT_COOKBOOK_PATH)/include/Makefile.inc

LDFLAG     += -lnvToolsExt

OBJ         = $(shell find . -name *.o 2>/dev/null)
DEP         = $(OBJ:.o=.d)

-include $(DEP)

all: main.exe

main.exe: main.o cookbookHelper.o
	$(NVCC) $(LDFLAG) -o $@ $+

main.o: main.cpp
	$(NVCC) $(CUFLAG) $(INCLUDE) -M -MT $@ -o $(@:.o=.d) $<
	$(NVCC) $(CUFLAG) $(INCLUDE) -o $@ -c $<

cookbookHelper.o: $(TRT_COOKBOOK_PATH)/include/cookbookHelper.cu
	$(NVCC) $(CUFLAG) $(INCLUDE) -M -MT $@ -o $(@:.o=.d) $<
	$(NVCC) $(CUFLAG) $(INCLUDE) -o $@ -c $<

.PHONY: test
test:
	make clean
	make -j

	python3 main.py > result-Python.log
	./main.exe > result-C++.log

.PHONY: clean
clean:
	rm -rf ./*.d ./*.o ./*.so ./*.exe ./*.trt
