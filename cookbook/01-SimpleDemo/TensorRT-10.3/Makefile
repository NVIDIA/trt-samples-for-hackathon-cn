include $(TRT_COOKBOOK_PATH)/include/Makefile.inc

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
	make
	python3 main_numpy.py
	python3 main_pytorch.py
	python3 main_cookbook_flavor.py
	rm -rf *.trt
	./main.exe

.PHONY: clean
clean:
	rm -rf *.d *.o *.exe *.trt
