include $(TRT_COOKBOOK_PATH)/include/Makefile.inc

DEP         = $(OBJ:.o=.d)
TARGET_EXE  = $(SOURCE_CPP:.cpp=.exe)

-include $(DEP)

all: main.exe

main.exe: main.o cookbookHelper.o
	$(NVCC) $(LDFLAG) -o $@ $+

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
	python3 main.py > result-Python.log
	chmod +x main.exe
	./main.exe > result-C++.log

.PHONY: clean
clean:
	rm -rf *.d *.o  *.onnx *.exe *.trt
