include $(TRT_COOKBOOK_PATH)/include/Makefile.inc

LDFLAG      += -lnvonnxparser

SOURCE_CPP  = $(shell find . -name '*.cpp' 2>/dev/null)
OBJ         = $(SOURCE_CPP:.cpp=.o)
DEP         = $(SOURCE_CPP:.cpp=.d)
TARGET_EXE  = $(SOURCE_CPP:.cpp=.exe)

-include $(DEP)

all: $(TARGET_EXE)

main.exe: main.o cnpy.o calibrator.o
	echo $(TARGET_EXE)
	$(NVCC) $(LDFLAG) -lz -o $@ $+

main.o: main.cpp
	echo $(TARGET_EXE)
	$(NVCC) $(CCFLAG) $(INCLUDE) -M -MT $@ -o $(@:.o=.d) $<
	$(NVCC) $(CCFLAG) $(INCLUDE) -Xcompiler -fPIC -o $@ -c $<

cnpy.o: $(TRT_COOKBOOK_PATH)/include/cnpy.cpp
	echo $(TARGET_EXE)
	$(NVCC) $(CCFLAG) $(INCLUDE) -Xcompiler -fPIC -o $@ -c $<

calibrator.o: $(TRT_COOKBOOK_PATH)/include/calibrator.cpp
	echo $(TARGET_EXE)
	$(NVCC) $(CCFLAG) $(INCLUDE) -Xcompiler -fPIC -o $@ -c $<

.PHONY: test
test:
	make clean
	make -j3
	./$(TARGET_EXE) > result-C++.log

.PHONY: clean
clean:
	rm -rf *.d *.exe *.o *.trt *.Int8Cache
