include ../../../include/Makefile.inc

LDFLAG      += -lnvonnxparser

SOURCE_CPP  = $(shell find . -name '*.cpp' 2>/dev/null)
SOURCE_PY   = $(shell find . -name '*.py' 2>/dev/null)
OBJ         = $(shell find . -name *.o 2>/dev/null)
DEP         = $(OBJ:.o=.d)
TARGET_EXE  = main.exe

-include $(DEP)

all: $(TARGET_EXE)

main.exe: main.o cnpy.o calibrator.o
	$(NVCC) $(LDFLAG) -lz -o $@ $+

%.o: %.cpp
	$(NVCC) $(CCFLAG) $(INCLUDE) -M -MT $@ -o $(@:.o=.d) $<
	$(NVCC) $(CCFLAG) $(INCLUDE) -Xcompiler -fPIC -o $@ -c $<

.PHONY: test
test:
	make clean
	make -j3
	python3 $(SOURCE_PY)
	./$(TARGET_EXE) > result-C++.log

.PHONY: clean
clean:
	rm -rf ./*.d ./*.o ./*.so ./*.exe ./*.plan

