include ./Makefile.inc

LDFLAG     += -lz
SOURCE_CPP  = $(shell find . -name '*.cpp' 2>/dev/null)
OBJ         = $(shell find . -name *.o 2>/dev/null)
DEP         = $(OBJ:.o=.d)
TARGET_SO   = main.exe

-include $(DEP)

all: $(TARGET_SO)

main.exe: main.o cnpy.o
	$(NVCC) $(LDFLAG) $(LD_LZ) -o $@ $+

%.o: %.cpp
	$(NVCC) $(CUFLAG) $(INCLUDE) -M -MT $@ -o $(@:.o=.d) $<
	$(NVCC) $(CUFLAG) $(INCLUDE) -o $@ -c $<

cnpy.o: cnpy.cpp
	$(NVCC) $(CCFLAG) $(INCLUDE) -M -MT $@ -o $(@:.o=.d) $<
	$(NVCC) $(CCFLAG) $(INCLUDE) -o $@ -c $<

.PHONY: test
test:
	make clean
	make -j2

.PHONY: clean
clean:
	rm -rf ./*.d ./*.o ./*.exe ./*.plan
