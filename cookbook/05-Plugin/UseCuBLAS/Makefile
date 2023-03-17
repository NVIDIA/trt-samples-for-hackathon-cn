include ../../include/Makefile.inc
LDFLAG     += -lcublas

SOURCE_CU0  = CuBLASGemmPlugin.cu
SOURCE_CU1  = useCuBLASAlone.cu
SOURCE_PY   = $(shell find . -name '*.py' 2>/dev/null)
OBJ         = $(shell find . -name *.o 2>/dev/null)
DEP         = $(OBJ:.o=.d)
TARGET_SO   = $(SOURCE_CU0:.cu=.so)
TARGET_EXE  = $(SOURCE_CU1:.cu=.exe)

-include $(DEP)

all: $(TARGET_SO) $(TARGET_EXE)

%.so: %.o
	$(NVCC) $(SOFLAG) $(LDFLAG) -o $@ $+

%.exe: %.o
	$(NVCC) $(LDFLAG) -o $@ $^

%.o: %.cu
	$(NVCC) $(CUFLAG) $(INCLUDE) -M -MT $@ -o $(@:.o=.d) $<
	$(NVCC) $(CUFLAG) $(INCLUDE) -o $@ -c $<

.PHONY: test
test:
	make clean
	make -j2
	python3 $(SOURCE_PY)
	./$(TARGET_EXE)

.PHONY: clean
clean:
	rm -rf ./*.d ./*.o ./*.so ./*.exe ./*.plan

