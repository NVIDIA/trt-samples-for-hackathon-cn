include ../../Makefile.inc

SOURCE_CU   = $(shell find . -name '*.cu' 2>/dev/null)
SOURCE_PY   = $(shell find . -name 'test*.py' 2>/dev/null)
OBJ         = $(shell find . -name *.o 2>/dev/null)
DEP         = $(OBJ:.o=.d)
TARGET_SO   = $(SOURCE_CU:.cu=.so)

-include $(DEP)

all: $(TARGET_SO)

%.so: %.o cnpy.o
	$(NVCC) $(SOFLAG) $(LDFLAG) -o $@ $+

%.o: %.cu
	$(NVCC) $(CUFLAG) $(INCLUDE) -M -MT $@ -o $(@:.o=.d) $<
	$(NVCC) $(CUFLAG) $(INCLUDE) -o $@ -c $<

cnpy.o: cnpy.cpp
	$(NVCC) $(CCFLAG) $(INCLUDE) -M -MT $@ -o $(@:.o=.d) $<
	$(NVCC) $(CCFLAG) $(INCLUDE) -o $@ -c $<

.PHONY: test
test:
	make clean
	make -j2
	python3 createData.py
	python3 $(SOURCE_PY)

.PHONY: clean
clean:
	rm -rf ./*.d ./*.o ./*.so ./*.exe ./*.plan ./*.npz

