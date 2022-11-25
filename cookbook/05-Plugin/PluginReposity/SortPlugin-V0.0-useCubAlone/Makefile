include ../../../include/Makefile.inc

SOURCE_CU   = $(shell find . -name '*.cu' 2>/dev/null)
OBJ         = $(shell find . -name *.o 2>/dev/null)
DEP         = $(OBJ:.o=.d)
TARGET_EXE   = $(SOURCE_CU:.cu=.exe)

-include $(DEP)

all: $(TARGET_EXE)

%.exe: %.cu
	$(NVCC) $(CCFLAG) $(LDFLAG) $(INCLUDE) -o $@ $<

.PHONY: test
test:
	make clean
	make
	./$(TARGET_EXE)

.PHONY: clean
clean:
	rm -rf ./*.d ./*.o ./*.so ./*.exe ./*.plan

