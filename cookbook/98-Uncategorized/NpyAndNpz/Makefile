include $(TRT_COOKBOOK_PATH)/include/Makefile.inc

LDFLAG      += -lz
OBJ         = $(shell find . -name *.o 2>/dev/null)
DEP         = $(OBJ:.o=.d)

-include $(DEP)

all: main.exe

main.exe: main.o
	$(NVCC) $(CCFLAG) $(LDFLAG) -o $@ $+

main.o: main.cpp
	$(NVCC) $(CUFLAG) $(INCLUDE) -M -MT $@ -o $(@:.o=.d) $<
	$(NVCC) $(CUFLAG) $(INCLUDE) -o $@ -c $<

.PHONY: test
test:
	make clean
	make main.exe
	python3 get_data.py
	./main.exe

.PHONY: clean
clean:
	rm -rf *.d *.exe *.npy *.npz *.o
