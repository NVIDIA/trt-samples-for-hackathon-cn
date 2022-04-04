CUDA_PATH   = /usr/local/cuda/
NVCC        = $(CUDA_PATH)/bin/nvcc
TRT_PATH    = /usr/lib/x86_64-linux-gnu/
SM          = 61
CCFLAG      = -std=c++11 -UNDEBUG -DDEBUG -O3 -arch=sm_$(SM)
SOFLAG          = $(CCFLAG) -shared
INCLUDE     = -I. -I$(CUDA_PATH)/include -I$(TRT_PATH)/include
LDFLAG      = -L$(CUDA_PATH)/lib64 -lcudart -L$(TRT_PATH)/lib -lnvinfer

SOURCE_CU   = $(shell find . -name '*.cu')
SOURCE_PY   = $(shell find . -name '*.py')
OBJ         = $(shell find . -name *.o)
DEP         = $(OBJ:.o=.d)

-include $(DEP)

all: $(SOURCE_CU:%.cu=%.so)

%.so: %.o
	$(NVCC) $(SOFLAG) $(LDFLAG) -o $@ $^

%.o: %.cu
	$(NVCC) $(CCFLAG) $(INCLUDE) -M -MT $@ -o $(@:.o=.d) $<
	$(NVCC) $(CCFLAG) $(INCLUDE) -Xcompiler -fPIC -o $@ -c $<
	
.PHONY: test
test:
	make clean
	make
	python $(SOURCE_PY)

.PHONY: clean
clean:
	rm -rf ./*.d ./*.o ./*.so ./*.plan

