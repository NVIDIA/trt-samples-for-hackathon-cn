CUDA_PATH   = /usr/local/cuda/
NVCC        = $(CUDA_PATH)/bin/nvcc
TRT_PATH    = /usr/lib/x86_64-linux-gnu/
#SM          = 61
            # 61 for GTX1070, 75 for T4,80 for A30
GENCODE     = -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_86,code=sm_86
CUFLAG      = -w -std=c++14 -O3 -DDEBUG -Xcompiler -fPIC $(GENCODE)
CPPFLAG     = -w -std=c++14 -O3 -use_fast_math
SOFLAG      = $(CUFLAG) -shared
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
	$(NVCC) $(CUFLAG) $(INCLUDE) -M -MT $@ -o $(@:.o=.d) $<
	$(NVCC) $(CUFLAG) $(INCLUDE) -o $@ -c $<

.PHONY: test
test:
	make clean
	make
	python $(SOURCE_PY)

.PHONY: clean
clean:
	rm -rf ./*.d ./*.o ./*.so ./*.plan

