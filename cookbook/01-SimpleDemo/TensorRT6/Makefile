CUDA_PATH   = /usr/local/cuda/
NVCC        = $(CUDA_PATH)/bin/nvcc
TRT_PATH    = /usr/lib/x86_64-linux-gnu/
SM          = 61
CCFLAG      = -std=c++11 -DNDEBUG -O3 -arch=sm_$(SM)
INCLUDE     = -I. -I$(CUDA_PATH)/include -I$(TRT_PATH)/include
LDFLAG      = -L$(CUDA_PATH)/lib64 -lcudart -L$(TRT_PATH)/lib -lnvinfer

SOURCE_CPP  = $(shell find . -name '*.cpp')
SOURCE_PY   = $(shell find . -name '*.py')
TARGET_EXE  = $(SOURCE_CPP:.cpp=.exe)
OBJ         = $(shell find . -name *.o)
DEP         = $(OBJ:.o=.d)

-include $(DEP)

all: $(SOURCE_CPP:.cpp=.exe)

%.exe: %.o
	$(NVCC) $(CCFLAG) $(LDFLAG) -o $@ $+

%.o: %.cpp
	$(NVCC) $(CCFLAG) $(INCLUDE) -M -MT $@ -o $(@:.o=.d) $<
	$(NVCC) $(CCFLAG) $(INCLUDE) -Xcompiler -fPIC -o $@ -c $<

.PHONY: test
test:
	make clean
	make
	python $(SOURCE_PY)
	rm ./*.plan
	./$(TARGET_EXE)

.PHONY: clean
clean:
	rm -rf ./*.d ./*.o ./*.so ./*.exe ./*.plan

