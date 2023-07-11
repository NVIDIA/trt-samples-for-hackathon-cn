CUDA_PATH       = /usr/local/cuda
NVCC            = $(CUDA_PATH)/bin/nvcc
SM              = 61
DEBUG_FLAG      = -DDEBUG
CCFLAG          = -w $(DEBUG_FLAG) -std=c++14 -O3
SOFLAG          = $(CCFLAG) -shared
INCLUDE         = -I. -I/usr/local/cuda/include -I/usr/include/x86_64-linux-gnu
LDFLAG          = -L/usr/local/cuda/lib64 -lcudart -L/usr/local/cuda-11.1/targets/x86_64-linux/lib/ -lcublas -L/usr/lib/x86_64-linux-gnu -lnvinfer
SOURCE_CU       = $(shell find . -name '*.cpp')
TEST_PYTHON     = $(shell find . -name '*.py')
DEP             = $(SOURCE:.cu=.d)

-include $(DEP)

all: $(SOURCE_CU:%.cpp=%.exe)

%.o: %.cu
	$(NVCC) $(CCFLAG) -Xcompiler -fPIC -arch=sm_$(SM) $(INCLUDE) -o $@ -c $<

%.o: %.cpp
	g++ -g $(CCFLAG) -fPIC $(INCLUDE) -o $@ -c $<

%.so: %.o
	$(NVCC) $(SOFLAG) $(LDFLAG) -o $@ $^
	
%.exe: %.cpp
	$(NVCC) $(INCLUDE) $(LDFLAG) -o $@ $^

.PHONY: test
test:
	python3 $(TEST_PYTHON)

.PHONY: clean
clean:
	rm -rf ./*.so ./*.d ./*.o ./*.plan

