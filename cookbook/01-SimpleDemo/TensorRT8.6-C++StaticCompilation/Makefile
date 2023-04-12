include ../../include/Makefile.inc

LDFLAG          = -L$(CUDA_PATH)/lib64 -lcudart -lnvrtc -lnvptxcompiler_static -L$(TRT_LIB_PATH) -lnvinfer_static -lcudnn -L$(TRT_LIB_PATH)/stubs/ -lcublas_static_stub_trt -lcublasLt_static_stub_trt

SOURCE_CPP  = $(shell find . -name '*.cpp' 2>/dev/null)
SOURCE_PY   = $(shell find . -name '*.py' 2>/dev/null)
OBJ         = $(shell find . -name *.o 2>/dev/null)
DEP         = $(OBJ:.o=.d)
TARGET_EXE  = $(SOURCE_CPP:.cpp=.exe)

-include $(DEP)

all: $(TARGET_EXE)
#.SECONDARY:

%.exe: %.o
	$(NVCC) $(CCFLAG) $(LDFLAG) -o $@ $+

%.o: %.cpp
	$(NVCC) $(CCFLAG) $(INCLUDE) -M -MT $@ -o $(@:.o=.d) $<
	$(NVCC) $(CCFLAG) $(INCLUDE) -Xcompiler -fPIC -o $@ -c $<

.PHONY: test
test:
	make clean
	make
	python3 $(SOURCE_PY)
	rm -rf ./*.plan
	./$(TARGET_EXE)

.PHONY: clean
clean:
	rm -rf ./*.d ./*.o ./*.so ./*.exe ./*.plan

