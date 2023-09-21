CUDA_PATH       = /usr/local/cuda
TRT_LIB_PATH    = /home/player/TensorRT-8.6.1.6/lib
TRT_INC_PATH    = /home/player/TensorRT-8.6.1.6/include
NVCC            = $(CUDA_PATH)/bin/nvcc
GCC = g++

cflags = -pthread -B /opt/conda/compiler_compat -Wno-unused-result -Wsign-compare -DNDEBUG -fwrapv -O2 -Wall -fPIC -O2 -isystem /opt/conda/include -fPIC -O2 -isystem /opt/conda/include -fPIC -I./ -I./src -I./cutlass/include -I/usr/local/cuda/include -c
post_cflags = -O3 -std=c++17 -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1013"' -DTORCH_EXTENSION_NAME=flash_attn_cuda -D_GLIBCXX_USE_CXX11_ABI=1
cuda_cflags = -I./ -I./src -I./cutlass/include -I$(CUDA_PATH)/include -I$(TRT_INC_PATH) -c
cuda_post_cflags = -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr --compiler-options ''"'"'-fPIC'"'"'' -O3 -std=c++17 -U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__ -U__CUDA_NO_HALF2_OPERATORS__ -U__CUDA_NO_BFLOAT16_CONVERSIONS__ --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math -lineinfo -gencode arch=compute_80,code=sm_80 --threads 4 -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1013"' -DTORCH_EXTENSION_NAME=flash_attn_cuda -D_GLIBCXX_USE_CXX11_ABI=1
LDFLAG          = -L$(CUDA_PATH)/lib64 -lcudart -L$(TRT_LIB_PATH) -lnvinfer

BUILD_DIR = build
OBJ_DIR = $(BUILD_DIR)/obj

SO_CUDA = $(addprefix $(BUILD_DIR)/, fMHAPlugin.so)
OBJ = $(shell find . -name '*.o')
DEP = $(OBJ:.o=.d)

all: $(SO_CUDA)

$(BUILD_DIR)/fMHAPlugin.so: $(addprefix $(OBJ_DIR)/, fMHAPlugin.o src/flash_fwd_hdim128_bf16_sm80.o src/flash_fwd_hdim128_fp16_sm80.o src/flash_fwd_hdim160_bf16_sm80.o src/flash_fwd_hdim160_fp16_sm80.o src/flash_fwd_hdim192_bf16_sm80.o src/flash_fwd_hdim192_fp16_sm80.o src/flash_fwd_hdim224_bf16_sm80.o  src/flash_fwd_hdim224_fp16_sm80.o src/flash_fwd_hdim256_bf16_sm80.o src/flash_fwd_hdim256_fp16_sm80.o src/flash_fwd_hdim32_bf16_sm80.o src/flash_fwd_hdim32_fp16_sm80.o src/flash_fwd_hdim64_bf16_sm80.o src/flash_fwd_hdim64_fp16_sm80.o src/flash_fwd_hdim96_bf16_sm80.o src/flash_fwd_hdim96_fp16_sm80.o)

-include $(DEP)

VPATH = src

$(OBJ_DIR)/%.o: %.cpp
	@mkdir -p $(@D)
	$(GCC) $(cflags) -MD -MP -o $@ -c $< $(post_cflags)

$(OBJ_DIR)/%.o: %.cu
	@mkdir -p $(@D)
	$(NVCC) $(cuda_cflags) -M -MT $@ -o $(@:.o=.d) $< $(cuda_post_cflags)
	$(NVCC) $(cuda_cflags) -o $@ -c $< $(cuda_post_cflags)

$(SO_CUDA):
	$(NVCC) -shared $(LDFLAG) -o $@ $+ $(cuda_post_cflags)

clean:
	rm -r $(SO_CUDA) $(OBJ_DIR)

distclean: clean
	cd $(BUILD_DIR) && rm -f out.*
