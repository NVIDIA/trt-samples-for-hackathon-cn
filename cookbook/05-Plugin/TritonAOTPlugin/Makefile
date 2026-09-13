TRT_COOKBOOK_PATH ?= $(shell d="$(abspath $(dir $(lastword $(MAKEFILE_LIST))))"; while [ "$$d" != "/" ] && [ ! -f "$$d/include/Makefile.inc" ]; do d=$$(dirname "$$d"); done; echo "$$d")
include $(TRT_COOKBOOK_PATH)/include/Makefile.inc

# The generated launcher uses the CUDA driver API (cuModuleLoadData / cuLaunchKernel), which
# Makefile.inc does not link
LDFLAG += -lcuda

STEM       := add_scalar
BLOCK_SIZE := 256

.PHONY: all
all: AddScalarTritonPlugin.so

# Step 1: ahead-of-time compile the Triton kernel into C source with the cubin embedded, then link
# the (here: single) specialization into a dispatcher whose symbol names do not contain the hash.
# Step 2 is the ABI fix: `triton.tools.compile` emits `double scalar` for an `fp32` kernel argument
# but passes `&scalar` straight to `cuLaunchKernel`, where the kernel's parameter slot is 4 bytes --
# the kernel then reads the low half of the double and silently computes `x + 0`. See README.md.
$(STEM).h $(STEM).c: kernel.py
	rm -f $(STEM).*.c $(STEM).*.h $(STEM).c $(STEM).h
	python3 -m triton.tools.compile kernel.py \
		--kernel-name add_scalar_kernel --out-name $(STEM) -o $(STEM) \
		--signature "*fp32:16, *fp32:16, fp32, i32, $(BLOCK_SIZE)" \
		--grid "(n_element + $(BLOCK_SIZE) - 1) / $(BLOCK_SIZE), 1, 1"
	python3 -m triton.tools.link $(STEM).*.h -o $(STEM)
	sed -i 's/double scalar/float scalar/g' $(STEM).c $(STEM).h $(STEM).*.c $(STEM).*.h

# The generated files are C, not CUDA: compile them with the host compiler
$(STEM).o: $(STEM).c $(STEM).h
	$(NVCC) -x c -O3 -Xcompiler -fPIC -I$(CUDA_PATH)/include -o $@ -c $<

# The per-specialization file carries the hash in its name (`add_scalar.<hash>_<spec>.c`), which
# changes whenever the kernel or the signature changes, so it has to be globbed by the shell rather
# than by `$(wildcard)`, whose directory listing is cached from before the files exist.
$(STEM)_kernel.o: $(STEM).c
	$(NVCC) -x c -O3 -Xcompiler -fPIC -I$(CUDA_PATH)/include -o $@ -c $$(ls $(STEM).*.c)

AddScalarTritonPlugin.o: AddScalarTritonPlugin.cu AddScalarTritonPlugin.h $(STEM).h
	$(COMMAND_O)

AddScalarTritonPlugin.so: AddScalarTritonPlugin.o cookbookHelper.o $(STEM).o $(STEM)_kernel.o
	$(COMMAND_SO)

.PHONY: build
build:
	make clean; make all -j

.PHONY: clean
clean:
	rm -rf *.d *.o *.so *.exe $(STEM).c $(STEM).h $(STEM).*.c $(STEM).*.h __pycache__ generated
