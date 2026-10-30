# ARCH=native builds for the GPU in this machine.
# Override for a specific card, e.g. make ARCH=sm_75 (T4), sm_80 (A100), sm_89 (L4), sm_90 (H100)
ARCH ?= native

NVCC ?= nvcc
NVCCFLAGS = -O3 -std=c++17 -arch=$(ARCH) -lineinfo

matmul_gpu: matmul_gpu.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $< -lcublas

clean:
	rm -f matmul_gpu
	rm -rf reports

.PHONY: clean
