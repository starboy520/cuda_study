NVCC ?= nvcc
TARGET := gemm_naive
NVCCFLAGS := -O2 -std=c++17 -lineinfo

all: $(TARGET)

$(TARGET): gemm_naive.cu ../../common/cuda_check.cuh
	$(NVCC) $(NVCCFLAGS) gemm_naive.cu -o $(TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all run clean
