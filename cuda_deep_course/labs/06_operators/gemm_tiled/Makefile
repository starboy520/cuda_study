NVCC ?= nvcc
TARGET := gemm_tiled
NVCCFLAGS := -O3 -std=c++17 -arch=sm_75 -lineinfo

all: $(TARGET)

$(TARGET): gemm_tiled.cu ../../common/cuda_check.cuh
	$(NVCC) $(NVCCFLAGS) gemm_tiled.cu -o $(TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all run clean
