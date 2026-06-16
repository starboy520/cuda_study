NVCC ?= nvcc
TARGET := reduction
NVCCFLAGS := -O2 -std=c++17 -lineinfo

all: $(TARGET)

$(TARGET): reduction.cu ../../common/cuda_check.cuh
	$(NVCC) $(NVCCFLAGS) reduction.cu -o $(TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all run clean
