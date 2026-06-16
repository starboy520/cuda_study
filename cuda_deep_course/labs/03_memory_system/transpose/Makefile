NVCC ?= nvcc
TARGET := transpose
NVCCFLAGS := -O2 -std=c++17 -lineinfo

all: $(TARGET)

$(TARGET): transpose.cu ../../common/cuda_check.cuh
	$(NVCC) $(NVCCFLAGS) transpose.cu -o $(TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all run clean

