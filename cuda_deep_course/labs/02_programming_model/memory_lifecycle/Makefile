NVCC ?= nvcc
TARGET := memory_lifecycle
NVCCFLAGS := -O2 -std=c++17 -lineinfo

all: $(TARGET)

$(TARGET): memory_lifecycle.cu ../../common/cuda_check.cuh
	$(NVCC) $(NVCCFLAGS) memory_lifecycle.cu -o $(TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all run clean

