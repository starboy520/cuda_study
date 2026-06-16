NVCC ?= nvcc
TARGET := matrix_add_2d
NVCCFLAGS := -O2 -std=c++17 -lineinfo

all: $(TARGET)

$(TARGET): matrix_add_2d.cu ../../common/cuda_check.cuh
	$(NVCC) $(NVCCFLAGS) matrix_add_2d.cu -o $(TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all run clean

