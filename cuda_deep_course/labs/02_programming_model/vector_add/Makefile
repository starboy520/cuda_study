NVCC ?= nvcc
TARGET := vector_add
NVCCFLAGS := -O2 -std=c++17

all: $(TARGET)

$(TARGET): vector_add.cu ../../common/cuda_check.cuh
	$(NVCC) $(NVCCFLAGS) vector_add.cu -o $(TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all run clean

