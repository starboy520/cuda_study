NVCC ?= nvcc
TARGET := function_qualifiers
NVCCFLAGS := -O2 -std=c++17

all: $(TARGET)

$(TARGET): function_qualifiers.cu ../../common/cuda_check.cuh
	$(NVCC) $(NVCCFLAGS) function_qualifiers.cu -o $(TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all run clean

