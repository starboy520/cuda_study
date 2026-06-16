NVCC ?= nvcc
TARGET := memory_access
NVCCFLAGS := -O2 -std=c++17

all: $(TARGET)

$(TARGET): memory_access.cu ../../common/cuda_check.cuh
	$(NVCC) $(NVCCFLAGS) memory_access.cu -o $(TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all run clean

