NVCC ?= nvcc
TARGET := device_query
NVCCFLAGS := -O2 -std=c++17

all: $(TARGET)

$(TARGET): device_query.cu ../../common/cuda_check.cuh
	$(NVCC) $(NVCCFLAGS) device_query.cu -o $(TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all run clean

