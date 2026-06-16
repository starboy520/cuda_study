NVCC ?= nvcc
TARGET := async_errors
NVCCFLAGS := -O2 -std=c++17 -lineinfo

all: $(TARGET)

$(TARGET): async_errors.cu ../../common/cuda_check.cuh
	$(NVCC) $(NVCCFLAGS) async_errors.cu -o $(TARGET)

run: $(TARGET)
	./$(TARGET)

illegal: $(TARGET)
	./$(TARGET) --illegal-access

clean:
	rm -f $(TARGET)

.PHONY: all run illegal clean

