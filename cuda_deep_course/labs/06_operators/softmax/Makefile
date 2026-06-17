NVCC ?= nvcc
TARGET := softmax
NVCCFLAGS := -O3 -std=c++17 -arch=sm_75 -lineinfo

all: $(TARGET)

$(TARGET): softmax.cu ../../common/cuda_check.cuh
	$(NVCC) $(NVCCFLAGS) softmax.cu -o $(TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all run clean
