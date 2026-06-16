NVCC ?= nvcc
TARGET := cuda_graph
NVCCFLAGS := -O3 -std=c++17 -arch=sm_75 -lineinfo

all: $(TARGET)

$(TARGET): cuda_graph.cu ../../common/cuda_check.cuh
	$(NVCC) $(NVCCFLAGS) cuda_graph.cu -o $(TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all run clean
