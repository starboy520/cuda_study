NVCC ?= nvcc
TARGET := overlap_pipeline
NVCCFLAGS := -O3 -std=c++17 -arch=sm_75 -lineinfo

all: $(TARGET)

$(TARGET): overlap_pipeline.cu ../../common/cuda_check.cuh
	$(NVCC) $(NVCCFLAGS) overlap_pipeline.cu -o $(TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all run clean
