NVCC ?= nvcc
TARGET := index_mapping
NVCCFLAGS := -O2 -std=c++17

all: $(TARGET)

$(TARGET): index_mapping.cu ../../common/cuda_check.cuh
	$(NVCC) $(NVCCFLAGS) index_mapping.cu -o $(TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all run clean
