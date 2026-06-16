NVCC ?= nvcc
TARGET := event_timing
NVCCFLAGS := -O2 -std=c++17

all: $(TARGET)

$(TARGET): event_timing.cu ../../common/cuda_check.cuh
	$(NVCC) $(NVCCFLAGS) event_timing.cu -o $(TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all run clean

