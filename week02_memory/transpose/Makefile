NVCC ?= nvcc
NVCCFLAGS = -O3 -arch=sm_75 -std=c++14
TARGET = transpose

.PHONY: all clean run

all: $(TARGET)

$(TARGET): transpose.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
