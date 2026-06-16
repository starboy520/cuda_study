NVCC ?= nvcc
NVCCFLAGS = -O3 -arch=sm_75 -std=c++14
TARGET = vec_add

.PHONY: all clean run

all: $(TARGET)

$(TARGET): vec_add.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
