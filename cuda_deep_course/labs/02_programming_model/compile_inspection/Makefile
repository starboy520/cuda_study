NVCC ?= nvcc
TARGET := compile_inspection
NVCCFLAGS := -O2 -std=c++17 -lineinfo
ARCH ?= sm_75

all: $(TARGET)

$(TARGET): compile_inspection.cu ../../common/cuda_check.cuh
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) compile_inspection.cu -o $(TARGET)

ptx: compile_inspection.cu
	$(NVCC) $(NVCCFLAGS) -arch=compute_75 -code=compute_75 \
	  --ptx compile_inspection.cu -o compile_inspection.ptx

resource: compile_inspection.cu
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) -Xptxas=-v \
	  compile_inspection.cu -o $(TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) compile_inspection.ptx

.PHONY: all ptx resource run clean

