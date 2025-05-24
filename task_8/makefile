TARGET = cub_realization

all: $(TARGET)

$(TARGET): cub_realization_v3.cu
	nvcc cub_realization_v3.cu -o $(TARGET) -O3 -std=c++17 -lboost_program_options -lcudart
	./cub_realization --size=128

profile: $(TARGET)
	nsys profile --trace=nvtx,cuda --stats=true ./$(TARGET) --max_iterations=50