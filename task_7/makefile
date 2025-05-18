gpu_cublas:
	pgc++ -o cublas_realization_v4 -lboost_program_options -acc=gpu -fast \
		-I/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/cuda/12.3/include \
		-I/usr/local/cuda/include \
		-L/usr/local/cuda/lib64 \
		-I/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/math_libs/12.3/targets/x86_64-linux/include \
		-L/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/math_libs/12.3/targets/x86_64-linux/lib \
		-L/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/cuda/12.3/lib64 \
		-lcublas -lcudart cublas_realization_v4.cpp

	./cublas_realization_v4 --size=1024 --accuracy=0.000001 --max-iters=1000000

profile:
	pgc++ -o cublas_realization_v4 -lboost_program_options -acc=gpu -fast \
		-I/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/cuda/12.3/include \
		-I/usr/local/cuda/include \
		-L/usr/local/cuda/lib64 \
		-I/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/math_libs/12.3/targets/x86_64-linux/include \
		-L/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/math_libs/12.3/targets/x86_64-linux/lib \
		-L/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/cuda/12.3/lib64 \
		-lcublas -lcudart cublas_realization_v4.cpp

	nsys profile --trace=nvtx,cuda,openacc --stats=true ./cublas_realization_v4 --size=512 --accuracy=0.000001 --max-iters=100