cpu_sequential:
	pgc++ -o cpu_onecore -lboost_program_options -acc=host -Minfo=all -I/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/cuda/12.3/include cpu_realization.cpp
	./cpu_onecore --size=512 --accuracy=0.000001 --max_iterations=1000000

cpu_multicore:
	pgc++ -o cpu_multicore -lboost_program_options -acc=multicore -Minfo=all -I/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/cuda/12.3/include cpu_realization.cpp
	./cpu_multicore --size=1024 --accuracy=0.000001 --max_iterations=1000000

gpu:
	pgc++ -o gpu_realization_v4 -lboost_program_options -acc=gpu -Minfo=all -I/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/cuda/12.3/include gpu_realization_v4.cpp
	./gpu_realization_v4 --size=1024 --accuracy=0.000001 --max_iterations=1000000

profile:
	pgc++ -o gpu_realization_v4 -lboost_program_options -acc=gpu -Minfo=all -I/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/cuda/12.3/include gpu_realization_v4.cpp
	nsys profile --trace=nvtx,cuda,openacc --stats=true ./gpu_realization_v4 --size=512 --accuracy=0.0001 --max_iterations=100