#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <cstring>
#include <boost/program_options.hpp>
#include <nvtx3/nvToolsExt.h>

namespace po = boost::program_options;

void initialize_grid(double* grid, double* new_grid, size_t size) {
    std::memset(grid, 0, size * size * sizeof(double));
    std::memset(new_grid, 0, size * size * sizeof(double));

    grid[0]           = 10.0;
    grid[size - 1]       = 20.0;
    grid[size * (size - 1)] = 30.0;
    grid[size * size - 1]   = 20.0;

    double tl = grid[0];
    double tr = grid[size - 1];
    double bl = grid[size * (size - 1)];
    double br = grid[size * size - 1];

    for (int i = 1; i < size - 1; ++i) {
        grid[i]               = tl + (tr - tl) * i / (size - 1);           
        grid[size * (size - 1) + i] = bl + (br - bl) * i / (size - 1);           
        grid[size * i]           = tl + (bl - tl) * i / (size - 1);           
        grid[size * i + size - 1]   = tr + (br - tr) * i / (size - 1);          
    }

    #pragma acc enter data copyin(grid[0:size*size], new_grid[0:size*size]) 
        
}

double calculate_next_grid(double* grid, double* new_grid, size_t size) {
    double error = 0.0;

    #pragma acc parallel loop collapse(2) reduction(max:error) present(grid, new_grid) 
    for (int i = 1; i < size - 1; ++i) {                                                  
        for (int j = 1; j < size - 1; ++j) {
            int idx = i * size + j;
            new_grid[idx] = 0.25 * (
                grid[(i + 1) * size + j] +
                grid[(i - 1) * size + j] +
                grid[i * size + j - 1] +
                grid[i * size + j + 1]
            );
            error = fmax(error, fabs(new_grid[idx] - grid[idx]));
        }
    }
    
    return error;
}

void copy_grid(double* grid, const double* new_grid, size_t size) {
    #pragma acc parallel loop collapse(2) present(grid, new_grid)
    for (int i = 1; i < size - 1; ++i) {
        for (int j = 1; j < size - 1; ++j) {
            grid[i * size + j] = new_grid[i * size + j];
        }
    }
}

void deallocate(double* grid, double* new_grid) {
    #pragma acc exit data delete(grid[0:0], new_grid[0:0])  
    free(grid);                                             
    free(new_grid);                                        
}

void print_grid(const double* grid, size_t size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            std::cout << std::setprecision(4) << grid[i * size + j] << "  ";
        }
        std::cout << '\n';
    }
}

int main(int argc, char* argv[]) {
    int size;
    double accuracy;
    int max_iterations;

    po::options_description desc("Options");
    desc.add_options()
        ("help", "show help")
        ("size", po::value<int>(&size)->default_value(256), "Grid size (NxN)")
        ("accuracy", po::value<double>(&accuracy)->default_value(1e-6), "Convergence threshold")
        ("max_iterations", po::value<int>(&max_iterations)->default_value(1e6), "Max iterations");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 0;
    }

    std::cout << "Start\n";

    double* grid     = (double*)malloc(size * size * sizeof(double));
    double* new_grid = (double*)malloc(size * size * sizeof(double));

    double error = accuracy + 1.0;
    int iter = 0;

    nvtxRangePushA("init");
    initialize_grid(grid, new_grid, size);
    nvtxRangePop();

    std::cout << "Init\n";

    auto start = std::chrono::steady_clock::now();

    nvtxRangePushA("while");
    while (error > accuracy && iter < max_iterations) {
        nvtxRangePushA("calc");

        error = calculate_next_grid(grid, new_grid, size);
        nvtxRangePop();

        nvtxRangePushA("copy");
        copy_grid(grid, new_grid, size);
        nvtxRangePop();

        iter++;
    }
    nvtxRangePop();

    std::cout << "End\n";

    auto end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();

    std::cout << "Time:       " << elapsed << " sec\n";
    std::cout << "Iterations: " << iter << "\n";
    std::cout << "Final error:" << error << "\n";

    // #pragma acc update self(grid[0:size*size])
    // print_grid(grid, size);

    deallocate(grid, new_grid);

    return 0;
}
