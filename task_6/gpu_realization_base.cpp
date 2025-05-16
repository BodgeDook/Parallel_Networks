#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <cstring>  

#include <boost/program_options.hpp>

#include <omp.h>
#include <nvtx3/nvToolsExt.h>

namespace po = boost::program_options;

void initialize(double* A, double* Anew, size_t size) {
    std::memset(A, 0, size * size * sizeof(double));
    std::memset(Anew, 0, size * size * sizeof(double));

    A[0] = 10.0;
    A[size - 1] = 20.0;
    A[size * (size - 1)] = 30.0;
    A[size * size - 1] = 20.0;


    double top_left = A[0];
    double top_right = A[size - 1];
    double bottom_left = A[size * (size - 1)];
    double bottom_right = A[size * size - 1];

    for (int i = 1; i < size - 1; ++i) {
        A[i] = top_left + (top_right - top_left) * i / static_cast<double>(size - 1);             // Top
        A[size * (size - 1) + i] = bottom_left + (bottom_right - bottom_left) * i / static_cast<double>(size - 1); // Bottom
        A[size * i] = top_left + (bottom_left - top_left) * i / static_cast<double>(size - 1);     // Left
        A[size * i + size - 1] = top_right + (bottom_right - top_right) * i / static_cast<double>(size - 1); // Right
    }

    #pragma acc enter data copyin(A[:size*size], Anew[:size*size])
}

double calculate_next_grid(double* A, double* Anew, size_t size, bool check_error) {
    double error = 0.0;

    if (check_error) {
        #pragma acc parallel loop collapse(2) reduction(max:error) present(A, Anew)
        for (int i = 1; i < size - 1; ++i) {
            for (int j = 1; j < size - 1; ++j) {
                Anew[i * size + j] = 0.25 * (A[(i + 1) * size + j] + A[(i - 1) * size + j]
                                          + A[i * size + j - 1] + A[i * size + j + 1]);
                error = fmax(error, fabs(Anew[i * size + j] - A[i * size + j]));
            }
        }
    } else {
        #pragma acc parallel loop collapse(2) present(A, Anew)
        for (int i = 1; i < size - 1; ++i) {
            for (int j = 1; j < size - 1; ++j) {
                Anew[i * size + j] = 0.25 * (A[(i + 1) * size + j] + A[(i - 1) * size + j]
                                          + A[i * size + j - 1] + A[i * size + j + 1]);
            }
        }
    }

    return error;
}

void copy_matrix(double* A, double* Anew, size_t size) {
    #pragma acc parallel loop collapse(2) present(A, Anew)
    for (int i = 1; i < size - 1; ++i) {
        for (int j = 1; j < size - 1; ++j) {
            A[i * size + j] = Anew[i * size + j];
        }
    }
}

void deallocate(double* A, double* Anew) {
    #pragma acc exit data delete(A, Anew)
    free(A);
    free(Anew);
}

void print_grid(double* A, size_t size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            std::cout << std::fixed << std::setprecision(4) << A[i * size + j] << "  ";
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
        ("help", "Show option description")
        ("size", po::value<int>(&size)->default_value(256), "Grid size (NxN)")
        ("accuracy", po::value<double>(&accuracy)->default_value(1e-6), "Maximum allowed error")
        ("max_iterations", po::value<int>(&max_iterations)->default_value(1e6), "Maximum allowed iterations");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 0;
    }

    std::cout << "Start\n";

    double* A = static_cast<double*>(malloc(sizeof(double) * size * size));
    double* Anew = static_cast<double*>(malloc(sizeof(double) * size * size));

    double error = accuracy + 1.0;
    int iteration = 0;

    nvtxRangePushA("init");
    initialize(A, Anew, size);
    nvtxRangePop();

    std::cout << "Init\n";

    const auto start = std::chrono::steady_clock::now();

    nvtxRangePushA("while");
    while (error > accuracy && iteration < max_iterations) {
        nvtxRangePushA("calc");

        if (iteration % 1000 == 0) error = calculate_next_grid(A, Anew, size, true);
        else calculate_next_grid(A, Anew, size, false);

        nvtxRangePop();

        nvtxRangePushA("copy");
        copy_matrix(A, Anew, size);
        nvtxRangePop();

        ++iteration;
    }
    nvtxRangePop();

    std::cout << "End\n";

    const auto end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed_seconds = end - start;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Time:        " << elapsed_seconds.count() << " sec\n";
    std::cout << "Iterations:  " << iteration - 1 << "\n";
    std::cout << "Error value: " << error << "\n";

    deallocate(A, Anew);

    // #pragma acc update self(grid[0:size*size])
    // print_grid(grid, size);

    return 0;
}
