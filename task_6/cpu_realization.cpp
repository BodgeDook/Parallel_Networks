#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <boost/program_options.hpp>
#include <omp.h>
#include <nvtx3/nvToolsExt.h>

namespace po = boost::program_options;

void initialize(double* restrict A, double* restrict Anew, size_t size) {
    std::fill(A, A + size * size, 0.0);
    std::fill(Anew, Anew + size * size, 0.0);

    A[0] = 10.0;
    A[size - 1] = 20.0;
    A[size * (size - 1)] = 30.0;
    A[size * size - 1] = 20.0;

    double top_left = A[0];
    double top_right = A[size - 1];
    double bottom_left = A[size * (size - 1)];
    double bottom_right = A[size * size - 1];

    double inv_size = 1.0 / static_cast<double>(size - 1);
    for (size_t i = 1; i < size - 1; ++i) {
        A[i] = top_left + (top_right - top_left) * i * inv_size; // Top
        A[size * (size - 1) + i] = bottom_left + (bottom_right - bottom_left) * i * inv_size; // Bottom
        A[size * i] = top_left + (bottom_left - top_left) * i * inv_size; // Left
        A[size * i + size - 1] = top_right + (bottom_right - top_right) * i * inv_size; // Right
    }
}

double calculate_next_grid(const double* restrict A, double* restrict Anew, size_t size) {
    double error = 0.0;

    #pragma acc parallel loop reduction(max:error)
    for (size_t i = 1; i < size - 1; ++i) {
        #pragma acc loop
        for (size_t j = 1; j < size - 1; ++j) {
            Anew[i * size + j] = 0.25 * (A[(i + 1) * size + j] + A[(i - 1) * size + j]
                                       + A[i * size + (j - 1)] + A[i * size + (j + 1)]);
            error = fmax(error, fabs(Anew[i * size + j] - A[i * size + j]));
        }
    }
    return error;
}

void copy_matrix(const double* restrict Anew, double* restrict A, size_t size) {
    #pragma acc parallel loop
    for (size_t i = 1; i < size - 1; ++i) {
        #pragma acc loop
        for (size_t j = 1; j < size - 1; ++j) {
            A[i * size + j] = Anew[i * size + j];
        }
    }
}

void print_grid(const double* A, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < size; ++j) {
            std::cout << std::setprecision(4) << A[i * size + j] << "  ";
        }
        std::cout << '\n';
    }
}

template<typename Func>
double measure_execution_time(Func&& func) {
    const auto start = std::chrono::steady_clock::now();
    func();
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(end - start).count();
}

int main(int argc, char* argv[]) {
    size_t size = 128;
    double accuracy = 1e-6;
    size_t max_iterations = 1e6;

    po::options_description desc("Options");
    desc.add_options()
        ("help", "Show option description")
        ("size", po::value<size_t>(&size)->default_value(256), "Grid size (NxN)")
        ("accuracy", po::value<double>(&accuracy)->default_value(1e-6), "Maximum allowed error")
        ("max_iterations", po::value<size_t>(&max_iterations)->default_value(1e6), "Maximum allowed iterations");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 1;
    }

    std::cout << "Program started!\n\n";

    std::vector<double> A(size * size, 0.0);
    std::vector<double> Anew(size * size, 0.0);

    double error = accuracy + 1.0;
    size_t iteration = 0;

    nvtxRangePushA("init");
    initialize(A.data(), Anew.data(), size);
    nvtxRangePop();

    double elapsed_seconds = measure_execution_time([&] {
        nvtxRangePushA("while");
        while (error > accuracy && iteration < max_iterations) {
            nvtxRangePushA("calc");
            error = calculate_next_grid(A.data(), Anew.data(), size);
            nvtxRangePop();

            nvtxRangePushA("copy");
            copy_matrix(Anew.data(), A.data(), size);
            nvtxRangePop();

            ++iteration;
        }
        nvtxRangePop();
    });

    // #pragma acc update self(A[0:size*size])
    print_grid(A.data(), size);

    std::cout << std::fixed << std::setprecision(6)
              << "Time: " << elapsed_seconds << " sec\n"
              << "Iterations: " << iteration << "\n"
              << "Error value: " << error << std::endl;

    return 0;
}