#include <boost/program_options.hpp>
#include <cublas_v2.h>
#include <nvtx3/nvToolsExt.h>
#include <openacc.h>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <utility>

namespace po = boost::program_options;

void freeGrids(double* a, double* b) {
    std::free(a);
    std::free(b);
}

struct CublasHandleDeleter {
    void operator()(cublasHandle_t* h) const {
        if (h) {
            cublasDestroy(*h);
            delete h;
        }
    }
};
using CublasHandlePtr = std::unique_ptr<cublasHandle_t, CublasHandleDeleter>;

CublasHandlePtr createCublasHandle() {
    auto h = new cublasHandle_t;
    if (cublasCreate(h) != CUBLAS_STATUS_SUCCESS) {
        delete h;
        throw std::runtime_error("cuBLAS handle creation failed");
    }
    return CublasHandlePtr(h);
}

void initializeGrid(double* __restrict grid,
                    double* __restrict gridNew,
                    const int size)
{
    std::memset(grid,    0, sizeof(double) * size * size);
    std::memset(gridNew, 0, sizeof(double) * size * size);

    grid[0]                       = 10.0;
    grid[size - 1]                = 20.0;
    grid[size*(size-1)]           = 30.0;
    grid[size*size - 1]           = 20.0;

    const double tl = grid[0];
    const double tr = grid[size - 1];
    const double bl = grid[size*(size-1)];
    const double br = grid[size*size - 1];

    for (int i = 1; i < size-1; ++i) {
        double alpha = static_cast<double>(i) / (size - 1);
        grid[i]                     = tl + (tr - tl) * alpha;
        grid[size*(size-1) + i]    = bl + (br - bl) * alpha;
        grid[size*i]                = tl + (bl - tl) * alpha;
        grid[size*i + size - 1]    = tr + (br - tr) * alpha;
    }
}

void solve(double* __restrict grid,
           double* __restrict gridNew,
           const int size,
           const double accuracy,
           const int maxIters)
{
    double* errBuf = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&errBuf), sizeof(double) * size * size);
    acc_map_data(errBuf, errBuf, sizeof(double) * size * size);

    auto handle = createCublasHandle();
    double errorVal = accuracy + 1.0;
    int iteration   = 0;
    int maxErrIndex = 0;

    const auto t0 = std::chrono::steady_clock::now();
    nvtxRangePushA("SolverLoop");

    #pragma acc data copy(grid[0:size*size], gridNew[0:size*size]) present(errBuf[0:size*size])
    {
        while (errorVal > accuracy && iteration < maxIters) {
            nvtxRangePushA("Compute+Error");
            #pragma acc parallel loop collapse(2) present(grid, gridNew, errBuf)
            for (int i = 1; i < size - 1; ++i) {
                for (int j = 1; j < size - 1; ++j) {
                    int idx = i * size + j;
                    double newVal = 0.25 * (
                        grid[idx + size] +
                        grid[idx - size] +
                        grid[idx + 1]    +
                        grid[idx - 1]
                    );
                    gridNew[idx] = newVal;
                    if (iteration % 500 == 0) {
                        errBuf[idx] = std::fabs(grid[idx] - newVal);
                    }
                }
            }
            nvtxRangePop();

            if (iteration % 500 == 0) {
                nvtxRangePushA("ErrorReduce");
                #pragma acc host_data use_device(errBuf)
                {
                    cublasIdamax(*handle, size * size, errBuf, 1, &maxErrIndex);
                    cudaMemcpy(&errorVal,
                               &errBuf[maxErrIndex - 1],
                               sizeof(double),
                               cudaMemcpyDeviceToHost);
                }
                nvtxRangePop();
            }

            std::swap(grid, gridNew);
            ++iteration;
        }
    }
    nvtxRangePop();

    const auto t1 = std::chrono::steady_clock::now();
    std::chrono::duration<double> dt = t1 - t0;
    std::cout << "[Result] Time: " << dt.count()
              << " s | Iters: " << iteration
              << " | Err: " << errorVal << std::endl;

    acc_unmap_data(errBuf);
    cudaFree(errBuf);
}


int main(int argc, char* argv[]) {
    int size;
    double accuracy;
    int maxIters;

    po::options_description desc("Options");
    desc.add_options()
        ("help", "Show help message")
        ("size", po::value<int>(&size)->default_value(256), "Grid size (NxN)")
        ("accuracy", po::value<double>(&accuracy)->default_value(1e-6), "Maximum allowed error")
        ("max-iters", po::value<int>(&maxIters)->default_value(1000000), "Maximum allowed iterations");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 0;
    }

    std::cout << "[Info] Starting Poisson solver with size=" << size
              << ", accuracy=" << accuracy
              << ", maxIters=" << maxIters << std::endl;

    double* grid    = static_cast<double*>(std::malloc(sizeof(double) * size * size));
    double* gridNew = static_cast<double*>(std::malloc(sizeof(double) * size * size));

    nvtxRangePushA("InitGrid");
    initializeGrid(grid, gridNew, size);
    nvtxRangePop();

    solve(grid, gridNew, size, accuracy, maxIters);

    freeGrids(grid, gridNew);

    std::cout << "[Info] Solver finished" << std::endl;
    return 0;
}
