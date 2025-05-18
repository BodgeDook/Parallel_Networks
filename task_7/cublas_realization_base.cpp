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

namespace po = boost::program_options;

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

    // corners
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

void freeGrids(double* a, double* b) {
    std::free(a);
    std::free(b);
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
            nvtxRangePushA("Compute");
            #pragma acc parallel loop collapse(2) present(grid, gridNew)
            for (int i = 1; i < size-1; ++i) {
                for (int j = 1; j < size-1; ++j) {
                    gridNew[i*size+j] = 0.25 * (
                        grid[(i+1)*size+j] +
                        grid[(i-1)*size+j] +
                        grid[i*size+(j-1)] +
                        grid[i*size+(j+1)]);
                }
            }
            nvtxRangePop();

            if (iteration % 500 == 0) {
                nvtxRangePushA("ErrorCalc");
                #pragma acc parallel loop collapse(2) present(grid, gridNew, errBuf)
                for (int i = 1; i < size-1; ++i) {
                    for (int j = 1; j < size-1; ++j) {
                        errBuf[i*size+j] = std::fabs(grid[i*size+j] - gridNew[i*size+j]);
                    }
                }
                #pragma acc host_data use_device(errBuf)
                {
                    cublasIdamax(*handle, size*size, errBuf, 1, &maxErrIndex);
                    cudaMemcpy(&errorVal,
                               &errBuf[maxErrIndex-1],
                               sizeof(double),
                               cudaMemcpyDeviceToHost);
                }
                nvtxRangePop();
            }

            nvtxRangePushA("CopyBack");
            #pragma acc parallel loop collapse(2) present(grid, gridNew)
            for (int i = 1; i < size-1; ++i) {
                for (int j = 1; j < size-1; ++j) {
                    grid[i*size+j] = gridNew[i*size+j];
                }
            }
            nvtxRangePop();

            ++iteration;
        }
    }
    nvtxRangePop();

    const auto t1 = std::chrono::steady_clock::now();
    std::chrono::duration<double> dt = t1 - t0;
    std::cout << "[Result] Time: " << dt.count() << " s | Iters: "
              << iteration << " | Err: " << errorVal << std::endl;

    acc_unmap_data(errBuf);
    cudaFree(errBuf);
}

int main(int argc, char* argv[]) {
    int size, maxIters;
    double accuracy;

    po::options_description opts("Options");
    opts.add_options()
        ("help,h", "Show help")
        ("size,s", po::value<int>(&size)->default_value(256), "Grid side length")
        ("accuracy,a", po::value<double>(&accuracy)->default_value(1e-6), "Convergence threshold")
        ("max-iters,m", po::value<int>(&maxIters)->default_value(1000000), "Max iterations");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, opts), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << opts << std::endl;
        return 0;
    }

    std::cout << "[Info] Starting Poisson solver with size=" << size
              << ", accuracy=" << accuracy
              << ", maxIters=" << maxIters << std::endl;

    double* grid    = (double*)std::malloc(sizeof(double) * size * size);
    double* gridNew = (double*)std::malloc(sizeof(double) * size * size);

    nvtxRangePushA("InitGrid");
    initializeGrid(grid, gridNew, size);
    nvtxRangePop();

    solve(grid, gridNew, size, accuracy, maxIters);

    /*
    nvtxRangePushA("SaveMatrix");
    saveMatrixToFile(grid, size, "result_" + std::to_string(size) + ".txt");
    nvtxRangePop();
    */

    freeGrids(grid, gridNew);

    std::cout << "[Info] Solver finished" << std::endl;
    return 0;
}
