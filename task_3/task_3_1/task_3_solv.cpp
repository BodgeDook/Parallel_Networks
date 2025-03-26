#include <iostream>
#include <vector>
#include <chrono>
#include <thread>

const int M = 40000;
const int N = 40000;

void compute_matrix_vector(const std::vector<double>& matrix, 
                         const std::vector<double>& vec, 
                         std::vector<double>& result, 
                         int begin, int finish) {
    int i = begin;
    while (i < finish) {
        result[i] = 0.0;
        int j = 0;
        while (j < N) {
            result[i] += matrix[i * N + j] * vec[j];
            ++j;
        }
        ++i;
    }
}

void fill_matrix(std::vector<double>& matrix, int begin, int finish) {
    int i = begin;
    while (i < finish) {
        int j = 0;
        while (j < N) {
            matrix[i * N + j] = static_cast<double>(i + j);
            ++j;
        }
        ++i;
    }
}

void fill_vector(std::vector<double>& vec, int begin, int finish) {
    int j = begin;
    while (j < finish) {
        vec[j] = static_cast<double>(j);
        ++j;
    }
}

void execute_parallel(int thread_count) {
    std::vector<double> matrix(M * N);
    std::vector<double> input_vec(N);
    std::vector<double> output_vec(M);
    
    std::vector<std::thread> workers;
    workers.reserve(thread_count);

    int portion = M / thread_count;
    int idx = 0;
    while (idx < thread_count) {
        int start = idx * portion;
        int end = (idx == thread_count - 1) ? M : start + portion;
        workers.emplace_back(fill_matrix, std::ref(matrix), start, end);
        ++idx;
    }
    for (auto& t : workers) {
        t.join();
    }
    workers.clear();

    portion = N / thread_count;
    idx = 0;
    while (idx < thread_count) {
        int start = idx * portion;
        int end = (idx == thread_count - 1) ? N : start + portion;
        workers.emplace_back(fill_vector, std::ref(input_vec), start, end);
        ++idx;
    }
    for (auto& t : workers) {
        t.join();
    }
    workers.clear();

    auto start_time = std::chrono::high_resolution_clock::now();
    
    portion = M / thread_count;
    idx = 0;
    while (idx < thread_count) {
        int start = idx * portion;
        int end = (idx == thread_count - 1) ? M : start + portion;
        workers.emplace_back(compute_matrix_vector, 
                          std::cref(matrix), 
                          std::cref(input_vec), 
                          std::ref(output_vec), 
                          start, end);
        ++idx;
    }
    for (auto& t : workers) {
        t.join();
    }
    workers.clear();

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;
    
    std::cout << "Time taken with " << thread_count 
              << " threads: " << duration.count() << " seconds\n";
}

int main() {
    std::cout << "Matrix-vector multiplication (output[M] = matrix[M,N] * input[N])\n";
    std::cout << "M = " << M << ", N = " << N << "\n";
    std::cout << "Memory usage: " 
              << ((M * N + M + N) * sizeof(double)) / (1024 * 1024) 
              << " MiB\n";

    std::vector<int> thread_counts = {1, 2, 4, 7, 8, 16, 20, 40};
    size_t i = 0;
    while (i < thread_counts.size()) {
        execute_parallel(thread_counts[i]);
        ++i;
    }

    return 0;
}