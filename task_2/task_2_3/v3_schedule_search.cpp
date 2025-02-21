#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <omp.h>

using namespace std;

const double EPSILON = 1e-5;
const double STEP_SIZE = 1e-6;

using Vector = vector<double>;
using Matrix = vector<Vector>;

double norm(const Vector &v) {
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (size_t i = 0; i < v.size(); ++i) {
        sum += v[i] * v[i];
    }
    return sqrt(sum);
}

Vector simpleIterationMethod(const Matrix &A, const Vector &b, int n_threads, 
                            const string& schedule_type, int chunk_size) {
    const int n = A.size();
    Vector x(n, 0.0);
    Vector Ax(n), r(n);
    bool stop = false;

    #pragma omp parallel num_threads(n_threads)
    {
        while (!stop) {
            #pragma omp single
            {
                Ax.assign(n, 0.0);
                r.assign(n, 0.0);
            }

            if (schedule_type == "static") {
                #pragma omp for schedule(static, chunk_size)
                for (int i = 0; i < n; ++i) {
                    for (int j = 0; j < n; ++j) {
                        Ax[i] += A[i][j] * x[j];
                    }
                }
            } else if (schedule_type == "dynamic") {
                #pragma omp for schedule(dynamic, chunk_size)
                for (int i = 0; i < n; ++i) {
                    for (int j = 0; j < n; ++j) {
                        Ax[i] += A[i][j] * x[j];
                    }
                }
            } else {
                #pragma omp for schedule(guided, chunk_size)
                for (int i = 0; i < n; ++i) {
                    for (int j = 0; j < n; ++j) {
                        Ax[i] += A[i][j] * x[j];
                    }
                }
            }

            #pragma omp for
            for (int i = 0; i < n; ++i) {
                r[i] = Ax[i] - b[i];
            }

            #pragma omp single
            {
                stop = (norm(r) / norm(b) < EPSILON);
            }

            if (stop) break;

            #pragma omp for
            for (int i = 0; i < n; ++i) {
                x[i] -= STEP_SIZE * r[i];
            }
        }
    }
    return x;
}

pair<Matrix, Vector> initializeSystem(int N, int n_threads) {
    Matrix A(N, Vector(N, 1.0));
    Vector b(N, N + 1.0);
    
    #pragma omp parallel for num_threads(n_threads)
    for (int i = 0; i < N; ++i) {
        A[i][i] = 2.0;
    }
    return {A, b};
}

void runExperiment(int n_threads, const string& schedule, int chunk_size, 
                  int N, ofstream &out) {
    auto [A, b] = initializeSystem(N, n_threads);
    
    const double start = omp_get_wtime();
    Vector solution = simpleIterationMethod(A, b, n_threads, schedule, chunk_size);
    const double duration = omp_get_wtime() - start;

    printf("Threads: %3d | Schedule: %-7s | Chunk: %3d | Time: %.5f sec\n", 
          n_threads, schedule.c_str(), chunk_size, duration);
    out << n_threads << "\t" << schedule << "\t" << chunk_size << "\t"
        << duration << "\t" << 82.108765 / duration << "\n";
}

void runExperiments(int N, ofstream &out) {
    const vector<string> schedules = {"static", "dynamic", "guided", "auto"};
    const vector<int> chunk_sizes = {1, 4, 8, 16};

    out << "# threads\tschedule\tchunk_size\ttime\tspeedup\n";
    
    for (int n_threads = 2; n_threads <= 80; n_threads += 2) {
        for (const auto& schedule : schedules) {
            for (int chunk_size : chunk_sizes) {
                runExperiment(n_threads, schedule, chunk_size, N, out);
            }
        }
    }
}

int main() {
    int N;
    cout << "Your number of equations (>= 5000): ";
    cin >> N;

    ofstream out("results_v3.txt");
    if (!out.is_open()) {
        cerr << "Failed to open output file!" << endl;
        return 1;
    }

    runExperiments(N, out);
    out.close();

    cout << "File has been written" << endl;
    return 0;
}