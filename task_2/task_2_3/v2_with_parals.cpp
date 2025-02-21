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

double computeNorm(const Vector &v) {
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (size_t i = 0; i < v.size(); ++i) {
        sum += v[i] * v[i];
    }
    return sqrt(sum);
}

Vector simpleIterationMethod(const Matrix &A, const Vector &b, int n_threads) {
    int n = A.size();
    Vector x(n, 0.0), Ax(n), residual(n);
    double normB = computeNorm(b);

    do {
        #pragma omp parallel for num_threads(n_threads)
        for (int i = 0; i < n; ++i) {
            Ax[i] = 0.0;
            for (int j = 0; j < n; ++j) {
                Ax[i] += A[i][j] * x[j];
            }
            residual[i] = Ax[i] - b[i];
        }

        double residualNorm = computeNorm(residual);
        if (residualNorm / normB < EPSILON) {
            break;
        }

        #pragma omp parallel for num_threads(n_threads)
        for (int i = 0; i < n; ++i) {
            x[i] -= STEP_SIZE * residual[i];
        }

    } while (true);

    return x;
}

void generateSystem(Matrix &A, Vector &b, int N, int n_threads) {
    A.assign(N, Vector(N, 1.0));
    b.assign(N, N + 1);

    #pragma omp parallel for num_threads(n_threads)
    for (int i = 0; i < N; ++i) {
        A[i][i] = 2.0;
    }
}

void executeSolver(int N) {
    ofstream out("results_v2.txt");
    if (!out.is_open()) {
        cerr << "Error opening output file!" << endl;
        return;
    }

    for (int n_threads = 2; n_threads <= 80; n_threads++) {
        Matrix A;
        Vector b;

        generateSystem(A, b, N, n_threads);

        double start_time = omp_get_wtime();
        Vector solution = simpleIterationMethod(A, b, n_threads);
        double elapsed_time = omp_get_wtime() - start_time;

        printf("Threads: %d Execution time: %.5f seconds\n", n_threads, elapsed_time);
        out << n_threads << " " << elapsed_time << " " << 82.108765 / elapsed_time << "\n";
    }

    out.close();
    cout << "Output file written successfully!" << endl;
}

int main() {
    int N;
    cout << "Your number of equations for the v2 (>= 5000): ";
    cin >> N;
    executeSolver(N);

    return 0;
}