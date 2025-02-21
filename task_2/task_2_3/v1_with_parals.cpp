#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <omp.h>

using namespace std;

constexpr double EPSILON = 1e-5;
constexpr double STEP_SIZE = 1e-6;

using Vector = vector<double>;
using Matrix = vector<Vector>;

double computeNorm(const Vector &v, int num_threads) {
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum) num_threads(num_threads)
    for (size_t i = 0; i < v.size(); ++i) {
        sum += v[i] * v[i];
    }
    return sqrt(sum);
}

Vector iterativeSolver(const Matrix &A, const Vector &b, int num_threads) {
    int size = A.size();
    Vector x(size, 0.0), Ax(size), residual(size);
    double normB = computeNorm(b, num_threads);

    do {
        #pragma omp parallel for num_threads(num_threads)
        for (int i = 0; i < size; ++i) {
            Ax[i] = 0.0;
            for (int j = 0; j < size; ++j) {
                Ax[i] += A[i][j] * x[j];
            }
            residual[i] = Ax[i] - b[i];
        }

        double residualNorm = computeNorm(residual, num_threads);
        if (residualNorm / normB < EPSILON) {
            break;
        }

        #pragma omp parallel for num_threads(num_threads)
        for (int i = 0; i < size; ++i) {
            x[i] -= STEP_SIZE * residual[i];
        }
    } while (true);

    return x;
}

void generateSystem(Matrix &A, Vector &b, int size) {
    A.assign(size, Vector(size, 1.0));
    #pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        A[i][i] = 2.0;
    }
    b.assign(size, size + 1);
}

void executeSolver(int N) {
    ofstream outputFile("results_v1.txt");
    if (!outputFile) {
        cerr << "Error opening output file!" << endl;
        return;
    }
    
    for (int threads = 2; threads <= 80; threads++) {
        Matrix A;
        Vector b;
        generateSystem(A, b, N);
        
        double startTime = omp_get_wtime();
        Vector solution = iterativeSolver(A, b, threads);
        double elapsedTime = omp_get_wtime() - startTime;
        
        printf("Threads: %d Execution Time: %.5f seconds\n", threads, elapsedTime);
        outputFile << threads << " " << elapsedTime << " " << 82.108765 / elapsedTime << "\n";
    }
    outputFile.close();
    cout << "Output file written successfully!" << endl;
}

int main() {
    int N;
    cout << "Your number of equations for the v1 (>= 5000): ";
    cin >> N;
    executeSolver(N);
    
    return 0;
}