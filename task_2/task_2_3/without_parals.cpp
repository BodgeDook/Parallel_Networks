#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>
#include <omp.h>

using namespace std;

constexpr double EPSILON = 1e-5;
constexpr double LEARNING_RATE = 1e-6;

using Vector = vector<double>;
using Matrix = vector<Vector>;

double calculateNorm(const Vector &vec) {
    double sum = 0.0;
    for (double value : vec) {
        sum += value * value;
    }
    return sqrt(sum);
}

Vector multiplyMatrixVector(const Matrix &A, const Vector &x) {
    int size = A.size();
    Vector result(size, 0.0);
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            result[i] += A[i][j] * x[j];
        }
    }
    return result;
}

Vector solveByIterations(const Matrix &A, const Vector &b) {
    int size = A.size();
    Vector x(size, 0.0);
    Vector Ax(size);
    Vector residual(size);
    double normRatio;
    
    do {
        Ax = multiplyMatrixVector(A, x);
        for (int i = 0; i < size; ++i) {
            residual[i] = Ax[i] - b[i];
        }
        
        normRatio = calculateNorm(residual) / calculateNorm(b);
        
        for (int i = 0; i < size; ++i) {
            x[i] -= LEARNING_RATE * residual[i];
        }
    } while (normRatio >= EPSILON);
    
    return x;
}

void constructLinearSystem(Matrix &coeffMatrix, Vector &rhsVector, int dimension) {
    coeffMatrix.resize(dimension, Vector(dimension, 1.0));
    for (int i = 0; i < dimension; ++i) {
        coeffMatrix[i][i] = 2.0;
    }
    rhsVector.assign(dimension, dimension + 1);
}

int readSystemSize() {
    int size;
    cout << "Your number of equations (>= 5000): ";
    cin >> size;
    return size;
}

double executeSolver(int size, Vector &solution) {
    Matrix A;
    Vector b;
    constructLinearSystem(A, b, size);
    
    double startTime = omp_get_wtime();
    solution = solveByIterations(A, b);
    return omp_get_wtime() - startTime;
}

void displayExecutionTime(double time) {
    printf("Elapsed time: %.5f seconds\n", time);
}

int main() {
    int systemSize = readSystemSize();
    Vector solution;
    double executionTime = executeSolver(systemSize, solution);
    displayExecutionTime(executionTime);
    
    return 0;
}