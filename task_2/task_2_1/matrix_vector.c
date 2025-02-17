#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 20000  // 40000

void matrix_vector_multiplication(double *matrix, double *vector, double *result, int size, int num_threads) {
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < size; i++) {
        double sum = 0.0;
        for (int j = 0; j < size; j++) {
            sum += matrix[i * size + j] * vector[j];
        }
        result[i] = sum;
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <num_threads>\n", argv[0]);
        return 1;
    }
    int num_threads = atoi(argv[1]);
    
    double *matrix = (double *)malloc(N * N * sizeof(double));
    double *vector = (double *)malloc(N * sizeof(double));
    double *result = (double *)malloc(N * sizeof(double));
    
    if (!matrix || !vector || !result) {
        printf("Memory allocation failed\n");
        return 1;
    }
    
    // initialization:
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < N; i++) {
        vector[i] = 1.0;
        for (int j = 0; j < N; j++) {
            matrix[i * N + j] = 1.0; // or other values
        }
    }
    
    double start_time = omp_get_wtime();
    matrix_vector_multiplication(matrix, vector, result, N, num_threads);
    double end_time = omp_get_wtime();
    
    printf("Time taken: %f seconds\n", end_time - start_time);
    
    free(matrix);
    free(vector);
    free(result);
    
    return 0;
}