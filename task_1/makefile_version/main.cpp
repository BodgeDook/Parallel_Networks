#include <iostream>
#include <cmath>
#include <numeric>
#include <vector>

template <typename T>
T calculate_sin_sum(size_t size) {
    constexpr T two_pi = 2 * M_PI;
    std::vector<T> sin_values(size);

    for (size_t i = 0; i < size; ++i) {
        sin_values[i] = std::sin(two_pi * static_cast<T>(i) / size);
    }

    T sum = std::accumulate(sin_values.begin(), sin_values.end(), static_cast<T>(0));
    
    return sum;
}

int main() {
#ifdef USE_DOUBLE
    constexpr size_t array_size = 10'000'000;
    double sum = calculate_sin_sum<double>(array_size);
    std::cout << "Sum (double): " << sum << std::endl;
#else
    constexpr size_t array_size = 10'000'000;
    float sum = calculate_sin_sum<float>(array_size);
    std::cout << "Sum (float): " << sum << std::endl;
#endif
    return 0;
}

/*
Commands for makefile:
1) make ARRAY_TYPE=float/double
2) make clean
*/

/*
The results are:
Sum (double): 4.89582e-11
Sum (float): 0.291951
*/