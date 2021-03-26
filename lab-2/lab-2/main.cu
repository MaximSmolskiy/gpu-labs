#include <chrono>
#include <cinttypes>
#include <iostream>
#include <random>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

double *randomMatrix(double const minimum_value, double const maximum_value, size_t const n) {
    std::random_device random_device;
    std::mt19937 generator(random_device());
    std::uniform_real_distribution<double> const uniform_real_distribution(minimum_value, maximum_value);
    double *const random_matrix = new double[n * n];
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            random_matrix[i * n + j] = uniform_real_distribution(generator);
        }
    }
    return random_matrix;
}

void cpuMultiplyMatrices(double const *const a, double const *const b, double *const c, size_t const n) {
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double c_i_j = 0;
            for (size_t k = 0; k < n; ++k) {
                c_i_j += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = c_i_j;
        }
    }
}

__global__ void gpuMatrixMultiplicationKernel(double const *const a, double const *const b, double *const c, size_t const n) {
    size_t const i = blockIdx.y * blockDim.y + threadIdx.y;
    size_t const j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n || j >= n) {
        return;
    }

    double c_i_j = 0;
    for (size_t k = 0; k < n; ++k) {
        c_i_j += a[i * n + k] * b[k * n + j];
    }
    c[i * n + j] = c_i_j;
}

void gpuMultiplyMatrices(double const *const a, double const *const b, double *const c, size_t const n, size_t const block_size) {
    double *device_a;
    double *device_b;
    double *device_c;

    cudaMalloc(&device_a, n * n * sizeof(double));
    cudaMalloc(&device_b, n * n * sizeof(double));
    cudaMalloc(&device_c, n * n * sizeof(double));

    cudaMemcpy(device_a, a, n * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, b, n * n * sizeof(double), cudaMemcpyHostToDevice);

    dim3 const block_dimensions(block_size, block_size);
    dim3 const grid_dimensions((n + block_dimensions.x - 1) / block_dimensions.x, (n + block_dimensions.y - 1) / block_dimensions.y);
    gpuMatrixMultiplicationKernel<<<grid_dimensions, block_dimensions>>>(device_a, device_b, device_c, n);

    cudaMemcpy(c, device_c, n * n * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);
}

double maximumMatrixDeviation(double const *const a, double const *const b, size_t const n) {
    double maximum_matrix_deviation = 0;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            maximum_matrix_deviation = std::max(maximum_matrix_deviation, std::abs(a[i * n + j] - b[i * n + j]));
        }
    }
    return maximum_matrix_deviation;
}

int main(int argc, char *argv []) {
    size_t const n = std::strtoumax(argv[1], nullptr, 10);
    std::cout << "n = " << n << std::endl;

    double const MINIMUM_VALUE = -10;
    double const MAXIMUM_VALUE = 10;
    double const *const a = randomMatrix(MINIMUM_VALUE, MAXIMUM_VALUE, n);
    double const *const b = randomMatrix(MINIMUM_VALUE, MAXIMUM_VALUE, n);

    double *const cpu_c = new double[n * n];
    {
        auto const start = std::chrono::high_resolution_clock::now();
        cpuMultiplyMatrices(a, b, cpu_c, n);
        auto const end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> const elapsed_seconds = end - start;
        std::cout << "CPU elapsed time = " << elapsed_seconds.count() << " seconds" << std::endl;
    }

    size_t const BLOCK_SIZE = 32;
    double *const gpu_c = new double[n * n];
    {
        auto const start = std::chrono::high_resolution_clock::now();
        gpuMultiplyMatrices(a, b, gpu_c, n, BLOCK_SIZE);
        auto const end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> const elapsed_seconds = end - start;
        std::cout << "GPU elapsed time = " << elapsed_seconds.count() << " seconds" << std::endl;
    }

    delete[] a;
    delete[] b;

    std::cout << "maximum CPU and GPU matrix deviation = " << maximumMatrixDeviation(cpu_c, gpu_c, n) << std::endl;

    delete[] cpu_c;
    delete[] gpu_c;
}
