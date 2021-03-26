#include <chrono>
#include <cinttypes>
#include <iostream>
#include <random>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

double *randomMatrix(double const minimum_value, double const maximum_value, size_t const N) {
    std::random_device random_device;
    std::mt19937 generator(random_device());
    std::uniform_real_distribution<double> const uniform_real_distribution(minimum_value, maximum_value);
    double *const random_matrix = new double[N * N];
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            random_matrix[i * N + j] = uniform_real_distribution(generator);
        }
    }
    return random_matrix;
}

void cpuMultiplyMatrices(double const *const A, double const *const B, double *const C, size_t const N) {
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            double C_i_j = 0;
            for (size_t k = 0; k < N; ++k) {
                C_i_j += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = C_i_j;
        }
    }
}

__global__ void gpuMatrixMultiplicationKernel(double const *const A, double const *const B, double *const C, size_t const N) {
    size_t const i = blockIdx.y * blockDim.y + threadIdx.y;
    size_t const j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= N || j >= N) {
        return;
    }

    double C_i_j = 0;
    for (size_t k = 0; k < N; ++k) {
        C_i_j += A[i * N + k] * B[k * N + j];
    }
    C[i * N + j] = C_i_j;
}

void gpuMultiplyMatrices(double const *const A, double const *const B, double *const C, size_t const N, size_t const BLOCK_SIZE) {
    double *device_A;
    double *device_B;
    double *device_C;

    cudaMalloc(&device_A, N * N * sizeof(double));
    cudaMalloc(&device_B, N * N * sizeof(double));
    cudaMalloc(&device_C, N * N * sizeof(double));

    cudaMemcpy(device_A, A, N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_B, B, N * N * sizeof(double), cudaMemcpyHostToDevice);

    dim3 const block_dimensions(BLOCK_SIZE, BLOCK_SIZE);
    dim3 const grid_dimensions((N + block_dimensions.x - 1) / block_dimensions.x, (N + block_dimensions.y - 1) / block_dimensions.y);
    gpuMatrixMultiplicationKernel<<<grid_dimensions, block_dimensions>>>(device_A, device_B, device_C, N);
    cudaDeviceSynchronize();

    cudaMemcpy(C, device_C, N * N * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(device_A);
    cudaFree(device_B);
    cudaFree(device_C);
}

double maximumMatrixDeviation(double const *const A, double const *const B, size_t const N) {
    double maximum_matrix_deviation = 0;
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            maximum_matrix_deviation = std::max(maximum_matrix_deviation, std::abs(A[i * N + j] - B[i * N + j]));
        }
    }
    return maximum_matrix_deviation;
}

int main(int argc, char *argv []) {
    size_t const N = std::strtoumax(argv[1], nullptr, 10);
    std::cout << "N = " << N << std::endl;

    double const MINIMUM_VALUE = -10;
    double const MAXIMUM_VALUE = 10;
    double const *const A = randomMatrix(MINIMUM_VALUE, MAXIMUM_VALUE, N);
    double const *const B = randomMatrix(MINIMUM_VALUE, MAXIMUM_VALUE, N);

    double *const CPU_C = new double[N * N];
    {
        auto const start = std::chrono::high_resolution_clock::now();
        cpuMultiplyMatrices(A, B, CPU_C, N);
        auto const end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> const elapsed_seconds = end - start;
        std::cout << "CPU elapsed time = " << elapsed_seconds.count() << " seconds" << std::endl;
    }

    size_t const BLOCK_SIZE = 32;
    double *const GPU_C = new double[N * N];
    {
        auto const start = std::chrono::high_resolution_clock::now();
        gpuMultiplyMatrices(A, B, GPU_C, N, BLOCK_SIZE);
        auto const end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> const elapsed_seconds = end - start;
        std::cout << "GPU elapsed time = " << elapsed_seconds.count() << " seconds" << std::endl;
    }

    delete[] A;
    delete[] B;

    std::cout << "maximum CPU and GPU matrix deviation = " << maximumMatrixDeviation(CPU_C, GPU_C, N) << std::endl;

    delete[] CPU_C;
    delete[] GPU_C;
}
