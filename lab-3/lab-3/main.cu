#include <chrono>
#include <cinttypes>
#include <iostream>
#include <random>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

size_t const BLOCK_SIZE = 32;

float *randomMatrix(float const minimum_value, float const maximum_value, size_t const n) {
    std::random_device random_device;
    std::mt19937 generator(random_device());
    std::uniform_real_distribution<float> const uniform_real_distribution(minimum_value, maximum_value);
    float *const random_matrix = new float[n * n];
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            random_matrix[i * n + j] = uniform_real_distribution(generator);
        }
    }
    return random_matrix;
}

__global__ void gpuSharedMemoryMatrixMultiplicationKernel(float const *const a, float const *const b, float *const c, size_t const n) {
    size_t const i = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    size_t const j = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    float c_i_j = 0;
    for (size_t submatrix_index = 0; submatrix_index * BLOCK_SIZE < n; ++submatrix_index) {
        __shared__ float submatrix_a[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float submatrix_b[BLOCK_SIZE][BLOCK_SIZE];

        submatrix_a[threadIdx.y][threadIdx.x] = 0;
        submatrix_b[threadIdx.y][threadIdx.x] = 0;

        size_t const submatrix_a_j = submatrix_index * BLOCK_SIZE + threadIdx.x;
        if (i < n && submatrix_a_j < n) {
            submatrix_a[threadIdx.y][threadIdx.x] = a[i * n + submatrix_a_j];
        }

        size_t const submatrix_b_i = submatrix_index * BLOCK_SIZE + threadIdx.y;
        if (submatrix_b_i < n && j < n) {
            submatrix_b[threadIdx.y][threadIdx.x] = b[submatrix_b_i * n + j];
        }

        __syncthreads();

        for (size_t k = 0; k < BLOCK_SIZE; ++k) {
            c_i_j += submatrix_a[threadIdx.y][k] * submatrix_b[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (i < n && j < n) {
        c[i * n + j] = c_i_j;
    }
}

__global__ void gpuWarpIntrinsicsMatrixMultiplicationKernel(float const *const a, float const *const b, float *const c, size_t const n) {
    size_t const start_i = blockIdx.y * BLOCK_SIZE;
    size_t const j = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    float column_c[BLOCK_SIZE] = { 0 };
    for (size_t submatrix_index = 0; submatrix_index * BLOCK_SIZE < n; ++submatrix_index) {
        for (size_t k = 0; k < BLOCK_SIZE; ++k) {
            float a_i_k = 0;
            float b_k_j = 0;

            size_t const i = start_i + threadIdx.x;
            size_t const submatrix_a_j = submatrix_index * BLOCK_SIZE + k;
            if (i < n && submatrix_a_j < n) {
                a_i_k = a[i * n + submatrix_a_j];
            }

            size_t const submatrix_b_i = submatrix_index * BLOCK_SIZE + k;
            if (submatrix_b_i < n && j < n) {
                b_k_j = b[submatrix_b_i * n + j];
            }

            for (size_t l = 0; l < BLOCK_SIZE; ++l) {
                column_c[l] += __shfl_sync(0xFFFFFFFF, a_i_k, l) * b_k_j;
            }
        }
    }

    for (size_t l = 0; l < BLOCK_SIZE; ++l) {
        size_t const i = start_i + l;
        if (i < n && j < n) {
            c[i * n + j] = column_c[l];
        }
    }
}

void gpuMultiplyMatrices(float const *const a, float const *const b, float *const c, size_t const n, bool const warp_intrinsics) {
    float *device_a;
    float *device_b;
    float *device_c;

    cudaMalloc(&device_a, n * n * sizeof(float));
    cudaMalloc(&device_b, n * n * sizeof(float));
    cudaMalloc(&device_c, n * n * sizeof(float));

    cudaMemcpy(device_a, a, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, b, n * n * sizeof(float), cudaMemcpyHostToDevice);

    auto const start = std::chrono::high_resolution_clock::now();

    if (warp_intrinsics) {
        dim3 const block_dimensions(BLOCK_SIZE, 1);
        dim3 const grid_dimensions((n + block_dimensions.x - 1) / block_dimensions.x, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
        gpuWarpIntrinsicsMatrixMultiplicationKernel<<<grid_dimensions, block_dimensions>>>(device_a, device_b, device_c, n);
    } else {
        dim3 const block_dimensions(BLOCK_SIZE, BLOCK_SIZE);
        dim3 const grid_dimensions((n + block_dimensions.x - 1) / block_dimensions.x, (n + block_dimensions.y - 1) / block_dimensions.y);
        gpuSharedMemoryMatrixMultiplicationKernel<<<grid_dimensions, block_dimensions>>>(device_a, device_b, device_c, n);
    }
    cudaDeviceSynchronize();

    auto const end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> const elapsed_seconds = end - start;

    if (warp_intrinsics) {
        std::cout << "GPU warp intrinsics elapsed time = ";
    } else {
        std::cout << "GPU shared memory elapsed time = ";
    }
    std::cout << elapsed_seconds.count() << " seconds" << std::endl;

    cudaMemcpy(c, device_c, n * n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);
}

float maximumMatrixDeviation(float const *const a, float const *const b, size_t const n) {
    float maximum_matrix_deviation = 0;
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

    float const MINIMUM_VALUE = -10;
    float const MAXIMUM_VALUE = 10;
    float const *const a = randomMatrix(MINIMUM_VALUE, MAXIMUM_VALUE, n);
    float const *const b = randomMatrix(MINIMUM_VALUE, MAXIMUM_VALUE, n);

    float *const gpu_shared_memory_c = new float[n * n];
    gpuMultiplyMatrices(a, b, gpu_shared_memory_c, n, false);

    float *const gpu_warp_intrinsics_c = new float[n * n];
    gpuMultiplyMatrices(a, b, gpu_warp_intrinsics_c, n, true);

    delete[] a;
    delete[] b;

    std::cout << "maximum GPU shared memory and warp intrinsics matrix deviation = " << maximumMatrixDeviation(gpu_shared_memory_c, gpu_warp_intrinsics_c, n) << std::endl;

    delete[] gpu_shared_memory_c;
    delete[] gpu_warp_intrinsics_c;
}
