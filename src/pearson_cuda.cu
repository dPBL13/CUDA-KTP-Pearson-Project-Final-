#include "pearson_cuda.h"

#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstddef>
#include <algorithm>

namespace pearson {

namespace {


constexpr int kThreadsPerBlock = 256;


constexpr int kMaxBlocks = 2048;


#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                          \
            throw CudaRuntimeError(                                          \
                std::string("CUDA error di ") + __FILE__ + ":" +             \
                std::to_string(__LINE__) + " -- " +                          \
                cudaGetErrorString(err__));                                  \
        }                                                                    \
    } while (0)


__global__ void pearson_partial_sums_kernel(const float* __restrict__ x,
                                             const float* __restrict__ y,
                                             std::size_t n,
                                             double* __restrict__ partial_sum_x,
                                             double* __restrict__ partial_sum_y,
                                             double* __restrict__ partial_sum_x2,
                                             double* __restrict__ partial_sum_y2,
                                             double* __restrict__ partial_sum_xy) {


    extern __shared__ double shared_mem[];
    double* s_sum_x  = shared_mem;
    double* s_sum_y  = s_sum_x  + blockDim.x;
    double* s_sum_x2 = s_sum_y  + blockDim.x;
    double* s_sum_y2 = s_sum_x2 + blockDim.x;
    double* s_sum_xy = s_sum_y2 + blockDim.x;

    const int tid = threadIdx.x;
    const std::size_t global_start = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;


    double local_sum_x = 0.0;
    double local_sum_y = 0.0;
    double local_sum_x2 = 0.0;
    double local_sum_y2 = 0.0;
    double local_sum_xy = 0.0;

    for (std::size_t i = global_start; i < n; i += stride) {
        const double xi = static_cast<double>(x[i]);
        const double yi = static_cast<double>(y[i]);
        local_sum_x += xi;
        local_sum_y += yi;
        local_sum_x2 += xi * xi;
        local_sum_y2 += yi * yi;
        local_sum_xy += xi * yi;
    }


    s_sum_x[tid] = local_sum_x;
    s_sum_y[tid] = local_sum_y;
    s_sum_x2[tid] = local_sum_x2;
    s_sum_y2[tid] = local_sum_y2;
    s_sum_xy[tid] = local_sum_xy;


    __syncthreads();


    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s_sum_x[tid] += s_sum_x[tid + offset];
            s_sum_y[tid] += s_sum_y[tid + offset];
            s_sum_x2[tid] += s_sum_x2[tid + offset];
            s_sum_y2[tid] += s_sum_y2[tid + offset];
            s_sum_xy[tid] += s_sum_xy[tid + offset];
        }


        __syncthreads();
    }


    if (tid == 0) {
        partial_sum_x[blockIdx.x] = s_sum_x[0];
        partial_sum_y[blockIdx.x] = s_sum_y[0];
        partial_sum_x2[blockIdx.x] = s_sum_x2[0];
        partial_sum_y2[blockIdx.x] = s_sum_y2[0];
        partial_sum_xy[blockIdx.x] = s_sum_xy[0];
    }
}

}

bool is_cuda_device_available() {
    int device_count = 0;
    const cudaError_t err = cudaGetDeviceCount(&device_count);


    if (err != cudaSuccess) {
        return false;
    }
    return device_count > 0;
}

PearsonResult compute_pearson_cuda(const std::vector<float>& x,
                                    const std::vector<float>& y,
                                    CudaTimingBreakdown& out_timing) {
    if (x.size() != y.size()) {
        throw std::invalid_argument(
            "Ukuran vector X (" + std::to_string(x.size()) +
            ") dan Y (" + std::to_string(y.size()) + ") tidak sama.");
    }

    if (!is_cuda_device_available()) {
        throw CudaRuntimeError(
            "Tidak ada CUDA device yang tersedia pada sistem ini. "
            "Pastikan driver NVIDIA terpasang dan GPU terdeteksi "
            "(cek dengan 'nvidia-smi').");
    }

    const std::size_t n = x.size();


    const int blocks_needed =
        static_cast<int>((n + kThreadsPerBlock - 1) / kThreadsPerBlock);
    const int num_blocks = std::min(blocks_needed, kMaxBlocks);

    const int grid_size = num_blocks > 0 ? num_blocks : 1;


    float* d_x = nullptr;
    float* d_y = nullptr;
    double* d_partial_sum_x = nullptr;
    double* d_partial_sum_y = nullptr;
    double* d_partial_sum_x2 = nullptr;
    double* d_partial_sum_y2 = nullptr;
    double* d_partial_sum_xy = nullptr;


    auto free_all_device_memory = [&]() {
        if (d_x) { cudaFree(d_x); d_x = nullptr; }
        if (d_y) { cudaFree(d_y); d_y = nullptr; }
        if (d_partial_sum_x) { cudaFree(d_partial_sum_x); d_partial_sum_x = nullptr; }
        if (d_partial_sum_y) { cudaFree(d_partial_sum_y); d_partial_sum_y = nullptr; }
        if (d_partial_sum_x2) { cudaFree(d_partial_sum_x2); d_partial_sum_x2 = nullptr; }
        if (d_partial_sum_y2) { cudaFree(d_partial_sum_y2); d_partial_sum_y2 = nullptr; }
        if (d_partial_sum_xy) { cudaFree(d_partial_sum_xy); d_partial_sum_xy = nullptr; }
    };


    try {
        CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_y, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_partial_sum_x, grid_size * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_partial_sum_y, grid_size * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_partial_sum_x2, grid_size * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_partial_sum_y2, grid_size * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_partial_sum_xy, grid_size * sizeof(double)));


        cudaEvent_t ev_start_h2d, ev_end_h2d;
        cudaEvent_t ev_start_kernel, ev_end_kernel;
        cudaEvent_t ev_start_d2h, ev_end_d2h;
        CUDA_CHECK(cudaEventCreate(&ev_start_h2d));
        CUDA_CHECK(cudaEventCreate(&ev_end_h2d));
        CUDA_CHECK(cudaEventCreate(&ev_start_kernel));
        CUDA_CHECK(cudaEventCreate(&ev_end_kernel));
        CUDA_CHECK(cudaEventCreate(&ev_start_d2h));
        CUDA_CHECK(cudaEventCreate(&ev_end_d2h));


        CUDA_CHECK(cudaEventRecord(ev_start_h2d));
        CUDA_CHECK(cudaMemcpy(d_x, x.data(), n * sizeof(float),
                               cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_y, y.data(), n * sizeof(float),
                               cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(ev_end_h2d));


        const size_t shared_mem_bytes =
            5 * static_cast<size_t>(kThreadsPerBlock) * sizeof(double);

        CUDA_CHECK(cudaEventRecord(ev_start_kernel));
        pearson_partial_sums_kernel<<<grid_size, kThreadsPerBlock,
                                       shared_mem_bytes>>>(
            d_x, d_y, n,
            d_partial_sum_x, d_partial_sum_y,
            d_partial_sum_x2, d_partial_sum_y2, d_partial_sum_xy);


        CUDA_CHECK(cudaGetLastError());


        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(ev_end_kernel));


        std::vector<double> h_partial_sum_x(grid_size);
        std::vector<double> h_partial_sum_y(grid_size);
        std::vector<double> h_partial_sum_x2(grid_size);
        std::vector<double> h_partial_sum_y2(grid_size);
        std::vector<double> h_partial_sum_xy(grid_size);

        CUDA_CHECK(cudaEventRecord(ev_start_d2h));
        CUDA_CHECK(cudaMemcpy(h_partial_sum_x.data(), d_partial_sum_x,
                               grid_size * sizeof(double),
                               cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_partial_sum_y.data(), d_partial_sum_y,
                               grid_size * sizeof(double),
                               cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_partial_sum_x2.data(), d_partial_sum_x2,
                               grid_size * sizeof(double),
                               cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_partial_sum_y2.data(), d_partial_sum_y2,
                               grid_size * sizeof(double),
                               cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_partial_sum_xy.data(), d_partial_sum_xy,
                               grid_size * sizeof(double),
                               cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(ev_end_d2h));
        CUDA_CHECK(cudaEventSynchronize(ev_end_d2h));


        float ms_h2d = 0.0f, ms_kernel = 0.0f, ms_d2h = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms_h2d, ev_start_h2d, ev_end_h2d));
        CUDA_CHECK(cudaEventElapsedTime(&ms_kernel, ev_start_kernel, ev_end_kernel));
        CUDA_CHECK(cudaEventElapsedTime(&ms_d2h, ev_start_d2h, ev_end_d2h));

        out_timing.host_to_device_ms = static_cast<double>(ms_h2d);
        out_timing.kernel_ms = static_cast<double>(ms_kernel);
        out_timing.device_to_host_ms = static_cast<double>(ms_d2h);


        const auto t_reduce_start = std::chrono::high_resolution_clock::now();
        PearsonAccumulators acc;
        acc.n = static_cast<double>(n);
        for (int b = 0; b < grid_size; ++b) {
            acc.sum_x += h_partial_sum_x[b];
            acc.sum_y += h_partial_sum_y[b];
            acc.sum_x2 += h_partial_sum_x2[b];
            acc.sum_y2 += h_partial_sum_y2[b];
            acc.sum_xy += h_partial_sum_xy[b];
        }
        const auto t_reduce_end = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::milli> reduce_elapsed =
            t_reduce_end - t_reduce_start;
        out_timing.final_reduction_cpu_ms = reduce_elapsed.count();


        CUDA_CHECK(cudaEventDestroy(ev_start_h2d));
        CUDA_CHECK(cudaEventDestroy(ev_end_h2d));
        CUDA_CHECK(cudaEventDestroy(ev_start_kernel));
        CUDA_CHECK(cudaEventDestroy(ev_end_kernel));
        CUDA_CHECK(cudaEventDestroy(ev_start_d2h));
        CUDA_CHECK(cudaEventDestroy(ev_end_d2h));


        free_all_device_memory();

        return finalize_pearson(acc);

    } catch (...) {


        free_all_device_memory();
        throw;
    }
}

}
