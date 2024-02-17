#include "cuda/include/thrust/device_vector.h"
#include "cuda/include/thrust/host_vector.h"
#include "include/nvToolsExt.h"
#include <iostream>
#include <random>

__global__ void reduce2(float *x, float *y, int m) {
    extern __shared__ float partial_sum[];

    int id     = threadIdx.x;
    int tid    = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // partial sum
    partial_sum[id] = 0.0f;
    for (int k{tid}; k < m; k += stride) {
        partial_sum[id] += x[k];
    }
    __syncthreads();

    // block reduction
    for (int k = blockDim.x / 2; k > 0; k /= 2) {
        if (id < k)
            partial_sum[id] += partial_sum[id + k];
        __syncthreads();
    }
    if (id == 0)
        y[blockIdx.x] = partial_sum[0];
}

template <typename T>
void init_data(thrust::host_vector<T> &x) {
    std::default_random_engine        seed(12345678);
    std::uniform_real_distribution<T> dist(0.0, 1.0);
    for (auto &elem : x)
        elem = dist(seed);
}

template <typename T>
double host_reduce2(const thrust::host_vector<T> &x) {
    RANGE("host call");
    double result = 0.0;
    for (const auto &elem : x) {
        result += elem;
    }
    return result;
}

template <typename T>
void device_reduce2(
  thrust::device_vector<T> &d_x,
  thrust::device_vector<T> &d_y,
  const int                 N,
  const int                 blocks,
  const int                 threads) {
    RANGE("device call");
    reduce2<<<blocks, threads, threads * sizeof(T)>>>(d_x.data().get(), d_y.data().get(), N);
    reduce2<<<1, blocks, blocks * sizeof(T)>>>(d_y.data().get(), d_x.data().get(), blocks);
    cudaDeviceSynchronize();
}

int main(int argc, char *argv[]) {
    int N       = (argc > 1) ? atoi(argv[1]) : 1 << 24; // default 2**24
    int blocks  = (argc > 2) ? atoi(argv[2]) : 256;
    int threads = (argc > 3) ? atoi(argv[3]) : 256;
    int nreps   = (argc > 4) ? atoi(argv[4]) : 100;

    thrust::host_vector<float>   x(N);
    thrust::device_vector<float> dev_x(N);
    thrust::device_vector<float> dev_y(blocks);

    // initialise x with random numbers and copy to dx
    init_data(x);

    dev_x          = x; // H2D copy (N words)
    float host_sum = host_reduce2(x);

    double gpu_sum = 0.0;
    for (int rep{0}; rep < nreps; ++rep) {
        device_reduce2(dev_x, dev_y, N, blocks, threads);
        if (rep == 0)
            gpu_sum = dev_x[0];
    }

    // double gpu_sum = dev_x[0]; // D2H copy (1 word)
    // clang-format off
    std::cout << "Sum of " << N << " random Numbers in reduce2" 
              << "\n\thost : " << std::fixed << host_sum 
              << "\n\tGPU  : " << std::fixed << gpu_sum 
              << "\nand " << (host_sum == gpu_sum ? "✅ Sum Match " : "❌ Not Match ") 
              << std::endl;
    // clang-format on
    return 0;
}
