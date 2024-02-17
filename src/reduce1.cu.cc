#include "cuda/include/thrust/device_vector.h"
#include "cuda/include/thrust/host_vector.h"
#include "include/nvToolsExt.h"
#include <iostream>
#include <random>

__global__ void reduce1(float *x, int m) {
    int   tid         = blockDim.x * blockIdx.x + threadIdx.x;
    int   stride      = gridDim.x * blockDim.x;
    float partial_sum = 0.0f;
    for (int k{tid}; k < m; k += stride) {
        partial_sum += x[k];
    }
    x[tid] = partial_sum;
}

template <typename T>
void init_data(thrust::host_vector<T> &x) {
    std::default_random_engine        seed(12345678);
    std::uniform_real_distribution<T> dist(0.0, 1.0);
    for (auto &elem : x)
        elem = dist(seed);
}

template <typename T>
double host_reduce1(const thrust::host_vector<T> &x) {
    RANGE("host call");
    double result = 0.0;
    for (const auto &elem : x) {
        result += elem;
    }
    return result;
}

template <typename T>
void device_reduce1(
  thrust::device_vector<T> &d_x, const int N, const int blocks, const int threads) {
    RANGE("device call");
    reduce1<<<blocks, threads>>>(d_x.data().get(), N);
    reduce1<<<1, threads>>>(d_x.data().get(), blocks * threads);
    reduce1<<<1, 1>>>(d_x.data().get(), threads);
    cudaDeviceSynchronize();
}

int main(int argc, char *argv[]) {
    int N       = (argc > 1) ? atoi(argv[1]) : 1 << 24; // default 2**24
    int blocks  = (argc > 2) ? atoi(argv[2]) : 288;
    int threads = (argc > 3) ? atoi(argv[3]) : 256;
    int nreps = (argc > 4) ? atoi(argv[4]) : 100;

    thrust::host_vector<float>   x(N);
    thrust::device_vector<float> dev_x(N);

    // initialise x with random numbers and copy to dx
    init_data(x);

    dev_x          = x; // H2D copy (N words)
    float host_sum = host_reduce1(x);

    double gpu_sum = 0.0;
    for (int i{0}; i < nreps; ++i) {
        device_reduce1(dev_x, N, blocks, threads);
        if (i == 0)
            gpu_sum = dev_x[0];
    }

    // double gpu_sum = dev_x[0]; // D2H copy (1 word)
    // clang-format off
    std::cout << "Sum of " << N << " random Numbers in reduce1" 
              << "\n\thost : " << std::fixed << host_sum 
              << "\n\tGPU  : " << std::fixed << gpu_sum 
              << "\nand " << (host_sum == gpu_sum ? "✅ Sum Match " : "❌ Not Match ") 
              << std::endl;
    // clang-format on
    return 0;
}
