#include "cuda/include/thrust/device_vector.h"
#include "cuda/include/thrust/host_vector.h"
#include "include/nvToolsExt.h"
#include <iostream>
#include <random>

__global__ void reduce0(float *x, int m) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    x[tid] += x[tid + m];
}

template <typename T>
void init_data(thrust::host_vector<T> &x) {
    std::default_random_engine        seed(12345678);
    std::uniform_real_distribution<T> dist(0.0, 1.0);
    for (auto &elem : x)
        elem = dist(seed);
}

template <typename T>
double host_reduce0(const thrust::host_vector<T> &x) {
    RANGE("host call");
    double result = 0.0;
    for (const auto &elem : x) {
        result += elem;
    }
    return result;
}

template <typename T>
void device_reduce0(thrust::device_vector<T> &d_x, const int N) {
    // simple GPU reduce for N = power of 2
    RANGE("device call");
    for (int m = N; m > 0; m /= 2) {
        int threads = std::min(256, m);
        int blocks  = std::max(m / 256, 1);
        reduce0<<<blocks, threads>>>(d_x.data().get(), m);
    }
    cudaDeviceSynchronize();
}

int main(int argc, char *argv[]) {
    int N     = (argc > 1) ? atoi(argv[1]) : 1 << 24; // default 2**24
    int nreps = (argc > 2) ? atoi(argv[2]) : 100;

    thrust::host_vector<float>   x(N);
    thrust::device_vector<float> dev_x(N);

    // initialise x with random numbers and copy to dx
    init_data(x);

    dev_x          = x; // H2D copy (N words)
    float host_sum = host_reduce0(x);

    double gpu_sum = 0.0;
    for (int i{0}; i < nreps; ++i) {
        device_reduce0(dev_x, N);
        if (i == 0)
            gpu_sum = dev_x[0];
    }

    // clang-format off
    std::cout << "Sum of " << N << " random Numbers " 
              << "\n\thost : " << host_sum 
              << "\n\tGPU  : " << gpu_sum 
              << std::endl;
    // clang-format on
    return 0;
}
