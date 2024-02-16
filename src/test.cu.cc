#include "cuda/include/thrust/device_vector.h"
#include "cuda/include/thrust/host_vector.h"
#include "cuda/include/nvtx3/nvToolsExt.h"
#include <random>


__global__ void reduce0(float *x, int m) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    x[tid] += x[tid + m];
}

int main(int argc, char *argv[]) {
    int N = (argc > 1) ? atoi(argv[1]) : 1 << 24; // default 2**24

    thrust::host_vector<float>   x(N);
    thrust::device_vector<float> dev_x(N);

    // initialise x with random numbers and copy to dx
    std::default_random_engine            seed(12345678);
    std::uniform_real_distribution<float> dist(0.0, 1.0);
    nvtxRangePushA("host_cal");
    for (int k = 0; k < N; k++)
        x[k] = dist(seed);
    nvtxRangePop();

    dev_x           = x; // H2D copy (N words)
    double host_sum = 0.0;
    for (int k = 0; k < N; k++)
        host_sum += x[k]; // host reduce!

    // simple GPU reduce for N = power of 2
    nvtxRangePushA("cuda_kernel");
    for (int m = N / 2; m > 0; m /= 2) {
        int threads = std::min(256, m);
        int blocks  = std::max(m / 256, 1);
        reduce0<<<blocks, threads>>>(dev_x.data().get(), m);
    }
    cudaDeviceSynchronize();
    nvtxRangePop();

    double gpu_sum = dev_x[0]; // D2H copy (1 word)
    printf("sum of %d random numbers: \n\thost %.3f, \n\tGPU  %.3f\n", N, host_sum, gpu_sum);
    return 0;
}
