/*
bazel run //src:reduce3

Sum of 16777216 random Numbers in reduce3
        host : 8389645.000000
        GPU  : 8389645.000000

* Use "Get nsys report" task  in tasks.json
nsys profile --force-overwrite true -o profiles/reduce3.nsys-rep bazel-bin/src/reduce3 
nsys stats --force-export true --timeunit microseconds --report nvtx_pushpop_sum profiles/reduce3.nsys-rep 

  Time (%)  Total Time (us)  Instances   Avg (us)    Med (us)    Min (us)    Max (us)   StdDev (us)     Range   
 --------  ---------------  ---------  ----------  ----------  ----------  ----------  -----------  -----------
     99.8       188043.928          1  188043.928  188043.928  188043.928  188043.928        0.000  host call  
      0.2          364.034        100       3.640       3.023       2.805      62.783        5.977  device call
*/

#include "cuda/include/thrust/device_vector.h"
#include "cuda/include/thrust/host_vector.h"
#include "include/nvToolsExt.h"
#include <iostream>
#include <random>

__global__ void reduce3(float *x, float *y, int m) {
    extern __shared__ float partial_sum[];

    int tid    = threadIdx.x;
    int gid    = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // partial sum
    partial_sum[tid] = 0.0f;
    for (int k{gid}; k < m; k += stride) {
        partial_sum[tid] += x[k];
    }
    __syncthreads();

    // same old reduction in-place within block
    if (blockDim.x >= 1024 && tid < 512)
        partial_sum[tid] += partial_sum[tid + 512];
    __syncthreads();
    if (blockDim.x >= 512 && tid < 256)
        partial_sum[tid] += partial_sum[tid + 256];
    __syncthreads();
    if (blockDim.x >= 256 && tid < 128)
        partial_sum[tid] += partial_sum[tid + 128];
    __syncthreads();
    if (blockDim.x >= 128 && tid < 64)
        partial_sum[tid] += partial_sum[tid + 64];
    __syncthreads();

    // last reduce with the warp
    if (tid < 32) {
        // volatile float *vsmem = partial_sum;
        // vsmem[tid] += vsmem[tid + 32];
        // vsmem[tid] += vsmem[tid + 16];
        // vsmem[tid] += vsmem[tid + 8];
        // vsmem[tid] += vsmem[tid + 4];
        // vsmem[tid] += vsmem[tid + 2];
        // vsmem[tid] += vsmem[tid + 1];

        // clang-format off
        partial_sum[tid]              += partial_sum[tid + 32]; __syncwarp(); // the syncwarps
		if(tid < 16) partial_sum[tid] += partial_sum[tid + 16]; __syncwarp(); // are required
		if(tid <  8) partial_sum[tid] += partial_sum[tid +  8]; __syncwarp(); // for devices of
		if(tid <  4) partial_sum[tid] += partial_sum[tid +  4]; __syncwarp(); // CC = 7 and above
		if(tid <  2) partial_sum[tid] += partial_sum[tid +  2]; __syncwarp(); // for CC < 7
		if(tid <  1) partial_sum[tid] += partial_sum[tid +  1]; __syncwarp(); // they do nothing
        // clang-format on
    }
    if (tid == 0)
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
double host_reduce3(const thrust::host_vector<T> &x) {
    RANGE("host call");
    double result = 0.0;
    for (const auto &elem : x) {
        result += elem;
    }
    return result;
}

template <typename T>
void device_reduce3(
  thrust::device_vector<T> &d_x,
  thrust::device_vector<T> &d_y,
  const int                 N,
  const int                 blocks,
  const int                 threads) {
    RANGE("device call");
    reduce3<<<blocks, threads, threads * sizeof(T)>>>(d_x.data().get(), d_y.data().get(), N);
    reduce3<<<1, blocks, blocks * sizeof(T)>>>(d_y.data().get(), d_x.data().get(), blocks);
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
    float host_sum = host_reduce3(x);

    double gpu_sum = 0.0;
    for (int rep{0}; rep < nreps; ++rep) {
        device_reduce3(dev_x, dev_y, N, blocks, threads);
        if (rep == 0)
            gpu_sum = dev_x[0];
    }
    cudaDeviceSynchronize();
    // double gpu_sum = dev_x[0]; // D2H copy (1 word)
    // clang-format off
    std::cout << "Sum of " << N << " random Numbers in reduce3" 
              << "\n\thost : " << std::fixed << host_sum 
              << "\n\tGPU  : " << std::fixed << gpu_sum 
              << "\nand " << (host_sum == gpu_sum ? "✅ Sum Match " : "❌ Not Match ") 
              << std::endl;
    // clang-format on
    return 0;
}
