/*
bazel run //src:reduce4

Sum of 16777216 random Numbers in reduce4
        host : 8389645.000000
        GPU  : 8389645.000000

* Use "Get nsys report" task  in tasks.json
nsys profile --force-overwrite true -o profiles/reduce4.nsys-rep bazel-bin/src/reduce4 
nsys stats --force-export true --timeunit microseconds --report nvtx_pushpop_sum profiles/reduce4.nsys-rep 

 Time (%)  Total Time (us)  Instances   Avg (us)    Med (us)    Min (us)    Max (us)   StdDev (us)     Range   
 --------  ---------------  ---------  ----------  ----------  ----------  ----------  -----------  -----------
     99.8       186528.787          1  186528.787  186528.787  186528.787  186528.787        0.000  host call  
      0.2          390.003        100       3.900       3.030       2.835      85.249        8.225  device call
*/

#include "cuda/include/thrust/device_vector.h"
#include "cuda/include/thrust/host_vector.h"
#include "cuda/include/cooperative_groups.h"
#include "include/nvToolsExt.h"
#include <iostream>
#include <random>

namespace cg = cooperative_groups;

__global__ void reduce4(float *x, float *y, int m) {
    extern __shared__ float partial_sum[];

    auto grid  = cg::this_grid();
    auto block = cg::this_thread_block();
    auto warp  = cg::tiled_partition<32>(block);

    int blk_tid = block.thread_rank();

    // partial sum
    partial_sum[blk_tid] = 0.0f;
    for (int grid_tid = grid.thread_rank(); grid_tid < m; grid_tid += grid.size()) {
        partial_sum[blk_tid] += x[grid_tid];
    }
    block.sync();

    // same old reduction in-place within block
    if (block.dim_threads().x >= 1024 && blk_tid < 512)
        partial_sum[blk_tid] += partial_sum[blk_tid + 512];
    block.sync();
    if (block.dim_threads().x >= 512 && blk_tid < 256)
        partial_sum[blk_tid] += partial_sum[blk_tid + 256];
    block.sync();
    if (block.dim_threads().x >= 256 && blk_tid < 128)
        partial_sum[blk_tid] += partial_sum[blk_tid + 128];
    block.sync();
    if (block.dim_threads().x >= 128 && blk_tid < 64)
        partial_sum[blk_tid] += partial_sum[blk_tid + 64];
    block.sync();

    // last reduce with the warp
    if (warp.meta_group_rank() == 0) {

        partial_sum[blk_tid] += partial_sum[blk_tid + 32];
        warp.sync();
        partial_sum[blk_tid] += warp.shfl_down(partial_sum[blk_tid], 16);
        partial_sum[blk_tid] += warp.shfl_down(partial_sum[blk_tid], 8);
        partial_sum[blk_tid] += warp.shfl_down(partial_sum[blk_tid], 4);
        partial_sum[blk_tid] += warp.shfl_down(partial_sum[blk_tid], 2);
        partial_sum[blk_tid] += warp.shfl_down(partial_sum[blk_tid], 1);
    }
    if (blk_tid == 0)
        y[grid.block_index().x] = partial_sum[0];
}

template <typename T>
void init_data(thrust::host_vector<T> &x) {
    std::default_random_engine        seed(12345678);
    std::uniform_real_distribution<T> dist(0.0, 1.0);
    for (auto &elem : x)
        elem = dist(seed);
}

template <typename T>
double host_reduce4(const thrust::host_vector<T> &x) {
    RANGE("host call");
    double result = 0.0;
    for (const auto &elem : x) {
        result += elem;
    }
    return result;
}

template <typename T>
void device_reduce4(
  thrust::device_vector<T> &d_x,
  thrust::device_vector<T> &d_y,
  const int                 N,
  const int                 blocks,
  const int                 threads) {
    RANGE("device call");
    reduce4<<<blocks, threads, threads * sizeof(T)>>>(d_x.data().get(), d_y.data().get(), N);
    reduce4<<<1, blocks, blocks * sizeof(T)>>>(d_y.data().get(), d_x.data().get(), blocks);
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
    float host_sum = host_reduce4(x);

    double gpu_sum = 0.0;
    for (int rep{0}; rep < nreps; ++rep) {
        device_reduce4(dev_x, dev_y, N, blocks, threads);
        if (rep == 0)
            gpu_sum = dev_x[0];
    }
    cudaDeviceSynchronize();

    // double gpu_sum = dev_x[0]; // D2H copy (1 word)
    // clang-format off
    std::cout << "Sum of " << N << " random Numbers in reduce4" 
              << "\n\thost : " << std::fixed << host_sum 
              << "\n\tGPU  : " << std::fixed << gpu_sum 
              << "\nand " << (host_sum == gpu_sum ? "✅ Sum Match " : "❌ Not Match ") 
              << std::endl;
    // clang-format on
    return 0;
}
