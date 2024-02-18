/*
bazel run //src:reduce6

Sum of 16777216 random Numbers in reduce6
        host : 8389645.000000
        GPU  : 8389645.000000

* Use "Get nsys report" task  in tasks.json
nsys profile --force-overwrite true -o profiles/reduce6.nsys-rep bazel-bin/src/reduce6
nsys stats --force-export true --timeunit microseconds --report nvtx_pushpop_sum
profiles/reduce6.nsys-rep


Time (%)  Total Time (us)  Instances   Avg (us)    Med (us)    Min (us)    Max (us)   StdDev (us)     Range
--------  ---------------  ---------  ----------  ----------  ----------  ----------  -----------  -----------
    99.8       183548.942          1  183548.942  183548.942  183548.942  183548.942        0.000  host call
     0.2          349.336        100       3.493       3.008       2.818      46.269        4.334  device call
*/

#include "cuda/include/thrust/device_vector.h"
#include "cuda/include/thrust/host_vector.h"
#include "cuda/include/cooperative_groups.h"
#include "cuda/include/cooperative_groups/reduce.h"
#include "include/nvToolsExt.h"
#include <iostream>
#include <random>

namespace cg = cooperative_groups;

__global__ void reduce6(float *x, float *y, int m) {

    auto grid  = cg::this_grid();
    auto block = cg::this_thread_block();
    auto warp  = cg::tiled_partition<32>(block);

    float partial_sum = 0.0f;
    for (size_t grid_tid{grid.thread_rank()}; grid_tid < m; grid_tid += grid.size()) {
        partial_sum += x[grid_tid];
    }
    warp.sync();

    //* Using the warp level functions
    partial_sum = cg::reduce(warp, partial_sum, cg::plus<float>());

    if (warp.thread_rank() == 0) {
        atomicAdd(&y[block.group_index().x], partial_sum);
    }
}

template <typename T>
void init_data(thrust::host_vector<T> &x) {
    std::default_random_engine        seed(12345678);
    std::uniform_real_distribution<T> dist(0.0, 1.0);
    for (auto &elem : x)
        elem = dist(seed);
}

template <typename T>
double host_reduce6(const thrust::host_vector<T> &x) {
    RANGE("host call");
    double result = 0.0;
    for (const auto &elem : x) {
        result += elem;
    }
    return result;
}

template <typename T>
void device_reduce6(
  thrust::device_vector<T> &d_x, thrust::device_vector<T> &d_y, const int N, const int blocks, const int threads) {
    RANGE("device call");
    reduce6<<<blocks, threads, threads * sizeof(T)>>>(d_x.data().get(), d_y.data().get(), N);
    reduce6<<<1, blocks, blocks * sizeof(T)>>>(d_y.data().get(), d_x.data().get(), blocks);
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
    float host_sum = host_reduce6(x);

    double gpu_sum = 0.0;
    for (int rep{0}; rep < nreps; ++rep) {
        device_reduce6(dev_x, dev_y, N, blocks, threads);
        if (rep == 0)
            gpu_sum = dev_x[0];
    }
    cudaDeviceSynchronize();

    // ! NOTE : Due to round off we off by 1
    // clang-format off
    std::cout << "Sum of " << N << " random Numbers in reduce6" 
              << "\n\thost : " << std::fixed << host_sum 
              << "\n\tGPU  : " << std::fixed << gpu_sum 
              << "\nand " << (host_sum == gpu_sum   ? "✅ Sum Match " : "❌ Not Match ") 
              << std::endl;
    // clang-format on
    return 0;
}
