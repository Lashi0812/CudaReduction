/*
bazel run //src:reduce7

Sum of 16777216 random Numbers in reduce7
        host : 8389645.000000
        GPU  : 8389645.000000

* Use "Get nsys report" task  in tasks.json
nsys profile --force-overwrite true -o profiles/reduce7.nsys-rep bazel-bin/src/reduce7
nsys stats --force-export true --timeunit microseconds --report nvtx_pushpop_sum
profiles/reduce7.nsys-rep


Time (%)  Total Time (us)  Instances   Avg (us)    Med (us)    Min (us)    Max (us)   StdDev (us)     Range
 --------  ---------------  ---------  ----------  ----------  ----------  ----------  -----------  -----------
     99.8       190369.036          1  190369.036  190369.036  190369.036  190369.036        0.000  host call
      0.2          383.991        100       3.840       2.979       2.790      78.158        7.536  device call
*/

#define RUNTIME_API_CALL(apiFunctionCall)                           \
    do {                                                            \
        cudaError_t _status = apiFunctionCall;                      \
        if (_status != cudaSuccess) {                               \
            fprintf(                                                \
              stderr,                                               \
              "%s:%d: Error: Function %s failed with error: %s.\n", \
              __FILE__,                                             \
              __LINE__,                                             \
              #apiFunctionCall,                                     \
              cudaGetErrorString(_status));                         \
                                                                    \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

#include "cuda/include/thrust/device_vector.h"
#include "cuda/include/thrust/host_vector.h"
#include <cuda/include/cooperative_groups.h>
#include <cuda/include/cub/warp/warp_load.cuh>
#include <cuda/atomic>
#include "include/nvToolsExt.h"
#include <iostream>
#include <random>

namespace cg = cooperative_groups;

__global__ void reduce8(float const *__restrict__ x, float *__restrict__ y, int m) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    auto warp  = cg::tiled_partition<32>(block);

    using WarpReduce = cub::WarpReduce<float, 32>;
    __shared__ typename WarpReduce::TempStorage temp_storage;

    int aggregate = WarpReduce(temp_storage).Sum(grid.thread_rank());
    if (warp.thread_rank() == 0) {
        cuda::atomic_ref<float, cuda::thread_scope_device> atomic_result(*y);
        atomic_result.fetch_add(aggregate, cuda::memory_order_relaxed);
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
double host_reduce7(const thrust::host_vector<T> &x) {
    RANGE("host call");
    double result = 0.0;
    for (const auto &elem : x) {
        result += elem;
    }
    return result;
}

template <typename T>
void device_reduce7(
  thrust::device_vector<T> &d_x, thrust::device_vector<T> &d_y, const int N, const int blocks, const int threads) {
    RANGE("device call");
    reduce8<<<3, 128>>>(d_x.data().get(), d_y.data().get(), N);
    RUNTIME_API_CALL(cudaGetLastError());
    // reduce7<<<1, blocks>>>(d_y.data().get(), d_x.data().get(), blocks);
    // RUNTIME_API_CALL(cudaGetLastError());
}

int main(int argc, char *argv[]) {
    int N       = (argc > 1) ? 1 << atoi(argv[1]) : 1 << 24; // default 2**24
    int blocks  = (argc > 2) ? 1 << atoi(argv[2]) : 1 << 8;
    int threads = (argc > 3) ? 1 << atoi(argv[3]) : 1 << 8;
    int nreps   = (argc > 4) ? atoi(argv[4]) : 100;

    thrust::host_vector<float>   host_x(128);
    thrust::host_vector<float>   host_y(1);
    thrust::device_vector<float> dev_x(128);
    thrust::device_vector<float> dev_y(1);

    // initialise x with random numbers and copy to dx
    init_data(host_x);

    dev_x = host_x; // H2D copy (N words)
    // float host_sum = host_reduce7(x);

    double gpu_sum = 0.0;
    for (int rep{0}; rep < nreps; ++rep) {
        device_reduce7(dev_x, dev_y, N, blocks, threads);
        if (rep == 0)
            // gpu_sum = dev_x[0];
            host_y = dev_y;
    }
    RUNTIME_API_CALL(cudaDeviceSynchronize());

    for (auto &x_value : host_y) {
        std::cout << x_value << std::endl;
    }

    // ! NOTE : Due to round off we off by 1
    // clang-format off
    // std::cout << "Sum of " << N << " random Numbers in reduce7" 
    //           << "\n\thost : " << std::fixed << host_sum 
    //           << "\n\tGPU  : " << std::fixed << gpu_sum 
    //           << "\nand " << (host_sum == gpu_sum   ? "✅ Sum Match " : "❌ Not Match ") 
    //           << std::endl;
    // clang-format on
    return 0;
}
