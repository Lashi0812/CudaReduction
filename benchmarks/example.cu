#include <nvbench/nvbench.cuh>
#include <random>

#include <cuda/include/thrust/device_vector.h>
#include <cuda/include/thrust/host_vector.h>
#include <cuda/include/cooperative_groups.h>
#include <cuda/include/cooperative_groups/reduce.h>
#include <string>
#include "include/nvToolsExt.h"

namespace cg = cooperative_groups;

__global__ void reduce7(float const *__restrict__ x, float *__restrict__ y, int m) {

    auto grid  = cg::this_grid();
    auto block = cg::this_thread_block();
    auto warp  = cg::tiled_partition<32>(block);

    // * vector loading
    float4 partial_sum4 = {0.0f, 0.0f, 0.0f, 0.0f};
    for (size_t grid_tid{grid.thread_rank()}; grid_tid < m / 4; grid_tid += grid.size()) {
        float4 temp4 = reinterpret_cast<const float4 *>(x)[grid_tid];
        partial_sum4.x += temp4.x;
        partial_sum4.y += temp4.y;
        partial_sum4.z += temp4.z;
        partial_sum4.w += temp4.w;
    }
    float partial_sum = partial_sum4.x + partial_sum4.y + partial_sum4.z + partial_sum4.w;
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

void reduce7_benchmark(nvbench::state &state) {
    auto      N       = state.get_int64("input_size");
    const int blocks  = 256;
    const int threads = 256;

    thrust::host_vector<float>   x(N);
    thrust::device_vector<float> dev_x(N);
    thrust::device_vector<float> dev_y(blocks);

    // state.collect_dram_throughput();
    init_data(x);
    dev_x = x;
    // state.collect_l1_hit_rates();
    // state.collect_l2_hit_rates();
    // state.collect_loads_efficiency();
    // state.collect_stores_efficiency();

    state.exec([&N, &blocks, &threads, &dev_x, &dev_y](nvbench::launch &launch) {
        RANGE(("kernel call" + std::to_string(N)).c_str());
        reduce7<<<blocks, threads, threads * sizeof(float), launch.get_stream()>>>(
          dev_x.data().get(), dev_y.data().get(), N);
        reduce7<<<1, blocks, blocks * sizeof(float), launch.get_stream()>>>(
          dev_y.data().get(), dev_x.data().get(), blocks);
    });
}
NVBENCH_BENCH(reduce7_benchmark)
  .add_int64_power_of_two_axis("input_size", nvbench::range(14, 24))
  .set_timeout(1); // Limit to one second per measurement.bazel-bin