#include "common.cuh"

#include <cuda_fp16.h>
#include <mma.h>

#include <iomanip>
#include <limits>

// G6：最小 WMMA GEMM 学习实现。
//
// 数据布局：
//   A[M,K]：row-major FP16
//   B[K,N]：col-major FP16
//   C[M,N]：row-major FP32
//
// 当前版本只支持 M/N/K 都是 16 的整数倍。一个 CUDA block 只有一个
// warp（32 threads），一个 warp 负责一个 16 x 16 的 C output tile。
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
constexpr int WARP_SIZE = 32;

// G7 staged WMMA 配置：8 个 warp 组成一个 CTA，计算 32 x 64 的 C tile。
constexpr int STAGED_BLOCK_M = 32;
constexpr int STAGED_BLOCK_N = 64;
constexpr int STAGED_BLOCK_K = 16;
constexpr int STAGED_WARPS_M = 2;
constexpr int STAGED_WARPS_N = 4;
constexpr int STAGED_WARPS = STAGED_WARPS_M * STAGED_WARPS_N;
constexpr int STAGED_THREADS = STAGED_WARPS * WARP_SIZE;

// CUDA Core 对照：一个线程计算一个 FP32 输出元素。
// A 为 row-major，B 为 col-major，与 WMMA kernel 使用相同输入。
__global__ void gemm_fp16_scalar_kernel(
    const half* a,
    const half* b,
    float* c,
    int m,
    int n,
    int k
) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= m || col >= n) return;

    float sum = 0.0f;
    for (int kk = 0; kk < k; ++kk) {
        const float av = __half2float(a[row * k + kk]);
        const float bv = __half2float(b[col * k + kk]);
        sum = fmaf(av, bv, sum);
    }
    c[row * n + col] = sum;
}

// Tensor Core / WMMA 版本。
__global__ void gemm_wmma_kernel(
    const half* a,
    const half* b,
    float* c,
    int m,
    int n,
    int k
) {
    // blockIdx 直接选择一个 16 x 16 的 C output tile。
    const int tile_row = blockIdx.y * WMMA_M;
    const int tile_col = blockIdx.x * WMMA_N;

    // 这些边界在当前 host 侧约束下不会触发。保留判断是为了明确接口边界；
    // 条件对整个 warp 一致，因此不会造成 WMMA warp divergence。
    if (tile_row >= m || tile_col >= n) return;

    using namespace nvcuda;

    // A/B 使用 FP16 fragment；accumulator 使用 FP32 fragment。
    wmma::fragment<wmma::matrix_a,
                   WMMA_M, WMMA_N, WMMA_K,
                   half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b,
                   WMMA_M, WMMA_N, WMMA_K,
                   half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator,
                   WMMA_M, WMMA_N, WMMA_K,
                   float> acc_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    // 沿完整 K 维每次处理 16 个元素。每一轮更新同一个 FP32
    // accumulator fragment。
    for (int k0 = 0; k0 < k; k0 += WMMA_K) {
        const half* a_tile = a + tile_row * k + k0;

        // B 是 col-major：B[k0, tile_col] 的一维地址为
        // tile_col * K + k0，leading dimension 为 K。
        const half* b_tile = b + tile_col * k + k0;

        // 下面三个调用都必须由 warp 的 32 个 lane 一致执行。
        wmma::load_matrix_sync(a_frag, a_tile, k);
        wmma::load_matrix_sync(b_frag, b_tile, k);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    float* c_tile = c + tile_row * n + tile_col;
    wmma::store_matrix_sync(c_tile, acc_frag, n, wmma::mem_row_major);
}

// G7：Shared-memory staged WMMA。
//
// 一个 CTA：
//   - 256 threads = 8 warps
//   - C tile: 32 x 64
//   - A K-tile: 32 x 16，row-major shared layout
//   - B K-tile: 16 x 64，按列存成 Bs[64][16]
//
// 每个 warp 计算一个 16 x 16 C tile。A tile 在 N 方向被 4 个 warp
// 复用，B tile 在 M 方向被 2 个 warp 复用。
__global__ void gemm_wmma_staged_kernel(
    const half* a,
    const half* b,
    float* c,
    int m,
    int n,
    int k
) {
    using namespace nvcuda;

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int warp_m = warp_id / STAGED_WARPS_N;  // 0～1
    const int warp_n = warp_id % STAGED_WARPS_N;  // 0～3

    const int block_row = blockIdx.y * STAGED_BLOCK_M;
    const int block_col = blockIdx.x * STAGED_BLOCK_N;
    const int warp_row = block_row + warp_m * WMMA_M;
    const int warp_col = block_col + warp_n * WMMA_N;

    // 32-byte alignment 满足当前 WMMA fragment load 的对齐要求。
    __shared__ __align__(32) half As[STAGED_BLOCK_M][STAGED_BLOCK_K];

    // B 的逻辑 shape 是 [K=16][N=64]，但为了 col-major WMMA load，
    // shared memory 中按 [column][k] 保存。
    __shared__ __align__(32) half Bs[STAGED_BLOCK_N][STAGED_BLOCK_K];

    wmma::fragment<wmma::matrix_a,
                   WMMA_M, WMMA_N, WMMA_K,
                   half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b,
                   WMMA_M, WMMA_N, WMMA_K,
                   half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator,
                   WMMA_M, WMMA_N, WMMA_K,
                   float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

    for (int k0 = 0; k0 < k; k0 += STAGED_BLOCK_K) {
        // A shared tile 有 32 x 16 = 512 个 half。256 个线程各加载 2 个。
        for (int index = tid;
             index < STAGED_BLOCK_M * STAGED_BLOCK_K;
             index += STAGED_THREADS) {
            const int local_row = index / STAGED_BLOCK_K;
            const int local_k = index % STAGED_BLOCK_K;
            const int global_row = block_row + local_row;
            const int global_k = k0 + local_k;
            As[local_row][local_k] =
                global_row < m && global_k < k
                    ? a[global_row * k + global_k]
                    : __float2half(0.0f);
        }

        // B shared tile 有 64 x 16 = 1024 个 half。256 个线程各加载 4 个。
        for (int index = tid;
             index < STAGED_BLOCK_N * STAGED_BLOCK_K;
             index += STAGED_THREADS) {
            const int local_col = index / STAGED_BLOCK_K;
            const int local_k = index % STAGED_BLOCK_K;
            const int global_col = block_col + local_col;
            const int global_k = k0 + local_k;
            Bs[local_col][local_k] =
                global_col < n && global_k < k
                    ? b[global_col * k + global_k]
                    : __float2half(0.0f);
        }

        // 整个 CTA 的 A/B tile 必须完成写入，各 warp 才能加载 fragment。
        __syncthreads();

        const half* a_tile = &As[warp_m * WMMA_M][0];
        const half* b_tile = &Bs[warp_n * WMMA_N][0];

        // As 的 row-major leading dimension 是 16；Bs 按 B 列组织，
        // 对 col-major fragment 来说 leading dimension 同样是 16。
        wmma::load_matrix_sync(a_frag, a_tile, STAGED_BLOCK_K);
        wmma::load_matrix_sync(b_frag, b_tile, STAGED_BLOCK_K);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

        // 所有 warp 使用完 shared tile 后，下一轮 k0 才能覆盖它。
        __syncthreads();
    }

    // 当前 host shape 保证 CTA tile 完整落在矩阵内。保留一致条件，确保
    // 整个 warp 共同进入或跳过 store。
    if (warp_row < m && warp_col < n) {
        float* c_tile = c + warp_row * n + warp_col;
        wmma::store_matrix_sync(
            c_tile, acc_frag, n, wmma::mem_row_major
        );
    }
}

struct ErrorStats {
    double max_absolute = 0.0;
    double mean_absolute = 0.0;
    double rmse = 0.0;
    size_t non_finite = 0;
    bool correct = true;
};

ErrorStats compare_result(
    const std::vector<float>& result,
    const std::vector<float>& reference,
    float atol,
    float rtol
) {
    ErrorStats stats;
    double sum_absolute = 0.0;
    double sum_squared = 0.0;

    for (size_t i = 0; i < result.size(); ++i) {
        if (!std::isfinite(result[i])) {
            ++stats.non_finite;
            stats.correct = false;
            continue;
        }
        const double error = std::abs(
            static_cast<double>(result[i]) - reference[i]
        );
        stats.max_absolute = std::max(stats.max_absolute, error);
        sum_absolute += error;
        sum_squared += error * error;
        if (error > atol + rtol * std::abs(reference[i])) {
            stats.correct = false;
        }
    }

    stats.mean_absolute = sum_absolute / result.size();
    stats.rmse = std::sqrt(sum_squared / result.size());
    return stats;
}

void print_stats(const char* name, const ErrorStats& stats) {
    std::cout << name << ": " << (stats.correct ? "PASS" : "FAIL")
              << ", max_abs=" << stats.max_absolute
              << ", mean_abs=" << stats.mean_absolute
              << ", rmse=" << stats.rmse
              << ", non_finite=" << stats.non_finite << '\n';
}

template <typename Launch>
double benchmark_kernel_ms(Launch launch, int warmup, int repeats) {
    for (int i = 0; i < warmup; ++i) launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeats; ++i) launch();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return static_cast<double>(total_ms) / repeats;
}

int main() {
    // 多个 output tiles + 四轮 K tile，仍保持 CPU reference 运行较快。
    constexpr int M = 256;
    constexpr int N = 256;
    constexpr int K = 256;
    constexpr int WARMUP = 20;
    constexpr int REPEATS = 100;
    constexpr float ATOL = 1.0e-2f;
    constexpr float RTOL = 1.0e-2f;

    static_assert(M % WMMA_M == 0);
    static_assert(N % WMMA_N == 0);
    static_assert(K % WMMA_K == 0);

    std::vector<half> a(M * K);
    std::vector<half> b(N * K);  // B col-major: b[col * K + row]
    std::vector<float> scalar(M * N);
    std::vector<float> wmma_result(M * N);
    std::vector<float> staged_result(M * N);
    std::vector<float> reference(M * N);

    for (int row = 0; row < M; ++row) {
        for (int kk = 0; kk < K; ++kk) {
            const float value = ((row * 17 + kk * 13) % 31 - 15) * 0.03125f;
            a[row * K + kk] = __float2half(value);
        }
    }
    for (int col = 0; col < N; ++col) {
        for (int kk = 0; kk < K; ++kk) {
            const float value = ((col * 11 + kk * 7) % 29 - 14) * 0.03125f;
            b[col * K + kk] = __float2half(value);
        }
    }

    // Reference 从已经舍入到 FP16 的输入读取，使用 double 累加，避免把
    // 输入量化误差误算成 WMMA kernel 的误差。
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            double sum = 0.0;
            for (int kk = 0; kk < K; ++kk) {
                sum += static_cast<double>(
                           __half2float(a[row * K + kk])
                       ) * static_cast<double>(
                           __half2float(b[col * K + kk])
                       );
            }
            reference[row * N + col] = static_cast<float>(sum);
        }
    }

    half *d_a, *d_b;
    float* d_c;
    CUDA_CHECK(cudaMalloc(&d_a, a.size() * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_b, b.size() * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_c, wmma_result.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_a, a.data(), a.size() * sizeof(half),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b.data(), b.size() * sizeof(half),
                          cudaMemcpyHostToDevice));

    const dim3 scalar_block(16, 16);
    const dim3 scalar_grid((N + scalar_block.x - 1) / scalar_block.x,
                           (M + scalar_block.y - 1) / scalar_block.y);
    const dim3 wmma_block(WARP_SIZE);
    const dim3 wmma_grid(N / WMMA_N, M / WMMA_M);
    const dim3 staged_block(STAGED_THREADS);
    const dim3 staged_grid(
        (N + STAGED_BLOCK_N - 1) / STAGED_BLOCK_N,
        (M + STAGED_BLOCK_M - 1) / STAGED_BLOCK_M
    );

    auto launch_scalar = [&]() {
        gemm_fp16_scalar_kernel<<<scalar_grid, scalar_block>>>(
            d_a, d_b, d_c, M, N, K
        );
    };
    auto launch_wmma = [&]() {
        gemm_wmma_kernel<<<wmma_grid, wmma_block>>>(
            d_a, d_b, d_c, M, N, K
        );
    };
    auto launch_staged = [&]() {
        gemm_wmma_staged_kernel<<<staged_grid, staged_block>>>(
            d_a, d_b, d_c, M, N, K
        );
    };

    const double scalar_ms =
        benchmark_kernel_ms(launch_scalar, WARMUP, REPEATS);
    CUDA_CHECK(cudaMemcpy(scalar.data(), d_c,
                          scalar.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));

    const double wmma_ms =
        benchmark_kernel_ms(launch_wmma, WARMUP, REPEATS);
    CUDA_CHECK(cudaMemcpy(wmma_result.data(), d_c,
                          wmma_result.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));

    const double staged_ms =
        benchmark_kernel_ms(launch_staged, WARMUP, REPEATS);
    CUDA_CHECK(cudaMemcpy(staged_result.data(), d_c,
                          staged_result.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));

    const ErrorStats scalar_stats =
        compare_result(scalar, reference, ATOL, RTOL);
    const ErrorStats wmma_stats =
        compare_result(wmma_result, reference, ATOL, RTOL);
    const ErrorStats staged_stats =
        compare_result(staged_result, reference, ATOL, RTOL);
    print_stats("CUDA Core scalar FP16", scalar_stats);
    print_stats("Tensor Core WMMA G6", wmma_stats);
    print_stats("Tensor Core staged WMMA G7", staged_stats);

    const double flops = 2.0 * M * N * K;
    const double scalar_gflops = flops / (scalar_ms * 1.0e6);
    const double wmma_gflops = flops / (wmma_ms * 1.0e6);
    const double staged_gflops = flops / (staged_ms * 1.0e6);

#ifndef NDEBUG
    std::cout << "NOTE: Debug build; use Release for meaningful timing.\n";
#endif
    std::cout << std::fixed << std::setprecision(6)
              << "Shape: M=" << M << ", N=" << N << ", K=" << K << '\n'
              << "Input: FP16, accumulator/output: FP32\n"
              << "CUDA Core scalar: " << scalar_ms << " ms, "
              << scalar_gflops << " GFLOP/s\n"
              << "Tensor Core WMMA:  " << wmma_ms << " ms, "
              << wmma_gflops << " GFLOP/s\n"
              << "Staged WMMA G7:    " << staged_ms << " ms, "
              << staged_gflops << " GFLOP/s\n";
    if (wmma_ms > 0.0) {
        std::cout << "WMMA speedup over scalar: "
                  << scalar_ms / wmma_ms << " x\n";
    }
    if (staged_ms > 0.0) {
        std::cout << "Staged speedup over direct WMMA: "
                  << wmma_ms / staged_ms << " x\n";
    }
    std::cout << "These are correctness-first direct/staged WMMA kernels, "
                 "not cuBLAS-class implementations.\n";

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    return scalar_stats.correct && wmma_stats.correct && staged_stats.correct
        ? EXIT_SUCCESS
        : EXIT_FAILURE;
}
