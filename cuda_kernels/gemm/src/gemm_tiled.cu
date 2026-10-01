#include "common.cuh"

#include <cublas_v2.h>
#include <iomanip>

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t status = (call); \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << __FILE__ << ':' << __LINE__ \
                  << ": cuBLAS error " << static_cast<int>(status) << '\n'; \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

// 这个文件只演示 GEMM 从 naive 到最基础 shared-memory tiling 的一步。
// 行优先布局：A[M,K] * B[K,N] = C[M,N]。
//
// 为了让线程和矩阵位置一一对应，当前版本故意使用：
//   1. 一个 CUDA block 有 TILE x TILE 个线程；
//   2. 一个 CUDA block 计算 C 的 TILE x TILE 个元素；
//   3. 一个线程只计算一个 C[row, col]。
// 后续 register tiling 才会让一个线程计算多个输出元素。
constexpr int TILE = 16;

// Register-tiled 版本仍使用 16 x 16 个线程，但每个线程计算 2 x 2 个
// 输出，因此一个 block 可以计算 32 x 32 个 C 元素。
constexpr int REG_THREADS_X = 16;
constexpr int REG_THREADS_Y = 16;
constexpr int THREAD_TILE_M = 2;
constexpr int THREAD_TILE_N = 2;
constexpr int REG_BLOCK_M = REG_THREADS_Y * THREAD_TILE_M;  // 32
constexpr int REG_BLOCK_N = REG_THREADS_X * THREAD_TILE_N;  // 32
constexpr int REG_BLOCK_K = 8;

// 对照组：每个线程直接从 global memory 读取 A、B。
__global__ void gemm_naive_kernel(const float* a, const float* b, float* c,
                                  int m, int n, int k) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int row = blockIdx.y * blockDim.y + ty;
    const int col = blockIdx.x * blockDim.x + tx;

    if (row >= m || col >= n) return;

    float sum = 0.0f;
    for (int kk = 0; kk < k; ++kk) {
        sum += a[row * k + kk] * b[kk * n + col];
    }
    c[row * n + col] = sum;
}

// 基础 tiled GEMM：一个线程仍然只计算一个 C 元素。
// 区别是同一个 block 的线程协作把 A、B 的小块搬进 shared memory，
// 然后反复复用这些片上数据。
__global__ void gemm_tiled_kernel(const float* a, const float* b, float* c,
                                  int m, int n, int k) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // 当前 block 负责 C 中一个 TILE x TILE 的区域。
    // blockIdx 给出区域编号，tx/ty 给出线程在区域内的位置。
    const int row = blockIdx.y * TILE + ty;
    const int col = blockIdx.x * TILE + tx;

    // 每个 block 都有自己独立的一份 As、Bs；同一 block 内所有线程共享。
    // 它们不是完整 A、B，只保存当前 K 分块对应的 TILE x TILE 小块。
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    float sum = 0.0f;

    // 每次沿 K 维处理 TILE 个元素：0~15、16~31、32~47……
    for (int k0 = 0; k0 < k; k0 += TILE) {
        // 256 个线程协作加载 A tile。越界位置必须写 0，不能提前 return，
        // 因为整个 block 后面都必须到达 __syncthreads()。
        if (row < m && k0 + tx < k) {
            As[ty][tx] = a[row * k + (k0 + tx)];
        } else {
            As[ty][tx] = 0.0f;
        }

        // 256 个线程协作加载 B tile。
        if (k0 + ty < k && col < n) {
            Bs[ty][tx] = b[(k0 + ty) * n + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        // 保证 As、Bs 的全部元素都已经写完，才能开始读取。
        __syncthreads();

        // 当前线程读取 As 的第 ty 行和 Bs 的第 tx 列，完成当前 K tile
        // 上的点积。sum 保存在当前线程自己的寄存器中。
        for (int kk = 0; kk < TILE; ++kk) {
            sum += As[ty][kk] * Bs[kk][tx];
        }

        // 保证所有线程都用完当前 tile，才能在下一轮覆盖 As、Bs。
        __syncthreads();
    }

    if (row < m && col < n) {
        c[row * n + col] = sum;
    }
}

// Register-tiled GEMM：一个线程计算一个 2 x 2 的 C 子块。
//
// 一个 CUDA block：16 x 16 = 256 个线程
// 一个 thread tile：2 x 2 个输出
// 一个 C block tile：(16 x 2) x (16 x 2) = 32 x 32
// 一个 K tile：8
//
// 每个 kk 中，线程从 shared memory 读取 2 个 A 和 2 个 B 到寄存器，
// 再通过 2 x 2 外积完成 4 次 FMA。
__global__ void gemm_register_tiled_kernel(
    const float* a,
    const float* b,
    float* c,
    int m,
    int n,
    int k
) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // 当前 block 负责的 32 x 32 输出区域的全局起点。
    const int block_row = blockIdx.y * REG_BLOCK_M;
    const int block_col = blockIdx.x * REG_BLOCK_N;

    // 当前线程在这个输出区域中负责一个 2 x 2 子块。
    const int row0 = block_row + ty * THREAD_TILE_M;
    const int row1 = row0 + 1;
    const int col0 = block_col + tx * THREAD_TILE_N;
    const int col1 = col0 + 1;

    // 当前 K 分块需要：
    // A tile: 32 x 8
    // B tile:  8 x 32
    __shared__ float As[REG_BLOCK_M][REG_BLOCK_K];
    __shared__ float Bs[REG_BLOCK_K][REG_BLOCK_N];

    // 四个输出累加器属于当前线程，通常会放在寄存器中。
    float accum00 = 0.0f;
    float accum01 = 0.0f;
    float accum10 = 0.0f;
    float accum11 = 0.0f;

    for (int k0 = 0; k0 < k; k0 += REG_BLOCK_K) {
        // A tile 是 32 x 8。
        // tx=0~7 的线程负责 K 方向的 8 个位置；每个 ty 负责两行。
        if (tx < REG_BLOCK_K) {
            const int global_k = k0 + tx;
            As[ty * THREAD_TILE_M][tx] =
                row0 < m && global_k < k
                    ? a[row0 * k + global_k]
                    : 0.0f;
            As[ty * THREAD_TILE_M + 1][tx] =
                row1 < m && global_k < k
                    ? a[row1 * k + global_k]
                    : 0.0f;
        }

        // B tile 是 8 x 32。
        // ty=0~7 的线程负责 K 方向的 8 个位置；每个 tx 负责两列。
        if (ty < REG_BLOCK_K) {
            const int global_k = k0 + ty;
            Bs[ty][tx * THREAD_TILE_N] =
                global_k < k && col0 < n
                    ? b[global_k * n + col0]
                    : 0.0f;
            Bs[ty][tx * THREAD_TILE_N + 1] =
                global_k < k && col1 < n
                    ? b[global_k * n + col1]
                    : 0.0f;
        }

        __syncthreads();

        // 对当前 K tile 做 2 x 2 外积。
        for (int kk = 0; kk < REG_BLOCK_K; ++kk) {
            const float a0 = As[ty * THREAD_TILE_M][kk];
            const float a1 = As[ty * THREAD_TILE_M + 1][kk];
            const float b0 = Bs[kk][tx * THREAD_TILE_N];
            const float b1 = Bs[kk][tx * THREAD_TILE_N + 1];

            accum00 += a0 * b0;
            accum01 += a0 * b1;
            accum10 += a1 * b0;
            accum11 += a1 * b1;
        }

        __syncthreads();
    }

    // 四个输出分别检查边界；不能假设整个 2 x 2 子块都有效。
    if (row0 < m && col0 < n) c[row0 * n + col0] = accum00;
    if (row0 < m && col1 < n) c[row0 * n + col1] = accum01;
    if (row1 < m && col0 < n) c[row1 * n + col0] = accum10;
    if (row1 < m && col1 < n) c[row1 * n + col1] = accum11;
}

// 复用 CUDA Event 计时代码；Launch 是一个只负责启动 kernel 的 host lambda。
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
    // 三个维度都故意不是 TILE 的整数倍，用于验证边界处理。
    constexpr int M = 127;
    constexpr int N = 193;
    constexpr int K = 65;
    constexpr int WARMUP = 10;
    constexpr int REPEATS = 100;

    static_assert(TILE * TILE <= 1024,
                  "A CUDA block cannot exceed 1024 threads on this target");

    std::vector<float> a(M * K), b(K * N);
    std::vector<float> naive(M * N), tiled(M * N);
    std::vector<float> register_tiled(M * N), cublas_result(M * N);
    std::vector<float> reference(M * N);

    for (int i = 0; i < M * K; ++i) {
        a[i] = (i % 19 - 9) * 0.1f;
    }
    for (int i = 0; i < K * N; ++i) {
        b[i] = (i % 13 - 6) * 0.1f;
    }

    // CPU double 累加只作为正确性参考，不参与性能比较。
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            double sum = 0.0;
            for (int kk = 0; kk < K; ++kk) {
                sum += static_cast<double>(a[row * K + kk]) *
                       static_cast<double>(b[kk * N + col]);
            }
            reference[row * N + col] = static_cast<float>(sum);
        }
    }

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, a.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, b.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, tiled.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_a, a.data(), a.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b.data(), b.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    // x 方向覆盖 C 的 N 列，y 方向覆盖 C 的 M 行。
    const dim3 block(TILE, TILE);
    const dim3 grid((N + TILE - 1) / TILE,
                    (M + TILE - 1) / TILE);
    const dim3 register_block(REG_THREADS_X, REG_THREADS_Y);
    const dim3 register_grid((N + REG_BLOCK_N - 1) / REG_BLOCK_N,
                             (M + REG_BLOCK_M - 1) / REG_BLOCK_M);

    auto launch_naive = [&]() {
        gemm_naive_kernel<<<grid, block>>>(d_a, d_b, d_c, M, N, K);
    };
    auto launch_tiled = [&]() {
        gemm_tiled_kernel<<<grid, block>>>(d_a, d_b, d_c, M, N, K);
    };
    auto launch_register_tiled = [&]() {
        gemm_register_tiled_kernel<<<register_grid, register_block>>>(
            d_a, d_b, d_c, M, N, K
        );
    };
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    const float alpha = 1.0f;
    const float beta = 0.0f;
    auto launch_cublas = [&]() {
        // Row-major C=A*B is column-major C^T=B^T*A^T.
        CUBLAS_CHECK(cublasSgemm(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            N,
            M,
            K,
            &alpha,
            d_b,
            N,
            d_a,
            K,
            &beta,
            d_c,
            N
        ));
    };

    const double naive_ms =
        benchmark_kernel_ms(launch_naive, WARMUP, REPEATS);
    CUDA_CHECK(cudaMemcpy(naive.data(), d_c, naive.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));

    const double tiled_ms =
        benchmark_kernel_ms(launch_tiled, WARMUP, REPEATS);
    CUDA_CHECK(cudaMemcpy(tiled.data(), d_c, tiled.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));

    const double register_tiled_ms =
        benchmark_kernel_ms(launch_register_tiled, WARMUP, REPEATS);
    CUDA_CHECK(cudaMemcpy(register_tiled.data(), d_c,
                          register_tiled.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));

    const double cublas_ms =
        benchmark_kernel_ms(launch_cublas, WARMUP, REPEATS);
    CUDA_CHECK(cudaMemcpy(cublas_result.data(), d_c,
                          cublas_result.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));

    std::cout << "Naive GEMM: ";
    bool ok = check_result(naive, reference);
    std::cout << "Tiled GEMM: ";
    ok = check_result(tiled, reference) && ok;
    std::cout << "Register-tiled GEMM: ";
    ok = check_result(register_tiled, reference) && ok;
    std::cout << "cuBLAS SGEMM: ";
    ok = check_result(cublas_result, reference) && ok;

    const double flops = 2.0 * M * N * K;
    const double naive_gflops = flops / (naive_ms * 1.0e6);
    const double tiled_gflops = flops / (tiled_ms * 1.0e6);
    const double register_tiled_gflops =
        flops / (register_tiled_ms * 1.0e6);
    const double cublas_gflops = flops / (cublas_ms * 1.0e6);

#ifndef NDEBUG
    std::cout << "NOTE: Debug build; use Release for meaningful timing.\n";
#endif
    std::cout << std::fixed << std::setprecision(6)
              << "Shape: M=" << M << ", N=" << N << ", K=" << K << '\n'
              << "Thread block: " << TILE << " x " << TILE
              << " = " << TILE * TILE << " threads\n"
              << "Naive: " << naive_ms << " ms, "
              << naive_gflops << " GFLOP/s\n"
              << "Tiled: " << tiled_ms << " ms, "
              << tiled_gflops << " GFLOP/s\n"
              << "Register tiled (2x2/thread): " << register_tiled_ms
              << " ms, " << register_tiled_gflops << " GFLOP/s\n"
              << "cuBLAS SGEMM: " << cublas_ms << " ms, "
              << cublas_gflops << " GFLOP/s\n";
    if (tiled_ms > 0.0) {
        std::cout << "Tiled speedup over naive: "
                  << naive_ms / tiled_ms << " x\n";
    }
    if (register_tiled_ms > 0.0) {
        std::cout << "Register-tiled speedup over basic tiled: "
                  << tiled_ms / register_tiled_ms << " x\n";
    }
    if (cublas_ms > 0.0) {
        std::cout << "Naive / cuBLAS performance: "
                  << cublas_ms / naive_ms << " x\n"
                  << "Tiled / cuBLAS performance: "
                  << cublas_ms / tiled_ms << " x\n"
                  << "Register-tiled / cuBLAS performance: "
                  << cublas_ms / register_tiled_ms << " x\n";
    }
    std::cout << "This small, irregular shape is for learning and correctness; "
                 "tiled is not guaranteed to be faster.\n";

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    CUBLAS_CHECK(cublasDestroy(handle));
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
