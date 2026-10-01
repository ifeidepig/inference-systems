#include "common.cuh"
#include <chrono>
#include <iomanip>

// 入门 GEMM：行优先布局，A[M,K] * B[K,N] = C[M,N]。
// 一个线程计算一个输出元素；后续可改成 shared-memory tiling。
__global__ void gemm_kernel(const float* a, const float* b, float* c,
                            int m, int n, int k) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= m || col >= n) return;
    float sum = 0;
    for (int i = 0; i < k; ++i) sum += a[row * k + i] * b[i * n + col];
    c[row * n + col] = sum;
}

int main() {
    // 非整块尺寸用于检查边界条件。
    constexpr int M = 127, N = 193, K = 65;
    std::vector<float> a(M * K), b(K * N), c(M * N), ref(M * N);
    for (int i = 0; i < M * K; ++i) a[i] = (i % 19 - 9) * 0.1f;
    for (int i = 0; i < K * N; ++i) b[i] = (i % 13 - 6) * 0.1f;
    for (int row = 0; row < M; ++row)
        for (int col = 0; col < N; ++col) {
            double sum = 0;
            for (int i = 0; i < K; ++i)
                sum += double(a[row * K + i]) * b[i * N + col];
            ref[row * N + col] = float(sum);
        }

    // double 参考结果仅用于校验；计时用与 GPU 一致的 float 累加。
    using Clock = std::chrono::steady_clock;
    constexpr int CPU_REPEATS = 20, GPU_REPEATS = 100;
    std::vector<float> cpu_result(M * N);
    auto cpu_gemm = [&]() {
        for (int row = 0; row < M; ++row)
            for (int col = 0; col < N; ++col) {
                float sum = 0;
                for (int i = 0; i < K; ++i)
                    sum += a[row * K + i] * b[i * N + col];
                cpu_result[row * N + col] = sum;
            }
    };
    cpu_gemm(); // CPU 预热
    double cpu_ms = 0;
    // 在计时区间外读取每轮结果，避免重复计算被优化掉。
    volatile float checksum = 0;
    for (int repeat = 0; repeat < CPU_REPEATS; ++repeat) {
        auto begin = Clock::now();
        cpu_gemm();
        auto end = Clock::now();
        cpu_ms += std::chrono::duration<double, std::milli>(end - begin).count();
        checksum += cpu_result[repeat % cpu_result.size()];
    }
    (void)checksum;
    cpu_ms /= CPU_REPEATS;

    float *da, *db, *dc;
    CUDA_CHECK(cudaMalloc(&da, a.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&db, b.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dc, c.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(da, a.data(), a.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, b.data(), b.size() * sizeof(float), cudaMemcpyHostToDevice));
    dim3 block(16, 16), grid((N + 15) / 16, (M + 15) / 16);
    for (int warmup = 0; warmup < 10; ++warmup)
        gemm_kernel<<<grid, block>>>(da, db, dc, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // GPU 启动是异步的：用同一 stream 的 Event 记录设备时间。
    // 数据已经在显存中；此区间不包含分配和 CPU/GPU 数据拷贝。
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int repeat = 0; repeat < GPU_REPEATS; ++repeat)
        gemm_kernel<<<grid, block>>>(da, db, dc, M, N, K);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(stop));
    float gpu_total_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_total_ms, start, stop));
    double gpu_ms = gpu_total_ms / GPU_REPEATS;
    CUDA_CHECK(cudaMemcpy(c.data(), dc, c.size() * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "GEMM: ";
    bool ok = check_result(c, ref);
    std::cout << "CPU FP32: ";
    ok = check_result(cpu_result, ref) && ok;

    // 另测 CPU 视角的完整数据路径：H2D + kernel + D2H。
    // 使用已分配的缓冲区，不计 CUDA 初始化和内存分配。
    auto transfer_begin = Clock::now();
    for (int repeat = 0; repeat < CPU_REPEATS; ++repeat) {
        CUDA_CHECK(cudaMemcpy(da, a.data(), a.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(db, b.data(), b.size() * sizeof(float), cudaMemcpyHostToDevice));
        gemm_kernel<<<grid, block>>>(da, db, dc, M, N, K);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(c.data(), dc, c.size() * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    auto transfer_end = Clock::now();
    double gpu_transfer_ms = std::chrono::duration<double, std::milli>(
        transfer_end - transfer_begin).count() / CPU_REPEATS;
    std::cout << "GEMM with transfers: ";
    ok = check_result(c, ref) && ok;
#ifndef NDEBUG
    std::cout << "NOTE: Debug build; use Release without a debugger for timing.\n";
#endif
    std::cout << std::fixed << std::setprecision(6)
              << "Shape: M=" << M << ", N=" << N << ", K=" << K << '\n'
              << "CPU FP32 (single thread, average): " << cpu_ms << " ms\n"
              << "GPU kernel (average):             " << gpu_ms << " ms\n"
              << "GPU H2D + kernel + D2H (average): " << gpu_transfer_ms << " ms\n";
    if (gpu_ms > 0 && gpu_transfer_ms > 0)
        std::cout << "Speedup CPU / GPU kernel:         " << cpu_ms / gpu_ms << " x\n"
                  << "Speedup CPU / GPU with copies:    " << cpu_ms / gpu_transfer_ms << " x\n";
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(da));
    CUDA_CHECK(cudaFree(db));
    CUDA_CHECK(cudaFree(dc));
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
