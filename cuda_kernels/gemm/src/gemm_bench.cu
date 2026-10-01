#include "common.cuh"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>

#include <algorithm>
#include <iomanip>
#include <numeric>
#include <string>
#include <type_traits>

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t status = (call); \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << __FILE__ << ':' << __LINE__ \
                  << ": cuBLAS error " << static_cast<int>(status) << '\n'; \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

namespace {

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
constexpr int WARP_SIZE = 32;

constexpr int STAGED_BLOCK_M = 32;
constexpr int STAGED_BLOCK_N = 64;
constexpr int STAGED_BLOCK_K = 16;
constexpr int STAGED_WARPS_M = 2;
constexpr int STAGED_WARPS_N = 4;
constexpr int STAGED_WARPS = STAGED_WARPS_M * STAGED_WARPS_N;
constexpr int STAGED_THREADS = STAGED_WARPS * WARP_SIZE;

struct Shape {
    int m;
    int n;
    int k;
    const char* label;
};

const std::vector<Shape> kDefaultShapes = {
    {1, 6144, 1024, "qwen35_decode_qkv"},
    {16, 6144, 1024, "qwen35_prefill16_qkv"},
    {64, 6144, 1024, "qwen35_prefill64_qkv"},
    {256, 6144, 1024, "qwen35_prefill256_qkv"},
    {16, 1024, 2048, "qwen35_prefill16_out"},
    {64, 1024, 2048, "qwen35_prefill64_out"},
    {16, 7168, 1024, "qwen35_prefill16_gate_up"},
    {64, 7168, 1024, "qwen35_prefill64_gate_up"},
    {64, 1024, 3584, "qwen35_prefill64_down"},
    {256, 256, 256, "square256"},
    {512, 512, 512, "square512"},
};

template <typename T>
struct TypeTraits;

template <>
struct TypeTraits<half> {
    static constexpr cudaDataType_t cuda_type = CUDA_R_16F;
    static const char* name() { return "fp16"; }
    __host__ __device__ static half convert(float value) {
        return __float2half(value);
    }
};

template <>
struct TypeTraits<__nv_bfloat16> {
    static constexpr cudaDataType_t cuda_type = CUDA_R_16BF;
    static const char* name() { return "bf16"; }
    __host__ __device__ static __nv_bfloat16 convert(float value) {
        return __float2bfloat16(value);
    }
};

template <typename T>
__global__ void wmma_direct_kernel(
    const T* __restrict__ a,
    const T* __restrict__ b,
    float* __restrict__ c,
    int m,
    int n,
    int k
) {
    using namespace nvcuda;
    const int tile_row = blockIdx.y * WMMA_M;
    const int tile_col = blockIdx.x * WMMA_N;
    if (tile_row + WMMA_M > m || tile_col + WMMA_N > n) return;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                   T, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                   T, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K,
                   float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

    for (int k0 = 0; k0 < k; k0 += WMMA_K) {
        const T* a_tile = a + tile_row * k + k0;
        const T* b_tile = b + k0 * n + tile_col;
        wmma::load_matrix_sync(a_frag, a_tile, k);
        wmma::load_matrix_sync(b_frag, b_tile, n);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }
    wmma::store_matrix_sync(
        c + tile_row * n + tile_col,
        acc_frag,
        n,
        wmma::mem_row_major
    );
}

template <typename T>
__global__ void wmma_staged_kernel(
    const T* __restrict__ a,
    const T* __restrict__ b,
    float* __restrict__ c,
    int m,
    int n,
    int k
) {
    using namespace nvcuda;
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int warp_m = warp_id / STAGED_WARPS_N;
    const int warp_n = warp_id % STAGED_WARPS_N;
    const int block_row = blockIdx.y * STAGED_BLOCK_M;
    const int block_col = blockIdx.x * STAGED_BLOCK_N;
    const int warp_row = block_row + warp_m * WMMA_M;
    const int warp_col = block_col + warp_n * WMMA_N;

    __shared__ __align__(32) T shared_a[STAGED_BLOCK_M][STAGED_BLOCK_K];
    __shared__ __align__(32) T shared_b[STAGED_BLOCK_K][STAGED_BLOCK_N];

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                   T, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                   T, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K,
                   float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

    for (int k0 = 0; k0 < k; k0 += STAGED_BLOCK_K) {
        for (int index = tid;
             index < STAGED_BLOCK_M * STAGED_BLOCK_K;
             index += STAGED_THREADS) {
            const int local_row = index / STAGED_BLOCK_K;
            const int local_k = index % STAGED_BLOCK_K;
            const int global_row = block_row + local_row;
            const int global_k = k0 + local_k;
            shared_a[local_row][local_k] =
                global_row < m && global_k < k
                    ? a[global_row * k + global_k]
                    : TypeTraits<T>::convert(0.0f);
        }
        for (int index = tid;
             index < STAGED_BLOCK_K * STAGED_BLOCK_N;
             index += STAGED_THREADS) {
            const int local_k = index / STAGED_BLOCK_N;
            const int local_col = index % STAGED_BLOCK_N;
            const int global_k = k0 + local_k;
            const int global_col = block_col + local_col;
            shared_b[local_k][local_col] =
                global_k < k && global_col < n
                    ? b[global_k * n + global_col]
                    : TypeTraits<T>::convert(0.0f);
        }
        __syncthreads();

        wmma::load_matrix_sync(
            a_frag,
            &shared_a[warp_m * WMMA_M][0],
            STAGED_BLOCK_K
        );
        wmma::load_matrix_sync(
            b_frag,
            &shared_b[0][warp_n * WMMA_N],
            STAGED_BLOCK_N
        );
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        __syncthreads();
    }

    if (warp_row + WMMA_M <= m && warp_col + WMMA_N <= n) {
        wmma::store_matrix_sync(
            c + warp_row * n + warp_col,
            acc_frag,
            n,
            wmma::mem_row_major
        );
    }
}

template <typename Launch>
double benchmark_ms(Launch launch, int warmup, int repeats, int trials) {
    for (int i = 0; i < warmup; ++i) launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> samples;
    samples.reserve(trials);
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    for (int trial = 0; trial < trials; ++trial) {
        CUDA_CHECK(cudaEventRecord(start));
        for (int repeat = 0; repeat < repeats; ++repeat) launch();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsed = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
        samples.push_back(static_cast<double>(elapsed) / repeats);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

struct ErrorStats {
    double max_absolute = 0.0;
    double mean_absolute = 0.0;
    bool correct = true;
};

ErrorStats compare(
    const std::vector<float>& actual,
    const std::vector<float>& reference
) {
    ErrorStats stats;
    double sum = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double error = std::abs(
            static_cast<double>(actual[i]) - reference[i]
        );
        stats.max_absolute = std::max(stats.max_absolute, error);
        sum += error;
        if (!std::isfinite(actual[i]) ||
            error > 5.0e-2 + 1.0e-2 * std::abs(reference[i])) {
            stats.correct = false;
        }
    }
    stats.mean_absolute = sum / actual.size();
    return stats;
}

bool supports_wmma(const Shape& shape) {
    return shape.m % WMMA_M == 0 &&
           shape.n % WMMA_N == 0 &&
           shape.k % WMMA_K == 0;
}

enum class DispatchPath { Cublas, Direct, Staged };

template <typename T>
DispatchPath choose_path(const Shape& shape) {
    if (!supports_wmma(shape)) return DispatchPath::Cublas;
    // Local measurements show that direct WMMA is useful only for the
    // small-M / wide-N projection regime.  Larger M and narrow output
    // projections stay on cuBLAS; staged WMMA remains an explicit negative
    // experiment rather than a dispatch candidate.
    if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        if (shape.m == 16 && shape.k == 1024 && shape.n >= 4096) {
            return DispatchPath::Direct;
        }
    }
    return DispatchPath::Cublas;
}

const char* path_name(DispatchPath path) {
    switch (path) {
        case DispatchPath::Cublas: return "cublas";
        case DispatchPath::Direct: return "wmma_direct";
        case DispatchPath::Staged: return "wmma_staged";
    }
    return "unknown";
}

template <typename T>
bool run_shape(
    cublasHandle_t handle,
    const Shape& shape,
    int trials,
    int repeat_override
) {
    const size_t a_elements = static_cast<size_t>(shape.m) * shape.k;
    const size_t b_elements = static_cast<size_t>(shape.k) * shape.n;
    const size_t c_elements = static_cast<size_t>(shape.m) * shape.n;
    std::vector<T> a(a_elements);
    std::vector<T> b(b_elements);
    for (size_t index = 0; index < a.size(); ++index) {
        a[index] = TypeTraits<T>::convert(
            (static_cast<int>((index * 17 + 3) % 31) - 15) * 0.03125f
        );
    }
    for (size_t index = 0; index < b.size(); ++index) {
        b[index] = TypeTraits<T>::convert(
            (static_cast<int>((index * 11 + 5) % 29) - 14) * 0.03125f
        );
    }

    T* d_a = nullptr;
    T* d_b = nullptr;
    float* d_reference = nullptr;
    float* d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, a.size() * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_b, b.size() * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_reference, c_elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, c_elements * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(
        d_a, a.data(), a.size() * sizeof(T), cudaMemcpyHostToDevice
    ));
    CUDA_CHECK(cudaMemcpy(
        d_b, b.data(), b.size() * sizeof(T), cudaMemcpyHostToDevice
    ));

    const float alpha = 1.0f;
    const float beta = 0.0f;
    auto launch_cublas = [&]() {
        // Row-major C=A*B is column-major C^T=B^T*A^T.
        CUBLAS_CHECK(cublasGemmEx(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            shape.n,
            shape.m,
            shape.k,
            &alpha,
            d_b,
            TypeTraits<T>::cuda_type,
            shape.n,
            d_a,
            TypeTraits<T>::cuda_type,
            shape.k,
            &beta,
            d_reference,
            CUDA_R_32F,
            shape.n,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        ));
    };

    const dim3 direct_block(WARP_SIZE);
    const dim3 direct_grid(
        (shape.n + WMMA_N - 1) / WMMA_N,
        (shape.m + WMMA_M - 1) / WMMA_M
    );
    auto launch_direct = [&]() {
        wmma_direct_kernel<<<direct_grid, direct_block>>>(
            d_a, d_b, d_output, shape.m, shape.n, shape.k
        );
    };

    const dim3 staged_block(STAGED_THREADS);
    const dim3 staged_grid(
        (shape.n + STAGED_BLOCK_N - 1) / STAGED_BLOCK_N,
        (shape.m + STAGED_BLOCK_M - 1) / STAGED_BLOCK_M
    );
    auto launch_staged = [&]() {
        wmma_staged_kernel<<<staged_grid, staged_block>>>(
            d_a, d_b, d_output, shape.m, shape.n, shape.k
        );
    };

    const double flops = 2.0 * shape.m * shape.n * shape.k;
    const int repeats = repeat_override > 0
        ? repeat_override
        : std::clamp(
              static_cast<int>(2.0e10 / std::max(flops, 1.0)),
              10,
              200
          );
    const int warmup = repeat_override > 0 ? 1 : 10;
    const double cublas_ms = benchmark_ms(
        launch_cublas, warmup, repeats, trials
    );
    launch_cublas();
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> reference(c_elements);
    CUDA_CHECK(cudaMemcpy(
        reference.data(), d_reference, c_elements * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    bool ok = true;
    double direct_ms = 0.0;
    double staged_ms = 0.0;
    ErrorStats direct_error;
    ErrorStats staged_error;
    if (supports_wmma(shape)) {
        direct_ms = benchmark_ms(launch_direct, warmup, repeats, trials);
        launch_direct();
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<float> output(c_elements);
        CUDA_CHECK(cudaMemcpy(
            output.data(), d_output, c_elements * sizeof(float),
            cudaMemcpyDeviceToHost
        ));
        direct_error = compare(output, reference);

        staged_ms = benchmark_ms(launch_staged, warmup, repeats, trials);
        launch_staged();
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(
            output.data(), d_output, c_elements * sizeof(float),
            cudaMemcpyDeviceToHost
        ));
        staged_error = compare(output, reference);
        ok = direct_error.correct && staged_error.correct;
    }

    const DispatchPath selected = choose_path<T>(shape);
    const double selected_ms =
        selected == DispatchPath::Cublas ? cublas_ms :
        selected == DispatchPath::Direct ? direct_ms : staged_ms;
    const double selected_ratio = cublas_ms / selected_ms;

    auto tflops = [&](double milliseconds) {
        return flops / (milliseconds * 1.0e9);
    };
    std::cout << TypeTraits<T>::name() << ','
              << shape.label << ','
              << shape.m << ',' << shape.n << ',' << shape.k << ','
              << repeats << ','
              << std::fixed << std::setprecision(6)
              << cublas_ms << ',' << tflops(cublas_ms) << ',';
    if (supports_wmma(shape)) {
        std::cout << direct_ms << ',' << tflops(direct_ms) << ','
                  << staged_ms << ',' << tflops(staged_ms) << ','
                  << direct_error.max_absolute << ','
                  << staged_error.max_absolute << ',';
    } else {
        std::cout << "NA,NA,NA,NA,NA,NA,";
    }
    std::cout << path_name(selected) << ','
              << selected_ms << ',' << selected_ratio << ','
              << (ok ? "PASS" : "FAIL") << '\n';

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_reference));
    CUDA_CHECK(cudaFree(d_output));
    return ok;
}

}  // namespace

int main(int argc, char** argv) {
    std::string dtype = "both";
    std::string label_filter;
    int trials = 5;
    int repeat_override = -1;
    bool quick = false;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--dtype" && index + 1 < argc) {
            dtype = argv[++index];
        } else if (argument == "--label" && index + 1 < argc) {
            label_filter = argv[++index];
        } else if (argument == "--trials" && index + 1 < argc) {
            trials = std::stoi(argv[++index]);
        } else if (argument == "--repeats" && index + 1 < argc) {
            repeat_override = std::stoi(argv[++index]);
        } else if (argument == "--quick") {
            quick = true;
        } else {
            std::cerr << "Usage: gemm_bench [--dtype fp16|bf16|both] "
                         "[--label NAME] [--trials N] [--repeats N] "
                         "[--quick]\n";
            return EXIT_FAILURE;
        }
    }
    if (dtype != "fp16" && dtype != "bf16" && dtype != "both") {
        std::cerr << "dtype must be fp16, bf16, or both\n";
        return EXIT_FAILURE;
    }
    if (trials <= 0) {
        std::cerr << "trials must be positive\n";
        return EXIT_FAILURE;
    }
    if (repeat_override == 0 || repeat_override < -1) {
        std::cerr << "repeats must be positive when provided\n";
        return EXIT_FAILURE;
    }

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

    std::cout << "dtype,label,M,N,K,repeats,cublas_ms,cublas_tflops,"
                 "direct_ms,direct_tflops,staged_ms,staged_tflops,"
                 "direct_max_abs,staged_max_abs,dispatch,dispatch_ms,"
                 "dispatch_vs_cublas,status\n";
    bool ok = true;
    bool matched = false;
    const size_t shape_count = quick ? 3 : kDefaultShapes.size();
    for (size_t index = 0; index < shape_count; ++index) {
        const Shape& shape = kDefaultShapes[index];
        if (!label_filter.empty() && label_filter != shape.label) continue;
        matched = true;
        if (dtype == "fp16" || dtype == "both") {
            ok = run_shape<half>(
                handle, shape, trials, repeat_override
            ) && ok;
        }
        if (dtype == "bf16" || dtype == "both") {
            ok = run_shape<__nv_bfloat16>(
                handle, shape, trials, repeat_override
            ) && ok;
        }
    }
    if (!matched) {
        std::cerr << "no shape matched --label " << label_filter << '\n';
        ok = false;
    }
    CUBLAS_CHECK(cublasDestroy(handle));
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
