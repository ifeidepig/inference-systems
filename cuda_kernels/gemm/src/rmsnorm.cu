#include "common.cuh"

constexpr int THREADS = 256;
// 每个 block 处理一行；启动时必须使用 THREADS 个线程。
// y[r,j] = x[r,j] * rsqrt(mean(x[r,:]^2) + eps) * weight[j]
__global__ void rmsnorm_kernel(const float* x, const float* weight, float* y,
                               int columns, float eps) {
    __shared__ float sums[THREADS];
    int tid = threadIdx.x;
    int offset = blockIdx.x * columns;
    float sum = 0;
    for (int j = tid; j < columns; j += blockDim.x) {
        float value = x[offset + j];
        sum += value * value;
    }
    sums[tid] = sum;
    __syncthreads();
    for (int stride = THREADS / 2; stride > 0; stride /= 2) {
        if (tid < stride) sums[tid] += sums[tid + stride];
        __syncthreads();
    }
    float inv_rms = rsqrtf(sums[0] / columns + eps);
    for (int j = tid; j < columns; j += blockDim.x)
        y[offset + j] = x[offset + j] * inv_rms * weight[j];
}

int main() {
    constexpr int ROWS = 32, COLS = 1003;
    constexpr float EPS = 1e-5f;
    std::vector<float> x(ROWS * COLS), weight(COLS), y(ROWS * COLS), ref(ROWS * COLS);
    for (int i = 0; i < ROWS * COLS; ++i) x[i] = (i % 31 - 15) * 0.1f;
    for (int j = 0; j < COLS; ++j) weight[j] = 0.5f + (j % 7) * 0.1f;
    for (int row = 0; row < ROWS; ++row) {
        double sum = 0;
        for (int j = 0; j < COLS; ++j) {
            double value = x[row * COLS + j];
            sum += value * value;
        }
        double inv_rms = 1.0 / std::sqrt(sum / COLS + EPS);
        for (int j = 0; j < COLS; ++j)
            ref[row * COLS + j] = float(x[row * COLS + j] * inv_rms * weight[j]);
    }
    float *dx, *dw, *dy;
    CUDA_CHECK(cudaMalloc(&dx, x.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dw, weight.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dy, y.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dx, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dw, weight.data(), weight.size() * sizeof(float), cudaMemcpyHostToDevice));
    rmsnorm_kernel<<<ROWS, THREADS>>>(dx, dw, dy, COLS, EPS);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(y.data(), dy, y.size() * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "RMSNorm: ";
    bool ok = check_result(y, ref);
    CUDA_CHECK(cudaFree(dx));
    CUDA_CHECK(cudaFree(dw));
    CUDA_CHECK(cudaFree(dy));
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
