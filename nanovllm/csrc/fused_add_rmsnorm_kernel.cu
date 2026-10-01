#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <limits>
#include <tuple>

namespace {

__forceinline__ __device__ float warp_reduce_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

template <typename scalar_t, int hidden_size, int block_size>
__global__ void rmsnorm_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    float epsilon
) {
    static_assert(hidden_size % block_size == 0);
    constexpr int items_per_thread = hidden_size / block_size;
    constexpr int num_warps = block_size / 32;

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp_id = tid >> 5;
    int64_t row_offset = static_cast<int64_t>(row) * hidden_size;

    float cached_x[items_per_thread];
    float local_sum = 0.0f;

    #pragma unroll
    for (int item = 0; item < items_per_thread; ++item) {
        int column = tid + item * block_size;
        int64_t index = row_offset + column;
        float value = static_cast<float>(x[index]);
        cached_x[item] = value;
        local_sum = fmaf(value, value, local_sum);
    }

    local_sum = warp_reduce_sum(local_sum);
    __shared__ float warp_sums[num_warps];
    __shared__ float shared_scale;

    if (lane == 0) {
        warp_sums[warp_id] = local_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        float block_sum = lane < num_warps ? warp_sums[lane] : 0.0f;
        block_sum = warp_reduce_sum(block_sum);
        if (lane == 0) {
            shared_scale = rsqrtf(
                block_sum / static_cast<float>(hidden_size) + epsilon
            );
        }
    }
    __syncthreads();

    float scale = shared_scale;
    #pragma unroll
    for (int item = 0; item < items_per_thread; ++item) {
        int column = tid + item * block_size;
        int64_t index = row_offset + column;
        scalar_t rounded_normalized = static_cast<scalar_t>(
            cached_x[item] * scale
        );
        float product = static_cast<float>(rounded_normalized)
            * static_cast<float>(weight[column]);
        output[index] = static_cast<scalar_t>(product);
    }
}

template <typename scalar_t, int hidden_size, int block_size>
__global__ void fused_add_rmsnorm_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ residual,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    scalar_t* __restrict__ residual_output,
    float epsilon
) {
    static_assert(hidden_size % block_size == 0);
    constexpr int items_per_thread = hidden_size / block_size;
    constexpr int num_warps = block_size / 32;

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp_id = tid >> 5;
    int64_t row_offset = static_cast<int64_t>(row) * hidden_size;

    // nano-vLLM computes the variance from the unrounded FP32 residual sum.
    // Cache that FP32 value first; packed low-precision caching is a later,
    // explicitly different numerical/performance experiment.
    float cached_h[items_per_thread];
    float local_sum = 0.0f;

    #pragma unroll
    for (int item = 0; item < items_per_thread; ++item) {
        int column = tid + item * block_size;
        int64_t index = row_offset + column;
        float h = static_cast<float>(x[index])
            + static_cast<float>(residual[index]);

        cached_h[item] = h;
        residual_output[index] = static_cast<scalar_t>(h);
        local_sum = fmaf(h, h, local_sum);
    }

    local_sum = warp_reduce_sum(local_sum);

    __shared__ float warp_sums[num_warps];
    __shared__ float shared_scale;

    if (lane == 0) {
        warp_sums[warp_id] = local_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        float block_sum = lane < num_warps ? warp_sums[lane] : 0.0f;
        block_sum = warp_reduce_sum(block_sum);
        if (lane == 0) {
            shared_scale = rsqrtf(
                block_sum / static_cast<float>(hidden_size) + epsilon
            );
        }
    }
    __syncthreads();

    float scale = shared_scale;
    #pragma unroll
    for (int item = 0; item < items_per_thread; ++item) {
        int column = tid + item * block_size;
        int64_t index = row_offset + column;

        // Match: normalized = normalized.to(orig_dtype).mul_(weight)
        scalar_t rounded_normalized = static_cast<scalar_t>(
            cached_h[item] * scale
        );
        float product = static_cast<float>(rounded_normalized)
            * static_cast<float>(weight[column]);
        output[index] = static_cast<scalar_t>(product);
    }
}

template <typename scalar_t, int hidden_size, int block_size>
void launch_fused_add_rmsnorm(
    const torch::Tensor& x,
    const torch::Tensor& residual,
    const torch::Tensor& weight,
    torch::Tensor& output,
    torch::Tensor& residual_output,
    float epsilon,
    cudaStream_t stream
) {
    int64_t rows = x.numel() / hidden_size;
    TORCH_CHECK(rows <= std::numeric_limits<int>::max(), "too many rows");

    fused_add_rmsnorm_kernel<scalar_t, hidden_size, block_size>
        <<<static_cast<int>(rows), block_size, 0, stream>>>(
            x.data_ptr<scalar_t>(),
            residual.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            residual_output.data_ptr<scalar_t>(),
            epsilon
        );
}

template <typename scalar_t, int hidden_size, int block_size>
void launch_rmsnorm(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    torch::Tensor& output,
    float epsilon,
    cudaStream_t stream
) {
    int64_t rows = x.numel() / hidden_size;
    TORCH_CHECK(rows <= std::numeric_limits<int>::max(), "too many rows");
    rmsnorm_kernel<scalar_t, hidden_size, block_size>
        <<<static_cast<int>(rows), block_size, 0, stream>>>(
            x.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            epsilon
        );
}

template <typename scalar_t>
void dispatch_hidden_size(
    const torch::Tensor& x,
    const torch::Tensor& residual,
    const torch::Tensor& weight,
    torch::Tensor& output,
    torch::Tensor& residual_output,
    float epsilon,
    cudaStream_t stream
) {
    int64_t hidden_size = x.size(-1);
    if (hidden_size == 128) {
        launch_fused_add_rmsnorm<scalar_t, 128, 128>(
            x, residual, weight, output, residual_output, epsilon, stream
        );
    } else if (hidden_size == 1024) {
        launch_fused_add_rmsnorm<scalar_t, 1024, 256>(
            x, residual, weight, output, residual_output, epsilon, stream
        );
    } else {
        TORCH_CHECK(
            false,
            "fused_add_rmsnorm currently supports hidden_size 128 and 1024; got ",
            hidden_size
        );
    }
}

template <typename scalar_t>
void dispatch_rmsnorm_hidden_size(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    torch::Tensor& output,
    float epsilon,
    cudaStream_t stream
) {
    int64_t hidden_size = x.size(-1);
    if (hidden_size == 128) {
        launch_rmsnorm<scalar_t, 128, 128>(
            x, weight, output, epsilon, stream
        );
    } else if (hidden_size == 1024) {
        launch_rmsnorm<scalar_t, 1024, 256>(
            x, weight, output, epsilon, stream
        );
    } else {
        TORCH_CHECK(
            false,
            "rmsnorm currently supports hidden_size 128 and 1024; got ",
            hidden_size
        );
    }
}

void check_rmsnorm_inputs(
    const torch::Tensor& x,
    const torch::Tensor& weight
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(x.device() == weight.device(),
                "x and weight must be on the same CUDA device");
    TORCH_CHECK(x.scalar_type() == weight.scalar_type(),
                "x and weight must have the same dtype");
    TORCH_CHECK(x.dim() >= 1, "x must have at least one dimension");
    TORCH_CHECK(weight.dim() == 1, "weight must be one-dimensional");
    TORCH_CHECK(weight.size(0) == x.size(-1),
                "weight size must match x.size(-1)");
    TORCH_CHECK(x.numel() > 0 && x.size(-1) > 0,
                "empty tensors are not supported");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(
        x.scalar_type() == at::ScalarType::Half
            || x.scalar_type() == at::ScalarType::BFloat16,
        "rmsnorm supports only float16 and bfloat16"
    );
}

void check_inputs(
    const torch::Tensor& x,
    const torch::Tensor& residual,
    const torch::Tensor& weight
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(residual.is_cuda(), "residual must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(x.device() == residual.device()
                    && x.device() == weight.device(),
                "x, residual, and weight must be on the same CUDA device");
    TORCH_CHECK(x.scalar_type() == residual.scalar_type()
                    && x.scalar_type() == weight.scalar_type(),
                "x, residual, and weight must have the same dtype");
    TORCH_CHECK(x.sizes() == residual.sizes(),
                "x and residual must have the same shape");
    TORCH_CHECK(x.dim() >= 1, "x must have at least one dimension");
    TORCH_CHECK(weight.dim() == 1, "weight must be one-dimensional");
    TORCH_CHECK(weight.size(0) == x.size(-1),
                "weight size must match x.size(-1)");
    TORCH_CHECK(x.numel() > 0 && x.size(-1) > 0,
                "empty tensors are not supported");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(residual.is_contiguous(), "residual must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(
        x.scalar_type() == at::ScalarType::Half
            || x.scalar_type() == at::ScalarType::BFloat16,
        "fused_add_rmsnorm supports only float16 and bfloat16"
    );
}

}  // namespace

torch::Tensor rmsnorm_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    double epsilon
) {
    check_rmsnorm_inputs(x, weight);
    TORCH_CHECK(epsilon >= 0.0, "epsilon must be non-negative");

    c10::cuda::CUDAGuard device_guard(x.device());
    torch::Tensor output = torch::empty_like(x);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device());

    if (x.scalar_type() == at::ScalarType::Half) {
        dispatch_rmsnorm_hidden_size<at::Half>(
            x, weight, output, static_cast<float>(epsilon), stream
        );
    } else {
        dispatch_rmsnorm_hidden_size<at::BFloat16>(
            x, weight, output, static_cast<float>(epsilon), stream
        );
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

std::tuple<torch::Tensor, torch::Tensor> fused_add_rmsnorm_cuda(
    torch::Tensor x,
    torch::Tensor residual,
    torch::Tensor weight,
    double epsilon
) {
    check_inputs(x, residual, weight);
    TORCH_CHECK(epsilon >= 0.0, "epsilon must be non-negative");

    c10::cuda::CUDAGuard device_guard(x.device());
    torch::Tensor output = torch::empty_like(x);
    torch::Tensor residual_output = torch::empty_like(residual);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device());

    if (x.scalar_type() == at::ScalarType::Half) {
        dispatch_hidden_size<at::Half>(
            x, residual, weight, output, residual_output,
            static_cast<float>(epsilon), stream
        );
    } else {
        dispatch_hidden_size<at::BFloat16>(
            x, residual, weight, output, residual_output,
            static_cast<float>(epsilon), stream
        );
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {output, residual_output};
}
