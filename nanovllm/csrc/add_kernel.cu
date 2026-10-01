#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

namespace {

template <typename scalar_t>
__global__ void add_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ residual,
    scalar_t* __restrict__ output,
    int64_t numel
) {
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    for (; index < numel; index += stride) {
        float value = static_cast<float>(x[index])
            + static_cast<float>(residual[index]);
        output[index] = static_cast<scalar_t>(value);
    }
}

template <typename scalar_t>
void launch_add(
    const torch::Tensor& x,
    const torch::Tensor& residual,
    torch::Tensor& output,
    cudaStream_t stream
) {
    constexpr int threads = 256;
    int64_t numel = x.numel();
    int blocks = static_cast<int>((numel + threads - 1) / threads);
    blocks = std::min(blocks, 4096);

    add_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        x.data_ptr<scalar_t>(),
        residual.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        numel
    );
}

}  // namespace

torch::Tensor add_cuda(torch::Tensor x, torch::Tensor residual) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(residual.is_cuda(), "residual must be a CUDA tensor");
    TORCH_CHECK(x.device() == residual.device(),
                "x and residual must be on the same CUDA device");
    TORCH_CHECK(x.scalar_type() == residual.scalar_type(),
                "x and residual must have the same dtype");
    TORCH_CHECK(x.sizes() == residual.sizes(),
                "x and residual must have the same shape");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(residual.is_contiguous(), "residual must be contiguous");
    TORCH_CHECK(
        x.scalar_type() == at::ScalarType::Half
            || x.scalar_type() == at::ScalarType::BFloat16,
        "custom add currently supports only float16 and bfloat16"
    );

    c10::cuda::CUDAGuard device_guard(x.device());
    torch::Tensor output = torch::empty_like(x);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device());

    if (x.scalar_type() == at::ScalarType::Half) {
        launch_add<at::Half>(x, residual, output, stream);
    } else {
        launch_add<at::BFloat16>(x, residual, output, stream);
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
