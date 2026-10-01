#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <mutex>
#include <tuple>

namespace {

constexpr int kHeadDim = 128;
constexpr int kStateElements = kHeadDim * kHeadDim;
constexpr int kThreads = 128;

__forceinline__ __device__ float warp_reduce_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__forceinline__ __device__ float silu(float value) {
    return value / (1.0f + expf(-value));
}

__forceinline__ __device__ float softplus(float value) {
    return value > 20.0f ? value : log1pf(expf(value));
}

template <typename scalar_t>
__forceinline__ __device__ float convolve_channel(
    const scalar_t* __restrict__ projected_qkv,
    const scalar_t* __restrict__ conv_states,
    const scalar_t* __restrict__ conv_weight,
    scalar_t* __restrict__ new_conv_states,
    int64_t projected_index,
    int64_t conv_state_index,
    int64_t weight_index
) {
    float state0 = static_cast<float>(conv_states[conv_state_index]);
    float state1 = static_cast<float>(conv_states[conv_state_index + 1]);
    float state2 = static_cast<float>(conv_states[conv_state_index + 2]);
    float projected = static_cast<float>(projected_qkv[projected_index]);
    float mixed = state0 * static_cast<float>(conv_weight[weight_index])
        + state1 * static_cast<float>(conv_weight[weight_index + 1])
        + state2 * static_cast<float>(conv_weight[weight_index + 2])
        + projected * static_cast<float>(conv_weight[weight_index + 3]);

    new_conv_states[conv_state_index] = static_cast<scalar_t>(state1);
    new_conv_states[conv_state_index + 1] = static_cast<scalar_t>(state2);
    new_conv_states[conv_state_index + 2] = static_cast<scalar_t>(projected);
    // Match the reference rounding point: convolution FP32 -> input dtype -> SiLU.
    float rounded_mixed = static_cast<float>(static_cast<scalar_t>(mixed));
    return static_cast<float>(static_cast<scalar_t>(silu(rounded_mixed)));
}

template <typename scalar_t>
__global__ void gdn_decode_kernel(
    const scalar_t* __restrict__ projected_qkv,
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    const scalar_t* __restrict__ conv_weight,
    const float* __restrict__ A_log,
    const scalar_t* __restrict__ dt_bias,
    const float* __restrict__ recurrent_states,
    const scalar_t* __restrict__ conv_states,
    scalar_t* __restrict__ core_output,
    float* __restrict__ new_recurrent_states,
    scalar_t* __restrict__ new_conv_states,
    int num_heads,
    int conv_dim
) {
    int batch_head = blockIdx.x;
    int batch_index = batch_head / num_heads;
    int head_index = batch_head % num_heads;
    int dimension = threadIdx.x;

    extern __shared__ float shared[];
    float* state = shared;
    float* query = state + kStateElements;
    float* key = query + kHeadDim;
    float* warp_q_sums = key + kHeadDim;
    float* warp_k_sums = warp_q_sums + 4;
    float* scalars = warp_k_sums + 4;

    int64_t state_base = static_cast<int64_t>(batch_head) * kStateElements;
    for (int index = dimension; index < kStateElements; index += kThreads) {
        state[index] = recurrent_states[state_base + index];
    }

    int64_t projected_batch_base = static_cast<int64_t>(batch_index) * conv_dim;
    int q_channel = head_index * kHeadDim + dimension;
    int k_channel = num_heads * kHeadDim + q_channel;
    int v_channel = 2 * num_heads * kHeadDim + q_channel;

    auto mix_channel = [&](int channel) {
        int64_t projected_index = projected_batch_base + channel;
        int64_t conv_state_index =
            (projected_batch_base + channel) * 3;
        int64_t weight_index = static_cast<int64_t>(channel) * 4;
        return convolve_channel(
            projected_qkv,
            conv_states,
            conv_weight,
            new_conv_states,
            projected_index,
            conv_state_index,
            weight_index
        );
    };

    float q_value = mix_channel(q_channel);
    float k_value = mix_channel(k_channel);
    float v_value = mix_channel(v_channel);

    float q_sum = warp_reduce_sum(q_value * q_value);
    float k_sum = warp_reduce_sum(k_value * k_value);
    int lane = dimension & 31;
    int warp_id = dimension >> 5;
    if (lane == 0) {
        warp_q_sums[warp_id] = q_sum;
        warp_k_sums[warp_id] = k_sum;
    }
    __syncthreads();

    if (dimension == 0) {
        float total_q = 0.0f;
        float total_k = 0.0f;
        #pragma unroll
        for (int warp = 0; warp < 4; ++warp) {
            total_q += warp_q_sums[warp];
            total_k += warp_k_sums[warp];
        }
        scalars[0] = rsqrtf(total_q + 1e-6f) * rsqrtf(128.0f);
        scalars[1] = rsqrtf(total_k + 1e-6f);
        int64_t head_parameter_index =
            static_cast<int64_t>(batch_index) * num_heads + head_index;
        float a_value = static_cast<float>(a[head_parameter_index]);
        float b_value = static_cast<float>(b[head_parameter_index]);
        float dt_value = static_cast<float>(dt_bias[head_index]);
        scalars[2] = 1.0f / (1.0f + expf(-b_value));
        scalars[3] = expf(
            -expf(A_log[head_index]) * softplus(a_value + dt_value)
        );
    }
    __syncthreads();

    query[dimension] = q_value * scalars[0];
    key[dimension] = k_value * scalars[1];
    __syncthreads();

    float decay = scalars[3];
    #pragma unroll 4
    for (int key_dimension = 0; key_dimension < kHeadDim; ++key_dimension) {
        int state_index = key_dimension * kHeadDim + dimension;
        state[state_index] *= decay;
    }

    float predicted_value = 0.0f;
    #pragma unroll 4
    for (int key_dimension = 0; key_dimension < kHeadDim; ++key_dimension) {
        float decayed_state = state[key_dimension * kHeadDim + dimension];
        predicted_value = fmaf(
            key[key_dimension],
            decayed_state,
            predicted_value
        );
    }

    float delta = (v_value - predicted_value) * scalars[2];
    float output = 0.0f;
    #pragma unroll 4
    for (int key_dimension = 0; key_dimension < kHeadDim; ++key_dimension) {
        int state_index = key_dimension * kHeadDim + dimension;
        float updated_state = fmaf(
            key[key_dimension],
            delta,
            state[state_index]
        );
        state[state_index] = updated_state;
        output = fmaf(query[key_dimension], updated_state, output);
    }
    core_output[batch_head * kHeadDim + dimension] =
        static_cast<scalar_t>(output);
    __syncthreads();

    for (int index = dimension; index < kStateElements; index += kThreads) {
        new_recurrent_states[state_base + index] = state[index];
    }
}

void check_inputs(
    const torch::Tensor& projected_qkv,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& conv_weight,
    const torch::Tensor& A_log,
    const torch::Tensor& dt_bias,
    const torch::Tensor& recurrent_states,
    const torch::Tensor& conv_states,
    int64_t num_key_heads,
    int64_t num_value_heads,
    int64_t key_head_dim,
    int64_t value_head_dim
) {
    const torch::Tensor tensors[] = {
        projected_qkv,
        a,
        b,
        conv_weight,
        A_log,
        dt_bias,
        recurrent_states,
        conv_states,
    };
    for (const auto& tensor : tensors) {
        TORCH_CHECK(tensor.is_cuda(), "all GDN decode inputs must be CUDA tensors");
        TORCH_CHECK(
            tensor.device() == projected_qkv.device(),
            "all GDN decode inputs must be on the same CUDA device"
        );
        TORCH_CHECK(tensor.is_contiguous(), "all GDN decode inputs must be contiguous");
    }
    TORCH_CHECK(
        projected_qkv.scalar_type() == torch::kFloat16
            || projected_qkv.scalar_type() == torch::kBFloat16,
        "projected_qkv must use float16 or bfloat16"
    );
    for (const auto& tensor : {a, b, conv_weight, dt_bias, conv_states}) {
        TORCH_CHECK(
            tensor.scalar_type() == projected_qkv.scalar_type(),
            "projection, convolution, dt_bias, and conv-state dtypes must match"
        );
    }
    TORCH_CHECK(A_log.scalar_type() == torch::kFloat32, "A_log must use float32");
    TORCH_CHECK(
        recurrent_states.scalar_type() == torch::kFloat32,
        "recurrent state must use float32"
    );
    TORCH_CHECK(num_key_heads == num_value_heads, "GDN CUDA v0 requires equal heads");
    TORCH_CHECK(
        key_head_dim == kHeadDim && value_head_dim == kHeadDim,
        "GDN CUDA v0 requires 128-dimensional key/value heads"
    );
    TORCH_CHECK(projected_qkv.dim() == 2, "projected_qkv must be 2-D");
    int64_t batch_size = projected_qkv.size(0);
    int64_t conv_dim = 3 * num_key_heads * kHeadDim;
    TORCH_CHECK(projected_qkv.size(1) == conv_dim, "invalid projected_qkv width");
    TORCH_CHECK(
        a.sizes() == torch::IntArrayRef({batch_size, num_value_heads}),
        "invalid a shape"
    );
    TORCH_CHECK(
        b.sizes() == torch::IntArrayRef({batch_size, num_value_heads}),
        "invalid b shape"
    );
    TORCH_CHECK(
        conv_weight.sizes() == torch::IntArrayRef({conv_dim, 4}),
        "GDN CUDA v0 requires conv_weight [conv_dim, 4]"
    );
    TORCH_CHECK(A_log.numel() == num_value_heads, "invalid A_log shape");
    TORCH_CHECK(dt_bias.numel() == num_value_heads, "invalid dt_bias shape");
    TORCH_CHECK(
        recurrent_states.sizes() == torch::IntArrayRef(
            {batch_size, num_value_heads, kHeadDim, kHeadDim}
        ),
        "invalid recurrent state shape"
    );
    TORCH_CHECK(
        conv_states.sizes() == torch::IntArrayRef({batch_size, conv_dim, 3}),
        "invalid conv state shape"
    );
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> gdn_decode_cuda(
    torch::Tensor projected_qkv,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor conv_weight,
    torch::Tensor A_log,
    torch::Tensor dt_bias,
    torch::Tensor recurrent_states,
    torch::Tensor conv_states,
    int64_t num_key_heads,
    int64_t num_value_heads,
    int64_t key_head_dim,
    int64_t value_head_dim
) {
    check_inputs(
        projected_qkv,
        a,
        b,
        conv_weight,
        A_log,
        dt_bias,
        recurrent_states,
        conv_states,
        num_key_heads,
        num_value_heads,
        key_head_dim,
        value_head_dim
    );
    c10::cuda::CUDAGuard device_guard(projected_qkv.device());
    auto core_output = torch::empty(
        {projected_qkv.size(0), num_value_heads, value_head_dim},
        projected_qkv.options()
    );
    auto new_recurrent_states = torch::empty_like(recurrent_states);
    auto new_conv_states = torch::empty_like(conv_states);
    int blocks = static_cast<int>(projected_qkv.size(0) * num_value_heads);
    constexpr int shared_floats = kStateElements + 2 * kHeadDim + 12;
    constexpr int shared_bytes = shared_floats * sizeof(float);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        torch::kFloat16,
        torch::kBFloat16,
        projected_qkv.scalar_type(),
        "gdn_decode_cuda",
        [&] {
            auto kernel = gdn_decode_kernel<scalar_t>;
            // nano-vLLM uses one CUDA device per ModelRunner process. The
            // function attribute is invariant for that process/dtype, so do
            // not repeat this host-side configuration on every eager token.
            static std::once_flag shared_memory_attribute_once;
            std::call_once(shared_memory_attribute_once, [&] {
                C10_CUDA_CHECK(cudaFuncSetAttribute(
                    kernel,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    shared_bytes
                ));
            });
            kernel<<<blocks, kThreads, shared_bytes, stream>>>(
                projected_qkv.data_ptr<scalar_t>(),
                a.data_ptr<scalar_t>(),
                b.data_ptr<scalar_t>(),
                conv_weight.data_ptr<scalar_t>(),
                A_log.data_ptr<float>(),
                dt_bias.data_ptr<scalar_t>(),
                recurrent_states.data_ptr<float>(),
                conv_states.data_ptr<scalar_t>(),
                core_output.data_ptr<scalar_t>(),
                new_recurrent_states.data_ptr<float>(),
                new_conv_states.data_ptr<scalar_t>(),
                static_cast<int>(num_value_heads),
                static_cast<int>(projected_qkv.size(1))
            );
        }
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {core_output, new_recurrent_states, new_conv_states};
}
