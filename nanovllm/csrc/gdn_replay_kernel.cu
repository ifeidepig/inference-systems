#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

namespace {

constexpr int kThreads = 128;
constexpr int kCopyThreads = 256;

__global__ void gdn_replay_commit_kernel(
    float* __restrict__ recurrent_states,
    const int64_t* __restrict__ slot_ids,
    const float* __restrict__ replay_keys,
    const float* __restrict__ replay_deltas,
    const float* __restrict__ replay_log_decays,
    const int64_t* __restrict__ commit_lengths,
    int num_layers,
    int num_slots,
    int batch_size,
    int num_steps,
    int num_heads,
    int key_dim,
    int value_dim
) {
    int layer_batch_head = blockIdx.x;
    int layer_index = layer_batch_head / (batch_size * num_heads);
    int batch_head = layer_batch_head % (batch_size * num_heads);
    int batch_index = batch_head / num_heads;
    int head_index = batch_head % num_heads;
    int slot = static_cast<int>(slot_ids[batch_index]);
    int commit_length = static_cast<int>(commit_lengths[batch_index]);
    if (slot < 0 || slot >= num_slots || commit_length <= 0) {
        return;
    }
    commit_length = min(commit_length, num_steps);

    for (int value_index = threadIdx.x; value_index < value_dim;
         value_index += blockDim.x) {
        for (int key_index = 0; key_index < key_dim; ++key_index) {
            int64_t state_index =
                (((((static_cast<int64_t>(layer_index) * num_slots + slot)
                    * num_heads + head_index) * key_dim + key_index)
                  * value_dim) + value_index);
            float state = recurrent_states[state_index];
            for (int step = 0; step < commit_length; ++step) {
                int token_row = batch_index * num_steps + step;
                int64_t gate_index =
                    ((static_cast<int64_t>(layer_index)
                      * batch_size * num_steps + token_row)
                     * num_heads + head_index);
                int64_t key_record_index =
                    gate_index * key_dim + key_index;
                int64_t delta_record_index =
                    gate_index * value_dim + value_index;
                float retain = expf(replay_log_decays[gate_index]);
                state = fmaf(
                    replay_keys[key_record_index],
                    replay_deltas[delta_record_index],
                    retain * state
                );
            }
            recurrent_states[state_index] = state;
        }
    }
}

template <typename scalar_t>
__global__ void gdn_conv_commit_kernel(
    scalar_t* __restrict__ conv_states,
    const scalar_t* __restrict__ conv_history,
    const int64_t* __restrict__ slot_ids,
    const int64_t* __restrict__ state_boundaries,
    int num_slots,
    int batch_size,
    int num_steps,
    int state_elements
) {
    int layer_batch = blockIdx.x;
    int layer_index = layer_batch / batch_size;
    int batch_index = layer_batch % batch_size;
    int slot = static_cast<int>(slot_ids[batch_index]);
    int boundary = static_cast<int>(state_boundaries[batch_index]);
    if (slot < 0 || slot >= num_slots || boundary < 0 || boundary >= num_steps) {
        return;
    }
    int token_row = batch_index * num_steps + boundary;
    int64_t dst_base =
        (static_cast<int64_t>(layer_index) * num_slots + slot) * state_elements;
    int64_t src_base =
        (static_cast<int64_t>(layer_index) * batch_size * num_steps + token_row)
        * state_elements;
    for (int index = threadIdx.x; index < state_elements; index += blockDim.x) {
        conv_states[dst_base + index] = conv_history[src_base + index];
    }
}

void check_replay_inputs(
    const torch::Tensor& recurrent_states,
    const torch::Tensor& slot_ids,
    const torch::Tensor& replay_keys,
    const torch::Tensor& replay_deltas,
    const torch::Tensor& replay_log_decays,
    const torch::Tensor& commit_lengths,
    int64_t num_steps
) {
    const torch::Tensor tensors[] = {
        recurrent_states,
        slot_ids,
        replay_keys,
        replay_deltas,
        replay_log_decays,
        commit_lengths,
    };
    for (const auto& tensor : tensors) {
        TORCH_CHECK(tensor.is_cuda(), "all GDN replay inputs must be CUDA tensors");
        TORCH_CHECK(
            tensor.device() == recurrent_states.device(),
            "all GDN replay inputs must be on the same CUDA device"
        );
        TORCH_CHECK(tensor.is_contiguous(), "all GDN replay inputs must be contiguous");
    }
    TORCH_CHECK(recurrent_states.scalar_type() == torch::kFloat32,
                "recurrent state must use float32");
    TORCH_CHECK(replay_keys.scalar_type() == torch::kFloat32,
                "replay keys must use float32");
    TORCH_CHECK(replay_deltas.scalar_type() == torch::kFloat32,
                "replay deltas must use float32");
    TORCH_CHECK(replay_log_decays.scalar_type() == torch::kFloat32,
                "replay log decays must use float32");
    TORCH_CHECK(slot_ids.scalar_type() == torch::kInt64,
                "slot ids must use int64");
    TORCH_CHECK(commit_lengths.scalar_type() == torch::kInt64,
                "commit lengths must use int64");
    TORCH_CHECK(recurrent_states.dim() == 5,
                "recurrent state must be [layers, slots, heads, key, value]");
    TORCH_CHECK(replay_keys.dim() == 4,
                "replay keys must be [layers, batch*steps, heads, key]");
    TORCH_CHECK(replay_deltas.dim() == 4,
                "replay deltas must be [layers, batch*steps, heads, value]");
    TORCH_CHECK(replay_log_decays.dim() == 3,
                "replay decays must be [layers, batch*steps, heads]");
    TORCH_CHECK(slot_ids.dim() == 1 && commit_lengths.dim() == 1,
                "slot ids and commit lengths must be vectors");
    TORCH_CHECK(num_steps > 0, "num_steps must be positive");
    int64_t batch_size = slot_ids.numel();
    int64_t rows = batch_size * num_steps;
    TORCH_CHECK(commit_lengths.numel() == batch_size,
                "commit lengths must match batch size");
    TORCH_CHECK(replay_keys.size(0) == recurrent_states.size(0),
                "replay key layers must match recurrent state");
    TORCH_CHECK(replay_deltas.size(0) == recurrent_states.size(0),
                "replay delta layers must match recurrent state");
    TORCH_CHECK(replay_log_decays.size(0) == recurrent_states.size(0),
                "replay decay layers must match recurrent state");
    TORCH_CHECK(replay_keys.size(1) == rows,
                "replay key rows must match batch*steps");
    TORCH_CHECK(replay_deltas.size(1) == rows,
                "replay delta rows must match batch*steps");
    TORCH_CHECK(replay_log_decays.size(1) == rows,
                "replay decay rows must match batch*steps");
    TORCH_CHECK(replay_keys.size(2) == recurrent_states.size(2),
                "replay key heads must match recurrent state");
    TORCH_CHECK(replay_deltas.size(2) == recurrent_states.size(2),
                "replay delta heads must match recurrent state");
    TORCH_CHECK(replay_log_decays.size(2) == recurrent_states.size(2),
                "replay decay heads must match recurrent state");
    TORCH_CHECK(replay_keys.size(3) == recurrent_states.size(3),
                "replay key dim must match recurrent state");
    TORCH_CHECK(replay_deltas.size(3) == recurrent_states.size(4),
                "replay value dim must match recurrent state");
}

}  // namespace

torch::Tensor gdn_replay_commit_cuda(
    torch::Tensor recurrent_states,
    torch::Tensor slot_ids,
    torch::Tensor replay_keys,
    torch::Tensor replay_deltas,
    torch::Tensor replay_log_decays,
    torch::Tensor commit_lengths,
    int64_t num_steps
) {
    check_replay_inputs(
        recurrent_states,
        slot_ids,
        replay_keys,
        replay_deltas,
        replay_log_decays,
        commit_lengths,
        num_steps
    );
    c10::cuda::CUDAGuard device_guard(recurrent_states.device());
    int batch_size = static_cast<int>(slot_ids.numel());
    int num_layers = static_cast<int>(recurrent_states.size(0));
    int num_heads = static_cast<int>(recurrent_states.size(2));
    int blocks = num_layers * batch_size * num_heads;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    gdn_replay_commit_kernel<<<blocks, kThreads, 0, stream>>>(
        recurrent_states.data_ptr<float>(),
        slot_ids.data_ptr<int64_t>(),
        replay_keys.data_ptr<float>(),
        replay_deltas.data_ptr<float>(),
        replay_log_decays.data_ptr<float>(),
        commit_lengths.data_ptr<int64_t>(),
        num_layers,
        static_cast<int>(recurrent_states.size(1)),
        batch_size,
        static_cast<int>(num_steps),
        num_heads,
        static_cast<int>(recurrent_states.size(3)),
        static_cast<int>(recurrent_states.size(4))
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return recurrent_states;
}

torch::Tensor gdn_conv_commit_cuda(
    torch::Tensor conv_states,
    torch::Tensor conv_history,
    torch::Tensor slot_ids,
    torch::Tensor state_boundaries,
    int64_t num_steps
) {
    for (const auto& tensor : {
             conv_states, conv_history, slot_ids, state_boundaries}) {
        TORCH_CHECK(tensor.is_cuda(), "all GDN conv commit inputs must be CUDA tensors");
        TORCH_CHECK(tensor.device() == conv_states.device(),
                    "all GDN conv commit inputs must share a device");
        TORCH_CHECK(tensor.is_contiguous(),
                    "all GDN conv commit inputs must be contiguous");
    }
    TORCH_CHECK(conv_states.dim() == 4,
                "conv states must be [layers, slots, channels, window]");
    TORCH_CHECK(conv_history.dim() == 4,
                "conv history must be [layers, batch*steps, channels, window]");
    TORCH_CHECK(conv_history.scalar_type() == conv_states.scalar_type(),
                "conv history dtype must match conv states");
    TORCH_CHECK(conv_states.scalar_type() == torch::kFloat16
                    || conv_states.scalar_type() == torch::kBFloat16,
                "conv states must use float16 or bfloat16");
    TORCH_CHECK(slot_ids.scalar_type() == torch::kInt64
                    && state_boundaries.scalar_type() == torch::kInt64,
                "slot ids and boundaries must use int64");
    TORCH_CHECK(slot_ids.dim() == 1 && state_boundaries.dim() == 1,
                "slot ids and boundaries must be vectors");
    TORCH_CHECK(num_steps > 0, "num_steps must be positive");
    int batch_size = static_cast<int>(slot_ids.numel());
    TORCH_CHECK(state_boundaries.numel() == batch_size,
                "state boundaries must match batch size");
    TORCH_CHECK(conv_history.size(0) == conv_states.size(0),
                "conv history layers must match state pool");
    TORCH_CHECK(conv_history.size(1) == batch_size * num_steps,
                "conv history rows must match batch*steps");
    TORCH_CHECK(conv_history.size(2) == conv_states.size(2)
                    && conv_history.size(3) == conv_states.size(3),
                "conv history shape must match state pool");

    c10::cuda::CUDAGuard device_guard(conv_states.device());
    int blocks = static_cast<int>(conv_states.size(0)) * batch_size;
    int state_elements = static_cast<int>(
        conv_states.size(2) * conv_states.size(3));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        torch::kFloat16,
        torch::kBFloat16,
        conv_states.scalar_type(),
        "gdn_conv_commit_cuda",
        [&] {
            gdn_conv_commit_kernel<scalar_t><<<blocks, kCopyThreads, 0, stream>>>(
                conv_states.data_ptr<scalar_t>(),
                conv_history.data_ptr<scalar_t>(),
                slot_ids.data_ptr<int64_t>(),
                state_boundaries.data_ptr<int64_t>(),
                static_cast<int>(conv_states.size(1)),
                batch_size,
                static_cast<int>(num_steps),
                state_elements
            );
        }
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return conv_states;
}
