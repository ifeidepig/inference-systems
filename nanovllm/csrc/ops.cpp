#include <torch/extension.h>

#include <tuple>

torch::Tensor add_cuda(torch::Tensor x, torch::Tensor residual);
torch::Tensor rmsnorm_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    double epsilon
);
std::tuple<torch::Tensor, torch::Tensor> fused_add_rmsnorm_cuda(
    torch::Tensor x,
    torch::Tensor residual,
    torch::Tensor weight,
    double epsilon
);
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
);
torch::Tensor gdn_replay_commit_cuda(
    torch::Tensor recurrent_states,
    torch::Tensor slot_ids,
    torch::Tensor replay_keys,
    torch::Tensor replay_deltas,
    torch::Tensor replay_log_decays,
    torch::Tensor commit_lengths,
    int64_t num_steps
);
torch::Tensor gdn_conv_commit_cuda(
    torch::Tensor conv_states,
    torch::Tensor conv_history,
    torch::Tensor slot_ids,
    torch::Tensor state_boundaries,
    int64_t num_steps
);

TORCH_LIBRARY(nanovllm, m) {
    m.def("add(Tensor x, Tensor residual) -> Tensor");
    m.def("rmsnorm(Tensor x, Tensor weight, float epsilon) -> Tensor");
    m.def(
        "fused_add_rmsnorm(Tensor x, Tensor residual, Tensor weight, "
        "float epsilon) -> (Tensor, Tensor)"
    );
    m.def(
        "gdn_decode(Tensor projected_qkv, Tensor a, Tensor b, "
        "Tensor conv_weight, Tensor A_log, Tensor dt_bias, "
        "Tensor recurrent_states, Tensor conv_states, int num_key_heads, "
        "int num_value_heads, int key_head_dim, int value_head_dim) "
        "-> (Tensor, Tensor, Tensor)"
    );
    m.def(
        "gdn_replay_commit(Tensor(a!) recurrent_states, Tensor slot_ids, "
        "Tensor replay_keys, Tensor replay_deltas, Tensor replay_log_decays, "
        "Tensor commit_lengths, int num_steps) -> Tensor(a!)"
    );
    m.def(
        "gdn_conv_commit(Tensor(a!) conv_states, Tensor conv_history, "
        "Tensor slot_ids, Tensor state_boundaries, int num_steps) -> Tensor(a!)"
    );
}

TORCH_LIBRARY_IMPL(nanovllm, CUDA, m) {
    m.impl("add", &add_cuda);
    m.impl("rmsnorm", &rmsnorm_cuda);
    m.impl("fused_add_rmsnorm", &fused_add_rmsnorm_cuda);
    m.impl("gdn_decode", &gdn_decode_cuda);
    m.impl("gdn_replay_commit", &gdn_replay_commit_cuda);
    m.impl("gdn_conv_commit", &gdn_conv_commit_cuda);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
