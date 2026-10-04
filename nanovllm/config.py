import os
from dataclasses import dataclass, field
from transformers import AutoConfig

from nanovllm.models.registry import ModelCapabilities, normalize_hf_config


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    scheduling_policy: str = "prefill_first"
    waiting_admission_policy: str = "fcfs"
    preemption_policy: str = "lifo"
    hybrid_scheduler_candidate_window: int = 8
    hybrid_scheduler_aging_tokens_per_ms: float = 0.5
    hybrid_scheduler_max_wait_ms: float = 200.0
    hybrid_scheduler_min_saved_tokens: int = 16
    hybrid_scheduler_preemption_penalty: float = 128.0
    hybrid_scheduler_score_source: str = "joint"
    hybrid_scheduler_enable_aging: bool = True
    hybrid_scheduler_enable_hysteresis: bool = True
    hybrid_scheduler_enable_sticky_recovery: bool = True
    enable_scheduler_profiling: bool = False
    scheduler_decision_history_size: int = 4096
    scheduler_target_ttft_ms: float = 200.0
    scheduler_target_tpot_ms: float = 50.0
    slo_prefill_priority_threshold: float = 0.8
    slo_min_prefill_tokens: int = 64
    slo_kv_pressure_threshold: float = 0.9
    slo_queue_pressure_threshold: int = 3
    slo_latency_safety_margin_ms: float = 5.0
    scheduler_ewma_alpha: float = 0.2
    enable_prefix_cache: bool = True
    enable_chunked_prefill: bool = True
    request_metrics_history_size: int = 1024
    full_hf_config: AutoConfig | None = None
    hf_config: AutoConfig | None = None
    model_capabilities: ModelCapabilities | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    prefix_match_unit: int | None = None
    num_kvcache_blocks: int = -1
    max_num_kvcache_blocks: int | None = None
    max_num_state_slots: int = 1
    num_speculative_tokens: int = 0
    speculative_parallel_verify: bool = True
    enable_mtp_phase_profiling: bool = False
    gdn_decode_backend: str = "torch"
    enable_hybrid_prefix_cache: bool = False
    hybrid_prefix_checkpoint_interval_blocks: int = 8
    hybrid_prefix_checkpoint_interval_tokens: int | None = None
    hybrid_prefix_checkpoint_memory_bytes: int = 0
    hybrid_prefix_checkpoint_dtype: str = "fp32"
    hybrid_prefix_promotion_min_sightings: int = 2
    hybrid_prefix_retention_policy: str = "periodic"
    hybrid_prefix_eviction_policy: str = "lru"
    enable_hybrid_internal_checkpoints: bool = False
    hybrid_prefix_checkpoint_bytes_per_slot: int = field(default=0, init=False)

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        if self.prefix_match_unit is None:
            self.prefix_match_unit = self.kvcache_block_size
        if self.prefix_match_unit <= 0:
            raise ValueError("prefix_match_unit must be positive")
        if self.kvcache_block_size % self.prefix_match_unit:
            raise ValueError(
                "kvcache_block_size must be divisible by prefix_match_unit"
            )
        assert 1 <= self.tensor_parallel_size <= 8
        assert self.scheduling_policy in (
            "prefill_first",
            "decode_first",
            "slo_aware",
        )
        if self.waiting_admission_policy not in (
            "fcfs",
            "hybrid_state_aware",
        ):
            raise ValueError("unsupported waiting admission policy")
        if self.preemption_policy not in ("lifo", "recompute_aware"):
            raise ValueError("unsupported preemption policy")
        if self.hybrid_scheduler_candidate_window <= 0:
            raise ValueError("candidate window must be positive")
        if self.hybrid_scheduler_aging_tokens_per_ms < 0:
            raise ValueError("aging factor must be non-negative")
        if self.hybrid_scheduler_max_wait_ms <= 0:
            raise ValueError("maximum wait must be positive")
        if self.hybrid_scheduler_min_saved_tokens < 0:
            raise ValueError("minimum saved tokens must be non-negative")
        if self.hybrid_scheduler_preemption_penalty < 0:
            raise ValueError("preemption penalty must be non-negative")
        if self.hybrid_scheduler_score_source not in ("joint", "kv_only"):
            raise ValueError("scheduler score source must be joint or kv_only")
        if self.scheduler_decision_history_size < 0:
            raise ValueError("scheduler decision history size cannot be negative")
        assert self.scheduler_target_ttft_ms > 0
        assert self.scheduler_target_tpot_ms > 0
        assert self.slo_prefill_priority_threshold > 0
        assert self.slo_min_prefill_tokens > 0
        assert 0 < self.slo_kv_pressure_threshold <= 1
        assert self.slo_queue_pressure_threshold > 0
        assert self.slo_latency_safety_margin_ms >= 0
        assert 0 < self.scheduler_ewma_alpha <= 1
        assert self.request_metrics_history_size >= 0
        assert (
            self.max_num_kvcache_blocks is None
            or self.max_num_kvcache_blocks > 0
        )
        assert self.max_num_state_slots > 0
        assert 0 <= self.num_speculative_tokens <= 8
        if self.gdn_decode_backend not in ("torch", "cuda", "auto"):
            raise ValueError("gdn_decode_backend must be torch, cuda, or auto")
        assert self.hybrid_prefix_checkpoint_interval_blocks > 0
        if self.hybrid_prefix_checkpoint_interval_tokens is not None:
            if self.hybrid_prefix_checkpoint_interval_tokens <= 0:
                raise ValueError("checkpoint token interval must be positive")
            if (
                self.hybrid_prefix_checkpoint_interval_tokens
                % self.prefix_match_unit
            ):
                raise ValueError(
                    "checkpoint token interval must align to prefix_match_unit"
                )
        assert self.hybrid_prefix_checkpoint_memory_bytes >= 0
        if self.hybrid_prefix_checkpoint_dtype not in ("fp32", "bf16", "int8"):
            raise ValueError(
                "hybrid prefix checkpoint dtype must be fp32, bf16, or int8"
            )
        if self.hybrid_prefix_promotion_min_sightings < 2:
            raise ValueError(
                "hybrid prefix promotion requires at least two sightings"
            )
        if self.hybrid_prefix_retention_policy not in ("periodic", "adaptive"):
            raise ValueError("unsupported hybrid prefix retention policy")
        if self.hybrid_prefix_eviction_policy not in ("lru", "cost_aware"):
            raise ValueError("unsupported hybrid prefix eviction policy")
        self.full_hf_config = AutoConfig.from_pretrained(self.model)
        self.hf_config, self.model_capabilities = normalize_hf_config(
            self.full_hf_config
        )
        if self.num_speculative_tokens:
            if not self.model_capabilities.has_recurrent_state:
                raise ValueError("native MTP currently requires Qwen3.5 hybrid state")
            if self.model_capabilities.mtp_num_hidden_layers != 1:
                raise ValueError("native MTP requires exactly one checkpoint MTP layer")
        if self.enable_hybrid_prefix_cache:
            if self.enable_hybrid_internal_checkpoints and self.max_num_seqs != 1:
                raise ValueError(
                    "internal hybrid checkpoints currently require max_num_seqs=1"
                )
            if not self.enable_prefix_cache:
                raise ValueError("hybrid prefix cache requires prefix caching enabled")
            if not self.model_capabilities.has_recurrent_state:
                raise ValueError("hybrid prefix cache requires recurrent model state")
            if self.num_speculative_tokens:
                raise ValueError("hybrid prefix cache MVP does not yet support MTP")
            if self.hybrid_prefix_checkpoint_memory_bytes <= 0:
                raise ValueError("hybrid prefix cache requires a positive checkpoint byte budget")
            num_linear_layers = len(
                self.model_capabilities.linear_attention_layer_indices
            )
            num_key_heads = (
                self.hf_config.linear_num_key_heads
                // self.tensor_parallel_size
            )
            num_value_heads = (
                self.hf_config.linear_num_value_heads
                // self.tensor_parallel_size
            )
            key_dim = self.hf_config.linear_key_head_dim
            value_dim = self.hf_config.linear_value_head_dim
            conv_dim = 2 * num_key_heads * key_dim + num_value_heads * value_dim
            recurrent_elements = (
                num_linear_layers
                * num_value_heads
                * key_dim
                * value_dim
            )
            conv_bytes = (
                num_linear_layers
                * conv_dim
                * (self.hf_config.linear_conv_kernel_dim - 1)
                * self.hf_config.dtype.itemsize
            )
            if self.hybrid_prefix_checkpoint_dtype == "fp32":
                recurrent_bytes = recurrent_elements * 4
                scale_bytes = 0
            elif self.hybrid_prefix_checkpoint_dtype == "bf16":
                recurrent_bytes = recurrent_elements * 2
                scale_bytes = 0
            else:
                recurrent_bytes = recurrent_elements
                scale_bytes = (
                    num_linear_layers
                    * num_value_heads
                    * key_dim
                    * 4
                )
            self.hybrid_prefix_checkpoint_bytes_per_slot = (
                recurrent_bytes + scale_bytes + conv_bytes
            )
            if (
                self.hybrid_prefix_checkpoint_memory_bytes
                < self.hybrid_prefix_checkpoint_bytes_per_slot
            ):
                raise ValueError(
                    "hybrid prefix checkpoint budget is smaller than one "
                    f"checkpoint ({self.hybrid_prefix_checkpoint_bytes_per_slot} bytes)"
                )
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
