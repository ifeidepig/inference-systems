import pickle
from contextlib import nullcontext
from time import perf_counter, perf_counter_ns
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.state_manager import (
    HybridPrefixCheckpointPool,
    HybridStateManager,
    transactional_capture_prefix_checkpoint,
    transactional_restore_prefix_checkpoint,
)
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.phase_profiler import PhaseProfiler
from nanovllm.layers.gated_delta_net import GatedDeltaNet
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.models.registry import create_model
from nanovllm.models.qwen3_5_mtp import Qwen3_5MTP
from nanovllm.layers.sampler import Sampler
from nanovllm.engine.speculative import (
    HybridSpeculativeTransaction,
    greedy_verify_tokens,
)
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:

    def __init__(
        self,
        config: Config,
        rank: int,
        event: Event | list[Event],
        ack_event: Event | list[Event] | None = None,
    ):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event
        self.ack_event = ack_event
        self.mtp_phase_profiler = PhaseProfiler(
            getattr(config, "enable_mtp_phase_profiling", False)
        )
        self.reset_metrics()

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        # The model receives the normalized HF config, so propagate the
        # engine-level backend choice before constructing GDN layers.
        hf_config.gdn_decode_backend = config.gdn_decode_backend
        hf_config.enable_mtp_phase_profiling = getattr(
            config, "enable_mtp_phase_profiling", False
        )
        # get_rope caches a shared module to avoid duplicating the large
        # cos/sin table across layers. Drop modules created by an earlier
        # CPU/model instance before constructing this CUDA model.
        get_rope.cache_clear()
        self.model = create_model(hf_config, config.model_capabilities)
        load_model(self.model, config.model, strict=True)
        self.num_speculative_tokens = getattr(
            config, "num_speculative_tokens", 0
        )
        self.speculative_parallel_verify = getattr(
            config, "speculative_parallel_verify", False
        )
        self.mtp = None
        if self.num_speculative_tokens:
            self.mtp = Qwen3_5MTP(
                hf_config,
                self.model.model.embed_tokens,
            )
            load_model(self.mtp, config.model, strict=True)
        self.state_manager = None
        if config.model_capabilities.has_recurrent_state:
            linear_layers = [
                module
                for module in self.model.modules()
                if isinstance(module, GatedDeltaNet)
            ]
            reference_layer = linear_layers[0]
            max_state_slots = min(config.max_num_state_slots, config.max_num_seqs)
            self.state_manager = HybridStateManager(
                max_num_seqs=max_state_slots,
                num_linear_layers=len(linear_layers),
                num_value_heads=reference_layer.num_value_heads,
                key_head_dim=reference_layer.key_head_dim,
                value_head_dim=reference_layer.value_head_dim,
                conv_dim=reference_layer.conv_dim,
                conv_kernel_size=reference_layer.conv_kernel_size,
                conv_dtype=hf_config.dtype,
                device="cuda",
            )
        self.mtp_prev_target_hidden = None
        if self.mtp is not None:
            self.mtp_prev_target_hidden = torch.zeros(
                self.state_manager.max_num_seqs + 1,
                hf_config.hidden_size,
                dtype=hf_config.dtype,
                device="cuda",
            )
        self.prefix_checkpoint_pool = None
        if getattr(config, "enable_hybrid_prefix_cache", False):
            self.prefix_checkpoint_pool = HybridPrefixCheckpointPool(
                self.state_manager,
                config.hybrid_prefix_checkpoint_memory_bytes,
                checkpoint_dtype=getattr(
                    config,
                    "hybrid_prefix_checkpoint_dtype",
                    "fp32",
                ),
            )
        self.enable_hybrid_internal_checkpoints = bool(
            getattr(config, "enable_hybrid_internal_checkpoints", False)
        )
        self.internal_prefix_states: dict[
            tuple[int, int], tuple[torch.Tensor, torch.Tensor]
        ] = {}
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
            if hasattr(self, "mtp_graphs"):
                del self.mtp_graphs, self.mtp_graph_vars
            if hasattr(self, "verify_graphs"):
                del self.verify_graphs, self.verify_graph_vars
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        if self.ack_event is not None:
            self.ack_event.set()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        if self.ack_event is not None:
            for ack_event in self.ack_event:
                ack_event.wait()
                ack_event.clear()
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        if self.state_manager is not None:
            # Keep the correctness-first reference warmup bounded.
            seq_len = min(seq_len, 8)
            # The FP32 reference GDN path has large temporary tensors. One
            # sequence is enough to warm kernels without poisoning KV budget
            # estimation with an artificial multi-sequence peak.
            num_seqs = 1
        else:
            num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
            if self.state_manager is not None:
                self.allocate_state_slot(seq)
        self.run(seqs, True)
        if self.state_manager is not None:
            for seq in seqs:
                self.free_state_slot(seq)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        attention_modules = [
            module
            for module in self.model.modules()
            if hasattr(module, "k_cache") and hasattr(module, "v_cache")
        ]
        if self.mtp is not None:
            attention_modules.extend(
                module
                for module in self.mtp.modules()
                if hasattr(module, "k_cache") and hasattr(module, "v_cache")
            )
        num_kv_layers = len(attention_modules)
        block_bytes = 2 * num_kv_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        if config.max_num_kvcache_blocks is not None:
            config.num_kvcache_blocks = min(
                config.num_kvcache_blocks,
                config.max_num_kvcache_blocks,
            )
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(2, num_kv_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        for layer_id, module in enumerate(attention_modules):
            module.k_cache = self.kv_cache[0, layer_id]
            module.v_cache = self.kv_cache[1, layer_id]

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        state_slot_ids = []
        for seq in seqs:
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if self.state_manager is not None:
                if seq.state_slot is None:
                    raise ValueError("hybrid sequence has no state slot")
                state_slot_ids.append(seq.state_slot)
            if not seq.block_table:    # warmup
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        state_slot_ids = (
            torch.tensor(state_slot_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
            if self.state_manager is not None
            else None
        )
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables, state_slot_ids)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        state_slot_ids = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
            if self.state_manager is not None:
                if seq.state_slot is None:
                    raise ValueError("hybrid sequence has no state slot")
                state_slot_ids.append(seq.state_slot)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        cu_seqlens_q = (
            torch.arange(len(seqs) + 1, dtype=torch.int32, device="cuda")
            if self.state_manager is not None
            else None
        )
        state_slot_ids = (
            torch.tensor(state_slot_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
            if self.state_manager is not None
            else None
        )
        set_context(False, cu_seqlens_q=cu_seqlens_q, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables, state_slot_ids=state_slot_ids)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    def _forward_target_eager(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        *,
        compute_logits: bool = True,
        all_logits: bool = False,
        return_state_history: bool = False,
        state_checkpoint_indices: tuple[int, ...] | None = None,
    ):
        """Run target eagerly and expose hidden states for the MTP drafter."""
        context = get_context()
        if self.state_manager is None:
            hidden_states = self.model(input_ids, positions)
            if not compute_logits:
                return hidden_states, None
            context = get_context()
            old_is_prefill = context.is_prefill
            if all_logits:
                context.is_prefill = False
            logits = self.model.compute_logits(hidden_states)
            context.is_prefill = old_is_prefill
            return hidden_states, logits

        num_layers = len(self.model.model.layers)
        linear_indices = self.model.model.linear_attention_layer_indices
        recurrent_states, conv_states = self.state_manager.gather(
            context.state_slot_ids,
            num_total_layers=num_layers,
            linear_layer_indices=linear_indices,
        )
        model_result = self.model(
            input_ids,
            positions,
            context.cu_seqlens_q,
            recurrent_states,
            conv_states,
            return_state_history=return_state_history,
            state_checkpoint_indices=state_checkpoint_indices,
        )
        capture_states = (
            return_state_history or state_checkpoint_indices is not None
        )
        if capture_states:
            (
                hidden_states,
                new_recurrent_states,
                new_conv_states,
                recurrent_histories,
                conv_histories,
            ) = model_result
        else:
            hidden_states, new_recurrent_states, new_conv_states = model_result
        self.state_manager.scatter(
            context.state_slot_ids,
            new_recurrent_states,
            new_conv_states,
            linear_layer_indices=linear_indices,
        )
        if not compute_logits:
            result = hidden_states, None
            if capture_states:
                return result + (recurrent_histories, conv_histories)
            return result
        old_is_prefill = context.is_prefill
        if all_logits:
            context.is_prefill = False
        logits = self.model.compute_logits(hidden_states)
        context.is_prefill = old_is_prefill
        result = hidden_states, logits
        if capture_states:
            return result + (recurrent_histories, conv_histories)
        return result

    def _seed_mtp_prefill(
        self,
        seqs: list[Sequence],
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
    ) -> None:
        """Build the shifted MTP KV history, including chunked prefill."""
        if self.mtp is None:
            return
        pair_input_indices = []
        pair_hidden_states = []
        pair_slot_mapping = []
        pair_counts = []
        mtp_context_lengths = []
        active_seqs = []
        cursor = 0
        for seq in seqs:
            length = seq.num_scheduled_tokens
            start = seq.num_cached_tokens
            count = 0
            for local_index in range(length):
                target_position = start + local_index
                if target_position == 0:
                    continue
                flat_index = cursor + local_index
                pair_input_indices.append(flat_index)
                if local_index:
                    pair_hidden_states.append(
                        target_hidden_states[flat_index - 1]
                    )
                else:
                    pair_hidden_states.append(
                        self.mtp_prev_target_hidden[seq.state_slot]
                    )
                mtp_position = target_position - 1
                if seq.block_table:
                    pair_slot_mapping.append(
                        seq.block_table[mtp_position // self.block_size]
                        * self.block_size
                        + mtp_position % self.block_size
                    )
                else:
                    pair_slot_mapping.append(-1)
                count += 1
            if length:
                self.mtp_prev_target_hidden[seq.state_slot].copy_(
                    target_hidden_states[cursor + length - 1]
                )
            if count:
                active_seqs.append(seq)
                pair_counts.append(count)
                mtp_context_lengths.append(start + length - 1)
            cursor += length
        if not pair_input_indices:
            return

        indices = torch.tensor(
            pair_input_indices,
            dtype=torch.long,
            device=input_ids.device,
        )
        cu_q = torch.tensor(
            [0] + list(torch.tensor(pair_counts).cumsum(0).tolist()),
            dtype=torch.int32,
            device=input_ids.device,
        )
        cu_k = torch.tensor(
            [0] + list(torch.tensor(mtp_context_lengths).cumsum(0).tolist()),
            dtype=torch.int32,
            device=input_ids.device,
        )
        block_tables = (
            self.prepare_block_tables(active_seqs)
            if all(seq.block_table for seq in active_seqs)
            else None
        )
        set_context(
            True,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=max(pair_counts),
            max_seqlen_k=max(mtp_context_lengths),
            slot_mapping=torch.tensor(
                pair_slot_mapping,
                dtype=torch.int32,
                device=input_ids.device,
            ),
            block_tables=block_tables,
        )
        self.mtp(
            input_ids.index_select(0, indices),
            positions.index_select(0, indices),
            torch.stack(pair_hidden_states),
            cu_q,
        )

    def _set_offset_decode_context(
        self,
        seqs: list[Sequence],
        target_positions: torch.Tensor,
        *,
        mtp_cache: bool,
    ) -> None:
        positions_cpu = target_positions.to("cpu", dtype=torch.long).tolist()
        slot_mapping = []
        context_lens = []
        state_slot_ids = []
        for seq, target_position in zip(seqs, positions_cpu):
            cache_position = target_position - 1 if mtp_cache else target_position
            if cache_position < 0:
                raise ValueError("MTP cache has no entry before target position one")
            slot_mapping.append(
                seq.block_table[cache_position // self.block_size]
                * self.block_size
                + cache_position % self.block_size
            )
            context_lens.append(cache_position + 1)
            state_slot_ids.append(seq.state_slot)
        batch_size = len(seqs)
        device = target_positions.device
        set_context(
            False,
            cu_seqlens_q=torch.arange(
                batch_size + 1,
                dtype=torch.int32,
                device=device,
            ),
            slot_mapping=torch.tensor(
                slot_mapping, dtype=torch.int32, device=device
            ),
            context_lens=torch.tensor(
                context_lens, dtype=torch.int32, device=device
            ),
            block_tables=self.prepare_block_tables(seqs),
            state_slot_ids=(
                None
                if mtp_cache
                else torch.tensor(
                    state_slot_ids, dtype=torch.long, device=device
                )
            ),
        )

    def _mtp_step(
        self,
        seqs: list[Sequence],
        input_ids: torch.Tensor,
        target_positions: torch.Tensor,
        hidden_states: torch.Tensor,
        *,
        compute_logits: bool = True,
        profile_name: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        def profile(suffix: str):
            if profile_name is None:
                return nullcontext()
            return self.mtp_phase_profiler.phase(
                f"{profile_name}.{suffix}"
            )

        with profile("context"):
            self._set_offset_decode_context(
                seqs, target_positions, mtp_cache=True
            )
        if not self.enforce_eager and hasattr(self, "mtp_graphs"):
            real_batch_size = input_ids.size(0)
            graph_batch_size = next(
                size for size in self.graph_bs if size >= real_batch_size
            )
            context = get_context()
            variables = self.mtp_graph_vars
            with profile("buffer_update"):
                variables["input_ids"][:real_batch_size] = input_ids
                variables["positions"][:real_batch_size] = target_positions
                variables["hidden_states"][:real_batch_size] = hidden_states
                variables["slot_mapping"].fill_(-1)
                variables["slot_mapping"][:real_batch_size] = context.slot_mapping
                variables["context_lens"].zero_()
                variables["context_lens"][:real_batch_size] = context.context_lens
                variables["block_tables"].fill_(-1)
                variables["block_tables"][
                    :real_batch_size,
                    : context.block_tables.size(1),
                ] = context.block_tables
            with profile("forward"):
                self.mtp_graphs[graph_batch_size].replay()
            draft_hidden = variables["outputs"][:real_batch_size]
            self.mtp_cudagraph_replays += 1
            if not compute_logits:
                return draft_hidden, None
            with profile("lm_head"):
                draft_logits = self.model.compute_logits(draft_hidden)
            return draft_hidden, draft_logits
        cu = get_context().cu_seqlens_q
        with profile("forward"):
            draft_hidden = self.mtp(
                input_ids,
                target_positions,
                hidden_states,
                cu,
            )
        if not compute_logits:
            return draft_hidden, None
        with profile("lm_head"):
            draft_logits = self.model.compute_logits(draft_hidden)
        return draft_hidden, draft_logits

    def _distributed_greedy_tokens(
        self,
        logits: torch.Tensor | None,
        batch_size: int,
    ) -> torch.Tensor:
        if self.rank == 0:
            tokens = logits.float().argmax(dim=-1).to(torch.long)
        else:
            tokens = torch.empty(
                batch_size, dtype=torch.long, device="cuda"
            )
        if self.world_size > 1:
            dist.broadcast(tokens, src=0)
        return tokens

    def _distributed_greedy_matrix(
        self,
        logits: torch.Tensor | None,
        batch_size: int,
        num_steps: int,
    ) -> torch.Tensor:
        if self.rank == 0:
            tokens = logits.float().argmax(dim=-1).view(
                batch_size, num_steps
            )
        else:
            tokens = torch.empty(
                batch_size,
                num_steps,
                dtype=torch.long,
                device="cuda",
            )
        if self.world_size > 1:
            dist.broadcast(tokens, src=0)
        return tokens

    def _prepare_target_verify_chunk(
        self,
        seqs: list[Sequence],
        token_rows: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(seqs) != len(token_rows):
            raise ValueError("verification rows must match sequences")
        flat_tokens = []
        flat_positions = []
        slot_mapping = []
        cu_q = [0]
        cu_k = [0]
        max_q = 0
        max_k = 0
        for seq, row in zip(seqs, token_rows):
            q_len = row.numel()
            if q_len <= 0:
                raise ValueError("verification prefix must be non-empty")
            start = len(seq)
            positions = torch.arange(
                start,
                start + q_len,
                dtype=torch.long,
                device=row.device,
            )
            flat_tokens.append(row)
            flat_positions.append(positions)
            for position in positions.to("cpu").tolist():
                slot_mapping.append(
                    seq.block_table[position // self.block_size]
                    * self.block_size
                    + position % self.block_size
                )
            context_length = start + q_len
            cu_q.append(cu_q[-1] + q_len)
            cu_k.append(cu_k[-1] + context_length)
            max_q = max(max_q, q_len)
            max_k = max(max_k, context_length)
        device = token_rows[0].device
        set_context(
            True,
            cu_seqlens_q=torch.tensor(
                cu_q, dtype=torch.int32, device=device
            ),
            cu_seqlens_k=torch.tensor(
                cu_k, dtype=torch.int32, device=device
            ),
            max_seqlen_q=max_q,
            max_seqlen_k=max_k,
            slot_mapping=torch.tensor(
                slot_mapping, dtype=torch.int32, device=device
            ),
            block_tables=self.prepare_block_tables(seqs),
            state_slot_ids=torch.tensor(
                [seq.state_slot for seq in seqs],
                dtype=torch.long,
                device=device,
            ),
        )
        return torch.cat(flat_tokens), torch.cat(flat_positions)

    def _replay_hybrid_decode_graph(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        real_batch_size = input_ids.size(0)
        graph_batch_size = next(
            size for size in self.graph_bs if size >= real_batch_size
        )
        padding = graph_batch_size - real_batch_size
        self.decode_cudagraph_padded_rows += padding
        self.decode_cudagraph_max_padding = max(
            self.decode_cudagraph_max_padding,
            padding,
        )
        self.decode_cudagraph_last_real_batch_size = real_batch_size
        self.decode_cudagraph_last_graph_batch_size = graph_batch_size
        context = get_context()
        graph = self.graphs[graph_batch_size]
        graph_vars = self.graph_vars
        graph_vars["input_ids"][:real_batch_size] = input_ids
        graph_vars["positions"][:real_batch_size] = positions
        graph_vars["slot_mapping"].fill_(-1)
        graph_vars["slot_mapping"][:real_batch_size] = context.slot_mapping
        graph_vars["context_lens"].zero_()
        graph_vars["context_lens"][:real_batch_size] = context.context_lens
        graph_vars["block_tables"].fill_(-1)
        graph_vars["block_tables"][
            :real_batch_size,
            : context.block_tables.size(1),
        ] = context.block_tables
        graph_vars["state_slot_ids"].fill_(
            self.state_manager.scratch_slot_id
        )
        graph_vars["state_slot_ids"][:real_batch_size] = (
            context.state_slot_ids
        )
        self.state_manager.reset_scratch()
        graph.replay()
        hidden_states = graph_vars["outputs"][:real_batch_size]
        return hidden_states, self.model.compute_logits(hidden_states)

    def _replay_verify_graph(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        batch_size: int,
    ):
        context = get_context()
        variables = self.verify_graph_vars
        num_tokens = input_ids.numel()
        variables["input_ids"][:num_tokens] = input_ids
        variables["positions"][:num_tokens] = positions
        variables["slot_mapping"].fill_(-1)
        variables["slot_mapping"][:num_tokens] = context.slot_mapping
        variables["cu_seqlens_k"][: batch_size + 1] = context.cu_seqlens_k
        variables["block_tables"].fill_(-1)
        variables["block_tables"][
            :batch_size,
            : context.block_tables.size(1),
        ] = context.block_tables
        variables["state_slot_ids"].fill_(
            self.state_manager.scratch_slot_id
        )
        variables["state_slot_ids"][:batch_size] = context.state_slot_ids
        self.state_manager.reset_scratch()
        self.verify_graphs[batch_size].replay()

        hidden_states = variables["outputs"][:num_tokens]
        old_is_prefill = context.is_prefill
        context.is_prefill = False
        logits = self.model.compute_logits(hidden_states)
        context.is_prefill = old_is_prefill
        num_layers = len(self.model.model.layers)
        recurrent_histories = [None] * num_layers
        conv_histories = [None] * num_layers
        history_steps = variables["history_steps"]
        if history_steps:
            num_history_tokens = batch_size * history_steps
            for compact_index, layer_index in enumerate(
                self.model.model.linear_attention_layer_indices
            ):
                recurrent_histories[layer_index] = variables[
                    "recurrent_history"
                ][compact_index, :num_history_tokens]
                conv_histories[layer_index] = variables["conv_history"][
                    compact_index, :num_history_tokens
                ]
        self.speculative_verify_cudagraph_replays += 1
        return hidden_states, logits, recurrent_histories, conv_histories

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if self.state_manager is not None:
            if not is_prefill and not self.enforce_eager:
                self.decode_cudagraph_replays += 1
                _, logits = self._replay_hybrid_decode_graph(
                    input_ids, positions
                )
                return logits

            context = get_context()
            hidden_states, logits = self._forward_target_eager(
                input_ids, positions
            )
            if is_prefill:
                self.prefill_model_runs += 1
            else:
                self.decode_eager_runs += 1
            return logits
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            if is_prefill:
                self.prefill_model_runs += 1
            else:
                self.decode_eager_runs += 1
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            self.decode_cudagraph_replays += 1
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def reset_metrics(self):
        self.mtp_phase_profiler.reset()
        self.prefill_model_runs = 0
        self.decode_eager_runs = 0
        self.decode_cudagraph_replays = 0
        self.decode_cudagraph_padded_rows = 0
        self.decode_cudagraph_max_padding = 0
        self.decode_cudagraph_last_real_batch_size = 0
        self.decode_cudagraph_last_graph_batch_size = 0
        self.speculative_decode_rounds = 0
        self.speculative_proposed_tokens = 0
        self.speculative_accepted_tokens = 0
        self.speculative_emitted_tokens = 0
        self.speculative_rejected_sequences = 0
        self.speculative_verify_forwards = 0
        self.speculative_recompute_forwards = 0
        self.speculative_target_cudagraph_replays = 0
        self.mtp_cudagraph_replays = 0
        self.speculative_verify_cudagraph_replays = 0
        self.hybrid_prefix_capture_count = 0
        self.hybrid_prefix_capture_ms = 0.0
        self.hybrid_prefix_restore_count = 0
        self.hybrid_prefix_restore_ms = 0.0
        self.hybrid_prefix_cow_count = 0
        self.hybrid_prefix_cow_bytes = 0
        self.hybrid_prefix_cow_ms = 0.0
        self.hybrid_internal_checkpoint_count = 0
        self.hybrid_internal_checkpoint_bytes = 0
        self.hybrid_internal_prefill_ms = 0.0

    def allocate_state_slot(self, sequence: Sequence) -> None:
        if self.state_manager is not None:
            slot = self.state_manager.allocate(sequence)
            if self.mtp_prev_target_hidden is not None:
                self.mtp_prev_target_hidden[slot].zero_()

    def free_state_slot(self, sequence: Sequence) -> None:
        if self.state_manager is not None:
            slot = sequence.state_slot
            if slot is not None and self.mtp_prev_target_hidden is not None:
                self.mtp_prev_target_hidden[slot].zero_()
            self.state_manager.free(sequence)

    def capture_prefix_checkpoint(
        self,
        request_state_slot: int,
        boundary_tokens: int | None = None,
    ) -> int:
        if self.prefix_checkpoint_pool is None:
            raise RuntimeError("hybrid prefix checkpoint pool is disabled")
        started = perf_counter()
        internal = self.internal_prefix_states.pop(
            (request_state_slot, boundary_tokens),
            None,
        )
        if internal is None:
            checkpoint_slot = transactional_capture_prefix_checkpoint(
                self.prefix_checkpoint_pool,
                request_state_slot,
            )
        else:
            checkpoint_slot = transactional_capture_prefix_checkpoint(
                self.prefix_checkpoint_pool,
                request_state_slot,
                recurrent_state=internal[0],
                conv_state=internal[1],
            )
        torch.cuda.synchronize()
        self.hybrid_prefix_capture_count += 1
        self.hybrid_prefix_capture_ms += (perf_counter() - started) * 1000
        return checkpoint_slot

    def restore_prefix_checkpoint(
        self,
        sequence: Sequence,
        checkpoint_slot: int,
    ) -> None:
        if self.prefix_checkpoint_pool is None:
            raise RuntimeError("hybrid prefix checkpoint pool is disabled")
        if sequence.state_slot is None:
            raise ValueError("sequence has no request state slot")
        started = perf_counter()
        transactional_restore_prefix_checkpoint(
            self.prefix_checkpoint_pool,
            checkpoint_slot,
            sequence.state_slot,
        )
        torch.cuda.synchronize()
        self.hybrid_prefix_restore_count += 1
        self.hybrid_prefix_restore_ms += (perf_counter() - started) * 1000

    def evict_prefix_checkpoint(self, checkpoint_slot: int) -> None:
        if self.prefix_checkpoint_pool is None:
            raise RuntimeError("hybrid prefix checkpoint pool is disabled")
        self.prefix_checkpoint_pool.free(checkpoint_slot)

    def discard_internal_prefix_states(self, request_state_slot: int) -> None:
        stale = [
            key for key in self.internal_prefix_states
            if key[0] == request_state_slot
        ]
        for key in stale:
            del self.internal_prefix_states[key]

    def copy_prefix_kv(
        self,
        source_block_id: int,
        destination_block_id: int,
        num_tokens: int,
    ) -> None:
        if not 0 < num_tokens < self.block_size:
            raise ValueError("partial KV COW must copy part of one page")
        started = perf_counter()
        local_error = None
        try:
            source = self.kv_cache[:, :, source_block_id, :num_tokens]
            destination = self.kv_cache[
                :, :, destination_block_id, :num_tokens
            ]
            destination.copy_(source)
            if self.kv_cache.is_cuda:
                torch.cuda.synchronize()
        except Exception as exc:
            local_error = exc
        if dist.is_initialized() and dist.get_world_size() > 1:
            success = torch.tensor(
                0 if local_error is not None else 1,
                dtype=torch.int32,
                device=self.kv_cache.device,
            )
            dist.all_reduce(success, op=dist.ReduceOp.MIN)
            global_success = int(success.item()) == 1
        else:
            global_success = local_error is None
        if not global_success:
            if not dist.is_initialized() or self.rank == 0:
                raise RuntimeError("partial KV COW rolled back on all ranks")
            return
        copied_bytes = source.numel() * source.element_size()
        self.hybrid_prefix_cow_count += 1
        self.hybrid_prefix_cow_bytes += copied_bytes
        self.hybrid_prefix_cow_ms += (perf_counter() - started) * 1000

    def get_metrics(self):
        metrics = {
            "cuda_graph_enabled": not self.enforce_eager,
            "prefill_model_runs": self.prefill_model_runs,
            "decode_eager_runs": self.decode_eager_runs,
            "decode_cudagraph_replays": self.decode_cudagraph_replays,
            "decode_cudagraph_padded_rows": self.decode_cudagraph_padded_rows,
            "decode_cudagraph_max_padding": self.decode_cudagraph_max_padding,
            "decode_cudagraph_last_real_batch_size": (
                self.decode_cudagraph_last_real_batch_size
            ),
            "decode_cudagraph_last_graph_batch_size": (
                self.decode_cudagraph_last_graph_batch_size
            ),
            "speculative_decode_rounds": self.speculative_decode_rounds,
            "speculative_proposed_tokens": self.speculative_proposed_tokens,
            "speculative_accepted_tokens": self.speculative_accepted_tokens,
            "speculative_emitted_tokens": self.speculative_emitted_tokens,
            "speculative_rejected_sequences": self.speculative_rejected_sequences,
            "speculative_verify_forwards": self.speculative_verify_forwards,
            "speculative_recompute_forwards": self.speculative_recompute_forwards,
            "speculative_target_cudagraph_replays": self.speculative_target_cudagraph_replays,
            "mtp_cudagraph_replays": self.mtp_cudagraph_replays,
            "speculative_verify_cudagraph_replays": self.speculative_verify_cudagraph_replays,
            "speculative_acceptance_rate": (
                self.speculative_accepted_tokens
                / self.speculative_proposed_tokens
                if self.speculative_proposed_tokens
                else 0.0
            ),
            "mtp_phase_profile": self.mtp_phase_profiler.metrics(),
        }
        if self.prefix_checkpoint_pool is not None:
            metrics.update(
                {
                    "hybrid_prefix_checkpoint_slots_total": (
                        self.prefix_checkpoint_pool.capacity
                    ),
                    "hybrid_prefix_checkpoint_slots_used": len(
                        self.prefix_checkpoint_pool.used_checkpoint_slots
                    ),
                    "hybrid_prefix_checkpoint_bytes": (
                        self.prefix_checkpoint_pool.memory_bytes()
                    ),
                    "hybrid_prefix_checkpoint_bytes_per_slot": (
                        self.prefix_checkpoint_pool.bytes_per_checkpoint
                    ),
                    "hybrid_prefix_checkpoint_dtype": (
                        self.prefix_checkpoint_pool.checkpoint_dtype
                    ),
                    "hybrid_prefix_checkpoint_recurrent_bytes_per_slot": (
                        self.prefix_checkpoint_pool
                        .recurrent_bytes_per_checkpoint
                    ),
                    "hybrid_prefix_checkpoint_scale_bytes_per_slot": (
                        self.prefix_checkpoint_pool.scale_bytes_per_checkpoint
                    ),
                    "hybrid_prefix_checkpoint_conv_bytes_per_slot": (
                        self.prefix_checkpoint_pool.conv_bytes_per_checkpoint
                    ),
                    "hybrid_prefix_capture_count": self.hybrid_prefix_capture_count,
                    "hybrid_prefix_capture_ms": self.hybrid_prefix_capture_ms,
                    "hybrid_prefix_restore_count": self.hybrid_prefix_restore_count,
                    "hybrid_prefix_restore_ms": self.hybrid_prefix_restore_ms,
                    "hybrid_prefix_cow_count": self.hybrid_prefix_cow_count,
                    "hybrid_prefix_cow_bytes": self.hybrid_prefix_cow_bytes,
                    "hybrid_prefix_cow_ms": self.hybrid_prefix_cow_ms,
                    "hybrid_internal_checkpoint_count": (
                        self.hybrid_internal_checkpoint_count
                    ),
                    "hybrid_internal_checkpoint_bytes": (
                        self.hybrid_internal_checkpoint_bytes
                    ),
                    "hybrid_internal_prefill_ms": (
                        self.hybrid_internal_prefill_ms
                    ),
                }
            )
        return metrics

    @torch.inference_mode()
    def run(
        self,
        seqs: list[Sequence],
        is_prefill: bool,
        internal_prefix_boundaries: dict[int, tuple[int, ...]] | None = None,
    ) -> list[int]:
        if is_prefill:
            input_ids, positions = self.prepare_prefill(seqs)
        else:
            with self.mtp_phase_profiler.phase(
                "target_only.prepare_decode"
            ):
                input_ids, positions = self.prepare_decode(seqs)
        if is_prefill:
            temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        else:
            with self.mtp_phase_profiler.phase(
                "target_only.prepare_sample",
                gpu=self.rank == 0,
            ):
                temperatures = (
                    self.prepare_sample(seqs) if self.rank == 0 else None
                )
        if is_prefill and internal_prefix_boundaries:
            if len(seqs) != 1:
                raise RuntimeError("internal checkpoints require one prefill sequence")
            seq = seqs[0]
            boundaries = internal_prefix_boundaries.get(seq.seq_id, ())
            started = perf_counter()
            relative_indices = tuple(
                boundary - seq.num_cached_tokens - 1
                for boundary in boundaries
            )
            (
                _,
                logits,
                recurrent_histories,
                conv_histories,
            ) = self._forward_target_eager(
                input_ids,
                positions,
                state_checkpoint_indices=relative_indices,
            )
            linear_indices = self.model.model.linear_attention_layer_indices
            for checkpoint_index, boundary in enumerate(boundaries):
                recurrent = torch.stack(
                    [
                        recurrent_histories[layer_index][checkpoint_index]
                        for layer_index in linear_indices
                    ]
                )
                conv = torch.stack(
                    [
                        conv_histories[layer_index][checkpoint_index]
                        for layer_index in linear_indices
                    ]
                )
                self.internal_prefix_states[(seq.state_slot, boundary)] = (
                    recurrent,
                    conv,
                )
                self.hybrid_internal_checkpoint_count += 1
                self.hybrid_internal_checkpoint_bytes += (
                    recurrent.numel() * recurrent.element_size()
                    + conv.numel() * conv.element_size()
                )
            self.hybrid_internal_prefill_ms += (
                perf_counter() - started
            ) * 1000
            self.prefill_model_runs += 1
            token_ids = (
                self.sampler(logits, temperatures).tolist()
                if self.rank == 0
                else None
            )
        elif is_prefill and self.mtp is not None:
            hidden_states, logits = self._forward_target_eager(
                input_ids, positions
            )
            self.prefill_model_runs += 1
            last_hidden = []
            cursor = 0
            for seq in seqs:
                cursor += seq.num_scheduled_tokens
                last_hidden.append(hidden_states[cursor - 1])
            self._seed_mtp_prefill(
                seqs,
                input_ids,
                positions,
                hidden_states,
            )
            sampled = self._distributed_greedy_tokens(logits, len(seqs))
            # The first completion token is produced by the target prefill.
            # Insert its shifted MTP entry now; otherwise the first speculative
            # round reads an uninitialized cache position.
            completed_indices = [
                index
                for index, seq in enumerate(seqs)
                if seq.num_cached_tokens + seq.num_scheduled_tokens
                == seq.num_tokens
                and seq.block_table
            ]
            if completed_indices:
                completed_seqs = [seqs[index] for index in completed_indices]
                index_tensor = torch.tensor(
                    completed_indices, dtype=torch.long, device="cuda"
                )
                self._mtp_step(
                    completed_seqs,
                    sampled.index_select(0, index_tensor),
                    torch.tensor(
                        [seq.num_tokens for seq in completed_seqs],
                        dtype=torch.long,
                        device="cuda",
                    ),
                    torch.stack(last_hidden).index_select(0, index_tensor),
                )
            token_ids = sampled.tolist() if self.rank == 0 else None
        else:
            if is_prefill:
                logits = self.run_model(input_ids, positions, is_prefill)
                token_ids = (
                    self.sampler(logits, temperatures).tolist()
                    if self.rank == 0
                    else None
                )
            else:
                with self.mtp_phase_profiler.phase(
                    "target_only.decode_forward"
                ):
                    logits = self.run_model(input_ids, positions, is_prefill)
                with self.mtp_phase_profiler.phase(
                    "target_only.sample",
                    gpu=self.rank == 0,
                ):
                    token_ids = (
                        self.sampler(logits, temperatures).tolist()
                        if self.rank == 0
                        else None
                    )
        if not is_prefill:
            self.mtp_phase_profiler.flush()
        reset_context()
        return token_ids

    @torch.inference_mode()
    def run_speculative(self, seqs: list[Sequence]) -> list[list[int]] | None:
        """Greedy native-MTP proposal with sequential batched verification.

        Target verification is deliberately eager and stepwise so every GDN
        boundary can be snapshotted. Rejected KV entries remain physically in
        cache but are unreachable through the committed logical lengths.
        """
        if self.mtp is None or self.num_speculative_tokens <= 0:
            raise RuntimeError("native MTP speculative decoding is disabled")
        if any(seq.temperature > 1e-10 for seq in seqs):
            raise ValueError("native MTP currently supports greedy sampling only")

        iteration_started_ns = perf_counter_ns()
        batch_size = len(seqs)
        with self.mtp_phase_profiler.phase("mtp.prepare_decode"):
            input_ids, positions = self.prepare_decode(seqs)
        with self.mtp_phase_profiler.phase("mtp.target_decode"):
            if self.enforce_eager:
                target_hidden, target_logits = self._forward_target_eager(
                    input_ids, positions
                )
            else:
                target_hidden, target_logits = self._replay_hybrid_decode_graph(
                    input_ids, positions
                )
                self.speculative_target_cudagraph_replays += 1
        with self.mtp_phase_profiler.phase("mtp.base_argmax"):
            target_state_slot_ids = get_context().state_slot_ids.clone()
            base_tokens = self._distributed_greedy_tokens(
                target_logits, batch_size
            )

        with self.mtp_phase_profiler.phase("mtp.initial_metadata"):
            initial_lengths = torch.tensor(
                [len(seq) for seq in seqs],
                dtype=torch.long,
                device="cuda",
            )
        proposals = []
        draft_input = base_tokens
        draft_hidden = target_hidden
        for step in range(self.num_speculative_tokens):
            draft_positions = initial_lengths + step
            draft_hidden, draft_logits = self._mtp_step(
                seqs,
                draft_input,
                draft_positions,
                draft_hidden,
                profile_name=f"mtp.draft.{step}",
            )
            with self.mtp_phase_profiler.phase(
                f"mtp.draft.{step}.argmax"
            ):
                draft_input = self._distributed_greedy_tokens(
                    draft_logits, batch_size
                )
            proposals.append(draft_input)
        proposal_tokens = torch.stack(proposals, dim=1)

        if self.speculative_parallel_verify:
            rollback_history_steps = max(
                self.num_speculative_tokens - 1,
                0,
            )
            rollback_checkpoint_indices = tuple(
                range(rollback_history_steps)
            )
            with self.mtp_phase_profiler.phase("mtp.verify_prepare"):
                verify_rows = [
                    torch.cat(
                        (
                            base_tokens[index : index + 1],
                            proposal_tokens[index, :-1],
                        )
                    )
                    for index in range(batch_size)
                ]
                verify_ids, verify_positions = self._prepare_target_verify_chunk(
                    seqs, verify_rows
                )
            with self.mtp_phase_profiler.phase("mtp.verify_forward"):
                if (
                    not self.enforce_eager
                    and hasattr(self, "verify_graphs")
                    and batch_size in self.verify_graphs
                ):
                    (
                        all_hidden,
                        all_logits,
                        recurrent_histories,
                        conv_histories,
                    ) = self._replay_verify_graph(
                        verify_ids,
                        verify_positions,
                        batch_size,
                    )
                else:
                    if rollback_history_steps:
                        (
                            all_hidden,
                            all_logits,
                            recurrent_histories,
                            conv_histories,
                        ) = self._forward_target_eager(
                            verify_ids,
                            verify_positions,
                            all_logits=True,
                            state_checkpoint_indices=(
                                rollback_checkpoint_indices
                            ),
                        )
                    else:
                        all_hidden, all_logits = (
                            self._forward_target_eager(
                                verify_ids,
                                verify_positions,
                                all_logits=True,
                            )
                        )
                        num_layers = len(self.model.model.layers)
                        recurrent_histories = [None] * num_layers
                        conv_histories = [None] * num_layers
            with self.mtp_phase_profiler.phase("mtp.verify_argmax"):
                target_tokens = self._distributed_greedy_matrix(
                    all_logits,
                    batch_size,
                    self.num_speculative_tokens,
                )
                hidden_stack = all_hidden.view(
                    batch_size,
                    self.num_speculative_tokens,
                    -1,
                )
            if self.rank == 0:
                self.speculative_verify_forwards += 1
        else:
            verifier_tokens = []
            verifier_hidden = []
            transaction = None
            for step in range(self.num_speculative_tokens):
                verify_input = (
                    base_tokens
                    if step == 0
                    else proposal_tokens[:, step - 1]
                )
                verify_positions = initial_lengths + step
                self._set_offset_decode_context(
                    seqs, verify_positions, mtp_cache=False
                )
                hidden, logits = self._forward_target_eager(
                    verify_input, verify_positions
                )
                verifier_hidden.append(hidden)
                verifier_tokens.append(
                    self._distributed_greedy_tokens(logits, batch_size)
                )
                if step == 0:
                    transaction = HybridSpeculativeTransaction(
                        self.state_manager,
                        get_context().state_slot_ids,
                    )
                else:
                    transaction.capture_step()
            target_tokens = torch.stack(verifier_tokens, dim=1)
            hidden_stack = torch.stack(verifier_hidden, dim=1)
            if self.rank == 0:
                self.speculative_verify_forwards += self.num_speculative_tokens

        with self.mtp_phase_profiler.phase(
            "mtp.accept_reject", gpu=False
        ):
            if self.rank == 0:
                verification = greedy_verify_tokens(
                    proposal_tokens, target_tokens
                )
                accepted_lengths = verification.accepted_lengths.to(
                    device="cuda"
                )
            else:
                verification = None
                accepted_lengths = torch.empty(
                    batch_size, dtype=torch.int32, device="cuda"
                )
            if self.world_size > 1:
                dist.broadcast(accepted_lengths, src=0)

        with self.mtp_phase_profiler.phase("mtp.state_boundary"):
            state_boundaries = accepted_lengths.clamp_max(
                self.num_speculative_tokens - 1
            )
        if self.speculative_parallel_verify:
            with self.mtp_phase_profiler.phase("mtp.state_select"):
                # Verify has already committed the state after its final input
                # token to each request's active slot. Keep that state in place
                # whenever it is the accepted boundary and only materialize
                # rows that must roll back to an earlier boundary.
                final_verify_boundary = self.num_speculative_tokens - 1
                rollback_rows = torch.nonzero(
                    state_boundaries < final_verify_boundary,
                    as_tuple=False,
                ).flatten()
                rollback_count = rollback_rows.numel()
                selected_recurrent = []
                selected_conv = []
                if rollback_count:
                    history_indices = (
                        rollback_rows
                        * rollback_history_steps
                        + state_boundaries.index_select(
                            0, rollback_rows
                        ).to(torch.long)
                    )
                    selected_recurrent = [
                        None
                        if history is None
                        else history.index_select(0, history_indices)
                        for history in recurrent_histories
                    ]
                    selected_conv = [
                        None
                        if history is None
                        else history.index_select(0, history_indices)
                        for history in conv_histories
                    ]
            with self.mtp_phase_profiler.phase("mtp.state_scatter"):
                if rollback_count:
                    self.state_manager.scatter(
                        target_state_slot_ids.index_select(0, rollback_rows),
                        selected_recurrent,
                        selected_conv,
                        linear_layer_indices=(
                            self.model.model.linear_attention_layer_indices
                        ),
                    )
            history_bytes = sum(
                history.numel() * history.element_size()
                for history in recurrent_histories + conv_histories
                if history is not None
            )
            selected_state_bytes = sum(
                state.numel() * state.element_size()
                for state in selected_recurrent + selected_conv
                if state is not None
            )
            self.mtp_phase_profiler.add_counter(
                "state_history_materialized_bytes", history_bytes
            )
            dense_history_bytes = (
                self.state_manager.bytes_per_slot()
                * batch_size
                * self.num_speculative_tokens
            )
            self.mtp_phase_profiler.add_counter(
                "state_history_dense_equivalent_bytes",
                dense_history_bytes,
            )
            self.mtp_phase_profiler.add_counter(
                "state_history_final_boundary_avoided_bytes",
                dense_history_bytes - history_bytes,
            )
            self.mtp_phase_profiler.add_counter(
                "selected_state_bytes", selected_state_bytes
            )
            self.mtp_phase_profiler.add_counter(
                "state_scatter_logical_bytes", selected_state_bytes
            )
            self.mtp_phase_profiler.add_counter(
                "state_active_reuse_rows",
                batch_size - rollback_count,
            )
            self.mtp_phase_profiler.add_counter(
                "state_rollback_rows", rollback_count
            )
        else:
            # Snapshot zero is after the guaranteed base token; snapshot i is
            # after i further accepted proposals. If all are accepted, the
            # final proposal stays pending and uses the last snapshot.
            with self.mtp_phase_profiler.phase("mtp.state_transaction"):
                transaction.commit(state_boundaries)

        with self.mtp_phase_profiler.phase(
            "mtp.output_metadata", gpu=False
        ):
            if self.rank == 0:
                output_tokens = [
                    [int(base)] + verified
                    for base, verified in zip(
                        base_tokens.to("cpu").tolist(),
                        verification.output_tokens,
                    )
                ]
                final_tokens = torch.tensor(
                    [tokens[-1] for tokens in output_tokens],
                    dtype=torch.long,
                    device="cuda",
                )
                final_positions = torch.tensor(
                    [
                        len(seq) + len(tokens) - 1
                        for seq, tokens in zip(seqs, output_tokens)
                    ],
                    dtype=torch.long,
                    device="cuda",
                )
                accepted_cpu = verification.accepted_lengths.to("cpu").tolist()
                self.speculative_decode_rounds += 1
                self.speculative_proposed_tokens += (
                    batch_size * self.num_speculative_tokens
                )
                self.speculative_accepted_tokens += sum(accepted_cpu)
                self.speculative_emitted_tokens += sum(
                    len(tokens) for tokens in output_tokens
                )
                self.speculative_rejected_sequences += sum(
                    accepted < self.num_speculative_tokens
                    for accepted in accepted_cpu
                )
            else:
                output_tokens = None
                final_tokens = torch.empty(
                    batch_size, dtype=torch.long, device="cuda"
                )
                final_positions = torch.empty(
                    batch_size, dtype=torch.long, device="cuda"
                )
            if self.world_size > 1:
                dist.broadcast(final_tokens, src=0)
                dist.broadcast(final_positions, src=0)
        with self.mtp_phase_profiler.phase("mtp.hidden_select"):
            selected_hidden = hidden_stack[
                torch.arange(batch_size, device="cuda"),
                state_boundaries.to(torch.long),
            ]
        # Correct/complete the shifted MTP KV at the final emitted token.
        self._mtp_step(
            seqs,
            final_tokens,
            final_positions,
            selected_hidden,
            compute_logits=False,
            profile_name="mtp.final_alignment",
        )
        self.mtp_phase_profiler.add_counter(
            "final_alignment_lm_head_skipped", batch_size
        )
        if self.enforce_eager:
            self.decode_eager_runs += 1
        self.mtp_phase_profiler.add_counter("iterations", 1)
        self.mtp_phase_profiler.add_counter(
            "iteration_body_cpu_ms",
            (perf_counter_ns() - iteration_started_ns) / 1e6,
        )
        if hasattr(self, "verify_graph_vars"):
            graph_history_bytes = (
                self.verify_graph_vars["recurrent_history"].numel()
                * self.verify_graph_vars["recurrent_history"].element_size()
                + self.verify_graph_vars["conv_history"].numel()
                * self.verify_graph_vars["conv_history"].element_size()
            )
            self.mtp_phase_profiler.set_counter(
                "verify_graph_history_capacity_bytes",
                graph_history_bytes,
            )
            dense_graph_history_bytes = (
                self.state_manager.bytes_per_slot()
                * max(self.graph_bs)
                * self.num_speculative_tokens
            )
            self.mtp_phase_profiler.set_counter(
                "verify_graph_dense_history_capacity_bytes",
                dense_graph_history_bytes,
            )
            self.mtp_phase_profiler.set_counter(
                "verify_graph_history_capacity_saved_bytes",
                dense_graph_history_bytes - graph_history_bytes,
            )
        self.mtp_phase_profiler.flush()
        reset_context()
        return output_tokens

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = sorted(
            set(
                [size for size in (1, 2, 4, 8) if size <= max_bs]
                + list(range(16, max_bs + 1, 16))
                + [max_bs]
            )
        )
        self.graphs = {}
        self.graph_pool = None

        if self.state_manager is not None:
            slot_mapping.fill_(-1)
            block_tables.fill_(-1)
            cu_seqlens_q = torch.arange(max_bs + 1, dtype=torch.int32)
            state_slot_ids = torch.full(
                (max_bs,),
                self.state_manager.scratch_slot_id,
                dtype=torch.int64,
            )
            self.graph_vars = dict(
                input_ids=input_ids,
                positions=positions,
                slot_mapping=slot_mapping,
                context_lens=context_lens,
                block_tables=block_tables,
                outputs=outputs,
                cu_seqlens_q=cu_seqlens_q,
                state_slot_ids=state_slot_ids,
            )
            num_layers = len(self.model.model.layers)
            linear_indices = self.model.model.linear_attention_layer_indices

            for bs in reversed(self.graph_bs):
                graph = torch.cuda.CUDAGraph()
                cu = cu_seqlens_q[: bs + 1]
                slots = state_slot_ids[:bs]
                set_context(
                    False,
                    cu_seqlens_q=cu,
                    slot_mapping=slot_mapping[:bs],
                    context_lens=context_lens[:bs],
                    block_tables=block_tables[:bs],
                    state_slot_ids=slots,
                )

                def step():
                    recurrent_states, conv_states = self.state_manager.gather(
                        slots,
                        num_total_layers=num_layers,
                        linear_layer_indices=linear_indices,
                    )
                    hidden_states, new_recurrent_states, new_conv_states = (
                        self.model(
                            input_ids[:bs],
                            positions[:bs],
                            cu,
                            recurrent_states,
                            conv_states,
                        )
                    )
                    outputs[:bs] = hidden_states
                    self.state_manager.scatter(
                        slots,
                        new_recurrent_states,
                        new_conv_states,
                        linear_layer_indices=linear_indices,
                    )

                step()
                self.state_manager.reset_scratch()
                torch.cuda.synchronize()
                with torch.cuda.graph(graph, self.graph_pool):
                    step()
                if self.graph_pool is None:
                    self.graph_pool = graph.pool()
                self.graphs[bs] = graph
                torch.cuda.synchronize()
                self.state_manager.reset_scratch()
                reset_context()
            if self.mtp is not None:
                self.capture_mtp_cudagraph()
                if self.speculative_parallel_verify:
                    self.capture_verify_cudagraph()
            return

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )

    @torch.inference_mode()
    def capture_mtp_cudagraph(self) -> None:
        """Capture one-token native-MTP proposal buckets."""
        max_bs = max(self.graph_bs)
        max_num_blocks = (
            self.config.max_model_len + self.block_size - 1
        ) // self.block_size
        hidden_size = self.config.hf_config.hidden_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        hidden_states = torch.zeros(
            max_bs,
            hidden_size,
            dtype=self.config.hf_config.dtype,
        )
        slot_mapping = torch.full((max_bs,), -1, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.full(
            (max_bs, max_num_blocks), -1, dtype=torch.int32
        )
        outputs = torch.zeros(
            max_bs,
            hidden_size,
            dtype=self.config.hf_config.dtype,
        )
        cu_seqlens_q = torch.arange(max_bs + 1, dtype=torch.int32)
        self.mtp_graph_vars = {
            "input_ids": input_ids,
            "positions": positions,
            "hidden_states": hidden_states,
            "slot_mapping": slot_mapping,
            "context_lens": context_lens,
            "block_tables": block_tables,
            "outputs": outputs,
            "cu_seqlens_q": cu_seqlens_q,
        }
        self.mtp_graphs = {}
        for batch_size in reversed(self.graph_bs):
            cu = cu_seqlens_q[: batch_size + 1]
            set_context(
                False,
                cu_seqlens_q=cu,
                slot_mapping=slot_mapping[:batch_size],
                context_lens=context_lens[:batch_size],
                block_tables=block_tables[:batch_size],
            )

            def step():
                outputs[:batch_size] = self.mtp(
                    input_ids[:batch_size],
                    positions[:batch_size],
                    hidden_states[:batch_size],
                    cu,
                )

            step()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, self.graph_pool):
                step()
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.mtp_graphs[batch_size] = graph
            torch.cuda.synchronize()
            reset_context()

    @torch.inference_mode()
    def capture_verify_cudagraph(self) -> None:
        """Capture fixed-width parallel target verification buckets."""
        max_bs = max(self.graph_bs)
        steps = self.num_speculative_tokens
        history_steps = max(steps - 1, 0)
        rollback_checkpoint_indices = tuple(range(history_steps))
        max_tokens = max_bs * steps
        max_history_tokens = max_bs * history_steps
        max_num_blocks = (
            self.config.max_model_len + self.block_size - 1
        ) // self.block_size
        hf_config = self.config.hf_config
        reference_layer = next(
            module
            for module in self.model.modules()
            if isinstance(module, GatedDeltaNet)
        )
        input_ids = torch.zeros(max_tokens, dtype=torch.int64)
        positions = torch.zeros(max_tokens, dtype=torch.int64)
        slot_mapping = torch.full((max_tokens,), -1, dtype=torch.int32)
        cu_seqlens_q = torch.arange(
            0,
            (max_bs + 1) * steps,
            steps,
            dtype=torch.int32,
        )
        cu_seqlens_k = cu_seqlens_q.clone()
        block_tables = torch.zeros(
            max_bs, max_num_blocks, dtype=torch.int32
        )
        state_slot_ids = torch.full(
            (max_bs,),
            self.state_manager.scratch_slot_id,
            dtype=torch.long,
        )
        outputs = torch.zeros(
            max_tokens,
            hf_config.hidden_size,
            dtype=hf_config.dtype,
        )
        recurrent_history = torch.zeros(
            self.state_manager.num_linear_layers,
            max_history_tokens,
            reference_layer.num_value_heads,
            reference_layer.key_head_dim,
            reference_layer.value_head_dim,
            dtype=torch.float32,
        )
        conv_history = torch.zeros(
            self.state_manager.num_linear_layers,
            max_history_tokens,
            reference_layer.conv_dim,
            reference_layer.conv_kernel_size - 1,
            dtype=hf_config.dtype,
        )
        self.verify_graph_vars = {
            "input_ids": input_ids,
            "positions": positions,
            "slot_mapping": slot_mapping,
            "cu_seqlens_q": cu_seqlens_q,
            "cu_seqlens_k": cu_seqlens_k,
            "block_tables": block_tables,
            "state_slot_ids": state_slot_ids,
            "outputs": outputs,
            "recurrent_history": recurrent_history,
            "conv_history": conv_history,
            "history_steps": history_steps,
        }
        self.verify_graphs = {}
        num_layers = len(self.model.model.layers)
        linear_indices = self.model.model.linear_attention_layer_indices
        for batch_size in reversed(self.graph_bs):
            num_tokens = batch_size * steps
            num_history_tokens = batch_size * history_steps
            cu_q = cu_seqlens_q[: batch_size + 1]
            cu_k = cu_seqlens_k[: batch_size + 1]
            slots = state_slot_ids[:batch_size]
            set_context(
                True,
                cu_seqlens_q=cu_q,
                cu_seqlens_k=cu_k,
                max_seqlen_q=steps,
                max_seqlen_k=self.config.max_model_len,
                slot_mapping=slot_mapping[:num_tokens],
                block_tables=block_tables[:batch_size],
                state_slot_ids=slots,
            )

            def step():
                recurrent_states, conv_states = self.state_manager.gather(
                    slots,
                    num_total_layers=num_layers,
                    linear_layer_indices=linear_indices,
                )
                if history_steps:
                    (
                        hidden_states,
                        new_recurrent_states,
                        new_conv_states,
                        recurrent_histories,
                        conv_histories,
                    ) = self.model(
                        input_ids[:num_tokens],
                        positions[:num_tokens],
                        cu_q,
                        recurrent_states,
                        conv_states,
                        state_checkpoint_indices=(
                            rollback_checkpoint_indices
                        ),
                    )
                else:
                    (
                        hidden_states,
                        new_recurrent_states,
                        new_conv_states,
                    ) = self.model(
                        input_ids[:num_tokens],
                        positions[:num_tokens],
                        cu_q,
                        recurrent_states,
                        conv_states,
                    )
                outputs[:num_tokens] = hidden_states
                self.state_manager.scatter(
                    slots,
                    new_recurrent_states,
                    new_conv_states,
                    linear_layer_indices=linear_indices,
                )
                if history_steps:
                    for compact_index, layer_index in enumerate(
                        linear_indices
                    ):
                        recurrent_history[
                            compact_index, :num_history_tokens
                        ] = recurrent_histories[layer_index]
                        conv_history[
                            compact_index, :num_history_tokens
                        ] = conv_histories[layer_index]

            step()
            self.state_manager.reset_scratch()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, self.graph_pool):
                step()
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.verify_graphs[batch_size] = graph
            torch.cuda.synchronize()
            self.state_manager.reset_scratch()
            reset_context()
        block_tables.fill_(-1)
