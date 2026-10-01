from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from safetensors.torch import save_file

from nanovllm import SamplingParams
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.layers.gated_delta_net import GatedDeltaNet
from nanovllm.layers.layernorm import Qwen3_5RMSNorm, Qwen3_5RMSNormGated
from nanovllm.models.registry import create_model, normalize_hf_config
from nanovllm.models.qwen3_5_mtp import Qwen3_5MTP
from nanovllm.utils.context import reset_context


def _full_config():
    text = SimpleNamespace(
        model_type="qwen3_5_text",
        vocab_size=64,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        hidden_act="silu",
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        attention_bias=False,
        tie_word_embeddings=False,
        dtype=torch.bfloat16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        layer_types=[
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ],
        mtp_num_hidden_layers=1,
        rope_parameters={
            "partial_rotary_factor": 0.25,
            "rope_theta": 10_000,
            "mrope_interleaved": True,
            "mrope_section": [2, 2, 4],
        },
    )
    full = SimpleNamespace(
        model_type="qwen3_5",
        architectures=["Qwen3_5ForConditionalGeneration"],
        text_config=text,
    )
    text, capabilities = normalize_hf_config(full)
    return text, capabilities


def _write_tiny_checkpoint(
    path: Path,
    text_config,
    capabilities,
    *,
    include_mtp: bool = False,
):
    dist.init_process_group(
        "gloo",
        init_method="tcp://127.0.0.1:29639",
        rank=0,
        world_size=1,
    )
    try:
        model = create_model(text_config, capabilities).float().eval()
        torch.manual_seed(321)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.normal_(0.0, 0.02)
            for module in model.modules():
                if isinstance(module, Qwen3_5RMSNorm):
                    module.weight.zero_()
                elif isinstance(module, Qwen3_5RMSNormGated):
                    module.weight.fill_(1.0)
                elif isinstance(module, GatedDeltaNet):
                    module.A_log.zero_()
                    module.dt_bias.fill_(1.0)
        tensors = {}
        for name, tensor in model.state_dict().items():
            tensor = tensor.detach().to(text_config.dtype).cpu().contiguous()
            if name.endswith("gate_up_proj.weight"):
                gate, up = tensor.chunk(2, dim=0)
                tensors[name.replace("gate_up_proj", "gate_proj")] = gate.contiguous()
                tensors[name.replace("gate_up_proj", "up_proj")] = up.contiguous()
            else:
                tensors[name] = tensor
        if include_mtp:
            mtp = Qwen3_5MTP(text_config, model.model.embed_tokens).float()
            with torch.no_grad():
                for parameter in mtp.parameters():
                    parameter.normal_(0.0, 0.02)
            for name, tensor in mtp.state_dict().items():
                tensor = tensor.detach().to(text_config.dtype).cpu().contiguous()
                if name.endswith("gate_up_proj.weight"):
                    gate, up = tensor.chunk(2, dim=0)
                    tensors[
                        "mtp." + name.replace("gate_up_proj", "gate_proj")
                    ] = gate.contiguous()
                    tensors[
                        "mtp." + name.replace("gate_up_proj", "up_proj")
                    ] = up.contiguous()
                else:
                    tensors["mtp." + name] = tensor
        # A_log is defined and loaded in FP32 by the real model.
        for name in list(tensors):
            if name.endswith("A_log"):
                tensors[name] = tensors[name].float()
        save_file(tensors, path / "model.safetensors")
    finally:
        dist.destroy_process_group()


def test_model_runner_tiny_hybrid_prefill_and_two_decode_steps(tmp_path):
    if not torch.cuda.is_available():
        return
    text_config, capabilities = _full_config()
    _write_tiny_checkpoint(tmp_path, text_config, capabilities)
    config = SimpleNamespace(
        model=str(tmp_path),
        hf_config=text_config,
        model_capabilities=capabilities,
        kvcache_block_size=256,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_num_batched_tokens=8,
        max_model_len=16,
        max_num_seqs=1,
        max_num_state_slots=1,
        gpu_memory_utilization=0.9,
        max_num_kvcache_blocks=2,
    )

    runner = None
    try:
        runner = ModelRunner(config, rank=0, event=[])
        runner.reset_metrics()
        sequence = Sequence(
            [1, 2, 3],
            SamplingParams(temperature=0.1, max_tokens=3, ignore_eos=True),
        )
        sequence.block_table = [0]
        sequence.num_scheduled_tokens = 3
        runner.state_manager.allocate(sequence)

        [first_token] = runner.run([sequence], is_prefill=True)
        sequence.num_cached_tokens = 3
        sequence.num_scheduled_tokens = 0
        sequence.append_token(first_token)
        prefill_state = runner.state_manager.recurrent_states.clone()

        sequence.num_scheduled_tokens = 1
        [second_token] = runner.run([sequence], is_prefill=False)
        sequence.num_cached_tokens += 1
        sequence.num_scheduled_tokens = 0
        sequence.append_token(second_token)

        sequence.num_scheduled_tokens = 1
        [third_token] = runner.run([sequence], is_prefill=False)
        assert 0 <= first_token < text_config.vocab_size
        assert 0 <= second_token < text_config.vocab_size
        assert 0 <= third_token < text_config.vocab_size
        assert not torch.equal(
            runner.state_manager.recurrent_states,
            prefill_state,
        )
        metrics = runner.get_metrics()
        assert metrics["prefill_model_runs"] == 1
        assert metrics["decode_eager_runs"] == 2
        assert metrics["decode_cudagraph_replays"] == 0
    finally:
        if runner is not None:
            runner.exit()
        elif dist.is_initialized():
            dist.destroy_process_group()


def test_model_runner_batched_state_slots_survive_staggered_reuse(tmp_path):
    if not torch.cuda.is_available():
        return
    text_config, capabilities = _full_config()
    _write_tiny_checkpoint(tmp_path, text_config, capabilities)
    config = SimpleNamespace(
        model=str(tmp_path),
        hf_config=text_config,
        model_capabilities=capabilities,
        kvcache_block_size=256,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_num_batched_tokens=8,
        max_model_len=16,
        max_num_seqs=2,
        max_num_state_slots=2,
        gpu_memory_utilization=0.9,
        max_num_kvcache_blocks=3,
    )

    runner = None
    try:
        runner = ModelRunner(config, rank=0, event=[])
        first = Sequence(
            [1, 2],
            SamplingParams(temperature=0.1, max_tokens=2, ignore_eos=True),
        )
        second = Sequence(
            [3, 4, 5],
            SamplingParams(temperature=0.1, max_tokens=3, ignore_eos=True),
        )
        first.block_table = [0]
        second.block_table = [1]
        first.num_scheduled_tokens = 2
        second.num_scheduled_tokens = 3
        first_slot = runner.state_manager.allocate(first)
        second_slot = runner.state_manager.allocate(second)

        first_token, second_token = runner.run([first, second], is_prefill=True)
        for sequence, token, prompt_len in (
            (first, first_token, 2),
            (second, second_token, 3),
        ):
            sequence.num_cached_tokens = prompt_len
            sequence.num_scheduled_tokens = 0
            sequence.append_token(token)
            sequence.num_scheduled_tokens = 1

        runner.run([first, second], is_prefill=False)
        assert first_slot != second_slot
        assert runner.state_manager.recurrent_states[:, first_slot].abs().sum() > 0
        assert runner.state_manager.recurrent_states[:, second_slot].abs().sum() > 0

        second_state_before = runner.state_manager.recurrent_states[
            :, second_slot
        ].clone()
        runner.state_manager.free(first)
        replacement = Sequence(
            [6, 7],
            SamplingParams(temperature=0.1, max_tokens=1, ignore_eos=True),
        )
        replacement.block_table = [2]
        replacement.num_scheduled_tokens = 2
        replacement_slot = runner.state_manager.allocate(replacement)
        assert replacement_slot == first_slot
        assert not runner.state_manager.recurrent_states[:, replacement_slot].any()
        assert not runner.state_manager.conv_states[:, replacement_slot].any()

        runner.run([replacement], is_prefill=True)
        torch.testing.assert_close(
            runner.state_manager.recurrent_states[:, second_slot],
            second_state_before,
            rtol=0,
            atol=0,
        )
    finally:
        if runner is not None:
            runner.exit()
        elif dist.is_initialized():
            dist.destroy_process_group()


def test_model_runner_hybrid_cuda_graph_matches_eager_decode(tmp_path):
    if not torch.cuda.is_available():
        return
    text_config, capabilities = _full_config()
    _write_tiny_checkpoint(tmp_path, text_config, capabilities)
    config = SimpleNamespace(
        model=str(tmp_path),
        hf_config=text_config,
        model_capabilities=capabilities,
        kvcache_block_size=256,
        enforce_eager=False,
        tensor_parallel_size=1,
        max_num_batched_tokens=8,
        max_model_len=16,
        max_num_seqs=1,
        max_num_state_slots=1,
        gpu_memory_utilization=0.9,
        max_num_kvcache_blocks=2,
    )

    runner = None
    try:
        runner = ModelRunner(config, rank=0, event=[])
        sequence = Sequence(
            [1, 2, 3],
            SamplingParams(temperature=0.1, max_tokens=2, ignore_eos=True),
        )
        sequence.block_table = [0]
        sequence.num_scheduled_tokens = 3
        runner.state_manager.allocate(sequence)
        [sampled] = runner.run([sequence], is_prefill=True)
        sequence.num_cached_tokens = 3
        sequence.num_scheduled_tokens = 0
        sequence.append_token(sampled)
        sequence.num_scheduled_tokens = 1

        state_before = runner.state_manager.recurrent_states.clone()
        conv_before = runner.state_manager.conv_states.clone()
        kv_before = runner.kv_cache.clone()

        input_ids, positions = runner.prepare_decode([sequence])
        runner.enforce_eager = True
        eager_logits = runner.run_model(input_ids, positions, is_prefill=False)
        eager_state = runner.state_manager.recurrent_states.clone()
        eager_conv = runner.state_manager.conv_states.clone()

        runner.state_manager.recurrent_states.copy_(state_before)
        runner.state_manager.conv_states.copy_(conv_before)
        runner.kv_cache.copy_(kv_before)
        input_ids, positions = runner.prepare_decode([sequence])
        runner.enforce_eager = False
        graph_logits = runner.run_model(input_ids, positions, is_prefill=False)

        torch.testing.assert_close(graph_logits, eager_logits, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(
            runner.state_manager.recurrent_states,
            eager_state,
            rtol=2e-2,
            atol=2e-2,
        )
        torch.testing.assert_close(
            runner.state_manager.conv_states.float(),
            eager_conv.float(),
            rtol=2e-2,
            atol=2e-2,
        )
        assert torch.equal(
            graph_logits.argmax(dim=-1),
            eager_logits.argmax(dim=-1),
        )
    finally:
        reset_context()
        if runner is not None:
            runner.exit()
        elif dist.is_initialized():
            dist.destroy_process_group()


def _generate_tiny_greedy(
    config,
    *,
    num_tokens: int,
    zero_mtp: bool = False,
    prefill_chunks: tuple[int, ...] | None = None,
    prompt: tuple[int, ...] = (1, 2, 3),
):
    runner = ModelRunner(config, rank=0, event=[])
    try:
        if zero_mtp:
            with torch.no_grad():
                for parameter in runner.mtp.parameters():
                    parameter.zero_()
        sequence = Sequence(
            list(prompt),
            SamplingParams(temperature=0.0, max_tokens=32, ignore_eos=True),
        )
        sequence.block_table = [0]
        runner.allocate_state_slot(sequence)
        first = None
        if prefill_chunks is None:
            prefill_chunks = (len(prompt),)
        for chunk_size in prefill_chunks:
            sequence.num_scheduled_tokens = chunk_size
            [candidate] = runner.run([sequence], is_prefill=True)
            sequence.num_cached_tokens += chunk_size
            sequence.num_scheduled_tokens = 0
            if sequence.num_cached_tokens == sequence.num_prompt_tokens:
                first = candidate
        assert sequence.num_cached_tokens == sequence.num_prompt_tokens
        assert first is not None
        generated = [first]
        sequence.append_token(first)
        rounds = []
        while len(generated) < num_tokens:
            sequence.num_scheduled_tokens = 1
            if config.num_speculative_tokens:
                tokens = runner.run_speculative([sequence])[0]
                rounds.append(tokens)
                sequence.num_cached_tokens += len(tokens)
                sequence.num_scheduled_tokens = 0
                for token in tokens:
                    sequence.append_token(token)
                sequence.num_cached_tokens = len(sequence) - 1
                generated.extend(tokens)
            else:
                [token] = runner.run([sequence], is_prefill=False)
                sequence.num_cached_tokens += 1
                sequence.num_scheduled_tokens = 0
                sequence.append_token(token)
                generated.append(token)
        return generated[:num_tokens], rounds
    finally:
        runner.exit()


def test_model_runner_mtp_matches_target_with_acceptance_and_rejection(tmp_path):
    if not torch.cuda.is_available():
        return
    text_config, capabilities = _full_config()
    _write_tiny_checkpoint(
        tmp_path,
        text_config,
        capabilities,
        include_mtp=True,
    )

    def make_config(
        num_speculative_tokens,
        parallel_verify=False,
        cuda_graph=False,
    ):
        return SimpleNamespace(
            model=str(tmp_path),
            hf_config=text_config,
            model_capabilities=capabilities,
            kvcache_block_size=256,
            enforce_eager=not cuda_graph,
            tensor_parallel_size=1,
            max_num_batched_tokens=8,
            max_model_len=32,
            max_num_seqs=1,
            max_num_state_slots=1,
            gpu_memory_utilization=0.9,
            max_num_kvcache_blocks=2,
            num_speculative_tokens=num_speculative_tokens,
            speculative_parallel_verify=parallel_verify,
        )

    expected, _ = _generate_tiny_greedy(
        make_config(0), num_tokens=8
    )
    speculative, normal_rounds = _generate_tiny_greedy(
        make_config(2), num_tokens=8
    )
    rejected, rejection_rounds = _generate_tiny_greedy(
        make_config(2), num_tokens=8, zero_mtp=True
    )
    chunked, _ = _generate_tiny_greedy(
        make_config(2), num_tokens=8, prefill_chunks=(1, 2)
    )
    parallel, _ = _generate_tiny_greedy(
        make_config(2, parallel_verify=True), num_tokens=8
    )
    graphed, _ = _generate_tiny_greedy(
        make_config(2, parallel_verify=True, cuda_graph=True),
        num_tokens=8,
    )

    assert speculative == expected
    assert rejected == expected
    assert chunked == expected
    assert parallel == expected
    assert graphed == expected
    assert normal_rounds
    assert rejection_rounds
    assert all(len(tokens) >= 2 for tokens in rejection_rounds)


def test_model_runner_mtp_continuous_batch_commits_variable_acceptance(tmp_path):
    if not torch.cuda.is_available():
        return
    text_config, capabilities = _full_config()
    _write_tiny_checkpoint(
        tmp_path,
        text_config,
        capabilities,
        include_mtp=True,
    )

    def make_config(num_speculative_tokens, max_num_seqs=1):
        return SimpleNamespace(
            model=str(tmp_path),
            hf_config=text_config,
            model_capabilities=capabilities,
            kvcache_block_size=256,
            enforce_eager=True,
            tensor_parallel_size=1,
            max_num_batched_tokens=8,
            max_model_len=32,
            max_num_seqs=max_num_seqs,
            max_num_state_slots=max_num_seqs,
            gpu_memory_utilization=0.9,
            max_num_kvcache_blocks=4,
            num_speculative_tokens=num_speculative_tokens,
            speculative_parallel_verify=True,
        )

    expected_a, _ = _generate_tiny_greedy(
        make_config(0), num_tokens=10, prompt=(1, 2, 3)
    )
    expected_b, _ = _generate_tiny_greedy(
        make_config(0), num_tokens=10, prompt=(4, 5)
    )

    runner = ModelRunner(make_config(2, 2), rank=0, event=[])
    try:
        sequences = [
            Sequence(
                [1, 2, 3],
                SamplingParams(temperature=0.0, max_tokens=32, ignore_eos=True),
            ),
            Sequence(
                [4, 5],
                SamplingParams(temperature=0.0, max_tokens=32, ignore_eos=True),
            ),
        ]
        for block_id, sequence in enumerate(sequences):
            sequence.block_table = [block_id]
            sequence.num_scheduled_tokens = len(sequence)
            runner.allocate_state_slot(sequence)
        first_tokens = runner.run(sequences, is_prefill=True)
        assert first_tokens == [expected_a[0], expected_b[0]]
        for sequence, token in zip(sequences, first_tokens):
            sequence.num_cached_tokens = sequence.num_prompt_tokens
            sequence.num_scheduled_tokens = 0
            sequence.append_token(token)
            sequence.num_scheduled_tokens = 1

        original_greedy = runner._distributed_greedy_tokens
        call_index = 0

        def controlled_greedy(logits, batch_size):
            nonlocal call_index
            current = call_index
            call_index += 1
            if current == 1:
                return torch.tensor(
                    [expected_a[2], (expected_b[2] + 1) % text_config.vocab_size],
                    dtype=torch.long,
                    device="cuda",
                )
            if current == 2:
                return torch.tensor(
                    [expected_a[3], (expected_b[3] + 1) % text_config.vocab_size],
                    dtype=torch.long,
                    device="cuda",
                )
            return original_greedy(logits, batch_size)

        runner._distributed_greedy_tokens = controlled_greedy
        outputs = runner.run_speculative(sequences)
        runner._distributed_greedy_tokens = original_greedy

        assert outputs[0] == expected_a[1:4]
        assert outputs[1] == expected_b[1:3]
        for sequence, tokens in zip(sequences, outputs):
            sequence.num_cached_tokens += len(tokens)
            sequence.num_scheduled_tokens = 0
            for token in tokens:
                sequence.append_token(token)
            sequence.num_cached_tokens = len(sequence) - 1
            sequence.num_scheduled_tokens = 1

        next_outputs = runner.run_speculative(sequences)
        assert next_outputs[0] == expected_a[4 : 4 + len(next_outputs[0])]
        assert next_outputs[1] == expected_b[3 : 3 + len(next_outputs[1])]
    finally:
        runner.exit()


def test_hybrid_prefix_cache_restores_aligned_state_and_matches_cold(tmp_path):
    if not torch.cuda.is_available():
        return
    text_config, capabilities = _full_config()
    text_config.max_position_embeddings = 512
    _write_tiny_checkpoint(tmp_path, text_config, capabilities)

    def make_config(enabled, cuda_graph=False):
        return SimpleNamespace(
            model=str(tmp_path),
            hf_config=text_config,
            model_capabilities=capabilities,
            kvcache_block_size=256,
            enforce_eager=not cuda_graph,
            tensor_parallel_size=1,
            max_num_batched_tokens=300,
            max_model_len=512,
            max_num_seqs=1,
            max_num_state_slots=1,
            gpu_memory_utilization=0.9,
            max_num_kvcache_blocks=4,
            num_speculative_tokens=0,
            speculative_parallel_verify=False,
            enable_prefix_cache=enabled,
            enable_hybrid_prefix_cache=enabled,
            hybrid_prefix_checkpoint_interval_blocks=1,
            hybrid_prefix_checkpoint_memory_bytes=5000,
            scheduling_policy="prefill_first",
            eos=-1,
            enable_chunked_prefill=True,
            request_metrics_history_size=16,
        )

    def run_request(runner, scheduler, token_ids):
        sequence = Sequence(
            token_ids,
            SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True),
        )
        scheduler.add(sequence)
        while not sequence.is_finished:
            for batch in scheduler.schedule():
                sampled = runner.run(batch.seqs, batch.is_prefill)
                pending, hashed = scheduler.prepare_prefix_captures(
                    batch.seqs, batch.is_prefill
                )
                for capture in pending:
                    if scheduler.reserve_prefix_capture(capture):
                        checkpoint_slot = runner.capture_prefix_checkpoint(
                            capture.state_slot
                        )
                        scheduler.publish_prefix_capture(
                            capture, checkpoint_slot
                        )
                scheduler.postprocess(
                    batch.seqs,
                    sampled,
                    batch.is_prefill,
                    blocks_already_hashed=hashed,
                )
        return sequence.completion_token_ids

    shared = [(index * 7 + 3) % text_config.vocab_size for index in range(256)]
    source = shared + [
        (index * 5 + 1) % text_config.vocab_size for index in range(44)
    ]
    target = shared + [
        (index * 11 + 2) % text_config.vocab_size for index in range(44)
    ]

    cold_runner = None
    try:
        cold_config = make_config(False)
        cold_runner = ModelRunner(cold_config, rank=0, event=[])
        cold_scheduler = Scheduler(
            cold_config,
            cold_runner.state_manager,
            cold_runner,
            cold_runner.prefix_checkpoint_pool,
        )
        expected = run_request(cold_runner, cold_scheduler, target)
    finally:
        if cold_runner is not None:
            cold_runner.exit()

    warm_runner = None
    try:
        warm_config = make_config(True)
        warm_runner = ModelRunner(warm_config, rank=0, event=[])
        warm_scheduler = Scheduler(
            warm_config,
            warm_runner.state_manager,
            warm_runner,
            warm_runner.prefix_checkpoint_pool,
        )
        run_request(warm_runner, warm_scheduler, source)
        assert warm_scheduler.get_metrics()["hybrid_prefix_cache_entries"] == 1

        preempted = Sequence(
            target,
            SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True),
        )
        warm_scheduler.add(preempted)
        scheduled = warm_scheduler.schedule()[0]
        assert scheduled.is_prefill
        assert preempted.num_cached_tokens == 256
        first_restored_slot = preempted.state_slot
        warm_scheduler.running.remove(preempted)
        warm_scheduler.preempt(preempted)
        assert preempted.state_slot is None
        assert not preempted.block_table
        assert first_restored_slot in warm_runner.state_manager.free_slot_ids
        assert warm_scheduler.get_metrics()["hybrid_prefix_cache_entries"] == 1

        scheduled_again = warm_scheduler.schedule()[0]
        assert scheduled_again.is_prefill
        assert preempted.num_cached_tokens == 256
        assert preempted.state_slot is not None
        assert warm_scheduler.abort(preempted.seq_id)
        assert preempted.state_slot is None
        assert warm_scheduler.get_metrics()["hybrid_prefix_cache_entries"] == 1

        warm_scheduler.reset_metrics()
        actual = run_request(warm_runner, warm_scheduler, target)
        metrics = warm_scheduler.get_metrics()
        assert actual == expected
        assert metrics["hybrid_prefix_committed_hit_tokens"] == 256
        assert metrics["prefix_cache_hit_blocks"] == 1
        assert metrics["hybrid_prefix_restore_count"] == 1

        other_shared = [
            (index * 29 + 17) % text_config.vocab_size
            for index in range(256)
        ]
        other_source = other_shared + [
            (index * 3 + 5) % text_config.vocab_size
            for index in range(44)
        ]
        run_request(warm_runner, warm_scheduler, other_source)
        assert warm_scheduler.get_metrics()["hybrid_prefix_eviction_count"] >= 1

        warm_scheduler.reset_metrics()
        after_eviction = run_request(warm_runner, warm_scheduler, target)
        fallback_metrics = warm_scheduler.get_metrics()
        assert after_eviction == expected
        assert fallback_metrics["hybrid_prefix_kv_candidate_tokens"] == 256
        assert fallback_metrics["hybrid_prefix_committed_hit_tokens"] == 0
        assert fallback_metrics["hybrid_prefix_fallback_count"] == 1
    finally:
        if warm_runner is not None:
            warm_runner.exit()

    graph_runner = None
    try:
        graph_config = make_config(True, cuda_graph=True)
        graph_runner = ModelRunner(graph_config, rank=0, event=[])
        graph_scheduler = Scheduler(
            graph_config,
            graph_runner.state_manager,
            graph_runner,
            graph_runner.prefix_checkpoint_pool,
        )
        run_request(graph_runner, graph_scheduler, source)
        graph_scheduler.reset_metrics()
        graph_runner.reset_metrics()
        graph_output = run_request(graph_runner, graph_scheduler, target)
        assert graph_output == expected
        assert (
            graph_scheduler.get_metrics()[
                "hybrid_prefix_committed_hit_tokens"
            ]
            == 256
        )
        assert graph_runner.get_metrics()["decode_cudagraph_replays"] > 0
    finally:
        if graph_runner is not None:
            graph_runner.exit()
