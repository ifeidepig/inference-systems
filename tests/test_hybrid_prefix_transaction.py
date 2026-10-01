import os
import tempfile
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from nanovllm.engine.state_manager import (
    HybridPrefixCheckpointPool,
    HybridStateManager,
    transactional_capture_prefix_checkpoint,
    transactional_restore_prefix_checkpoint,
)
from nanovllm.engine.model_runner import ModelRunner


def _transaction_worker(rank: int, world_size: int, init_file: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        manager = HybridStateManager(
            max_num_seqs=1,
            num_linear_layers=2,
            num_value_heads=1,
            key_head_dim=2,
            value_head_dim=3,
            conv_dim=4,
            conv_kernel_size=4,
            conv_dtype=torch.float32,
            device="cpu",
        )
        request = SimpleNamespace(state_slot=None)
        request_slot = manager.allocate(request)
        manager.recurrent_states[:, request_slot].fill_(rank + 1)
        manager.conv_states[:, request_slot].fill_(rank + 2)
        pool = HybridPrefixCheckpointPool(
            manager,
            memory_budget_bytes=manager.bytes_per_slot() * 2,
        )

        # Rank 1 fails prepare. Rank 0 must release its successful local slot.
        failing_request_slot = request_slot if rank == 0 else 99
        if rank == 0:
            try:
                transactional_capture_prefix_checkpoint(
                    pool,
                    failing_request_slot,
                )
            except RuntimeError as exc:
                assert "rolled back" in str(exc)
            else:
                raise AssertionError("capture failure did not reach rank 0")
        else:
            assert transactional_capture_prefix_checkpoint(
                pool,
                failing_request_slot,
            ) == -1
        assert not pool.used_checkpoint_slots
        assert list(pool.free_checkpoint_slots) == [0, 1]
        assert not pool.recurrent_checkpoints.any()
        assert not pool.conv_checkpoints.any()

        # Detect and repair pre-existing free-queue order divergence.
        if rank == 1:
            pool.free_checkpoint_slots.rotate(-1)
        if rank == 0:
            try:
                transactional_capture_prefix_checkpoint(pool, request_slot)
            except RuntimeError as exc:
                assert "diverged" in str(exc)
            else:
                raise AssertionError("slot divergence did not reach rank 0")
        else:
            assert transactional_capture_prefix_checkpoint(
                pool,
                request_slot,
            ) == -1
        assert list(pool.free_checkpoint_slots) == [0, 1]

        # A successful all-rank capture now publishes one deterministic slot.
        checkpoint_slot = transactional_capture_prefix_checkpoint(
            pool,
            request_slot,
        )
        assert checkpoint_slot == 0
        assert pool.used_checkpoint_slots == {0}

        # Rank 1 fails restore. Rank 0 may have copied its shard, but both
        # ranks must restore the private request state that existed beforehand.
        manager.recurrent_states[:, request_slot].fill_(10 + rank)
        manager.conv_states[:, request_slot].fill_(20 + rank)
        recurrent_before = manager.recurrent_states[:, request_slot].clone()
        conv_before = manager.conv_states[:, request_slot].clone()
        failing_checkpoint_slot = checkpoint_slot if rank == 0 else 99
        if rank == 0:
            try:
                transactional_restore_prefix_checkpoint(
                    pool,
                    failing_checkpoint_slot,
                    request_slot,
                )
            except RuntimeError as exc:
                assert "rolled back" in str(exc)
            else:
                raise AssertionError("restore failure did not reach rank 0")
        else:
            transactional_restore_prefix_checkpoint(
                pool,
                failing_checkpoint_slot,
                request_slot,
            )
        torch.testing.assert_close(
            manager.recurrent_states[:, request_slot],
            recurrent_before,
        )
        torch.testing.assert_close(
            manager.conv_states[:, request_slot],
            conv_before,
        )

        transactional_restore_prefix_checkpoint(
            pool,
            checkpoint_slot,
            request_slot,
        )
        torch.testing.assert_close(
            manager.recurrent_states[:, request_slot],
            torch.full_like(
                manager.recurrent_states[:, request_slot],
                rank + 1,
            ),
        )
        torch.testing.assert_close(
            manager.conv_states[:, request_slot],
            torch.full_like(
                manager.conv_states[:, request_slot],
                rank + 2,
            ),
        )

        runner = SimpleNamespace(
            kv_cache=torch.zeros(2, 1, 2, 4, 1, 1),
            block_size=4,
            rank=rank,
            hybrid_prefix_cow_count=0,
            hybrid_prefix_cow_bytes=0,
            hybrid_prefix_cow_ms=0.0,
        )
        runner.kv_cache[:, :, 0].fill_(rank + 3)
        failing_source = 0 if rank == 0 else 99
        if rank == 0:
            try:
                ModelRunner.copy_prefix_kv(runner, failing_source, 1, 3)
            except RuntimeError as exc:
                assert "rolled back" in str(exc)
            else:
                raise AssertionError("COW failure did not reach rank 0")
        else:
            ModelRunner.copy_prefix_kv(runner, failing_source, 1, 3)

        # The worker remains live after the failed transaction.
        runner.kv_cache[:, :, 1].zero_()
        ModelRunner.copy_prefix_kv(runner, 0, 1, 3)
        torch.testing.assert_close(
            runner.kv_cache[:, :, 1, :3],
            runner.kv_cache[:, :, 0, :3],
        )
    finally:
        dist.destroy_process_group()


def test_all_rank_checkpoint_capture_and_restore_transactions():
    handle, init_file = tempfile.mkstemp(prefix="nanovllm-prefix-tx-")
    os.close(handle)
    try:
        mp.spawn(
            _transaction_worker,
            args=(2, init_file),
            nprocs=2,
            join=True,
        )
    finally:
        if os.path.exists(init_file):
            os.unlink(init_file)


if __name__ == "__main__":
    test_all_rank_checkpoint_capture_and_restore_transactions()
