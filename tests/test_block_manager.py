import unittest

from nanovllm import SamplingParams
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence


class BlockManagerMetricsTest(unittest.TestCase):

    def setUp(self):
        Sequence.block_size = 2
        self.sampling = SamplingParams(
            temperature=1.0,
            max_tokens=1,
            ignore_eos=True,
        )

    def test_cached_block_reassignment_counts_reachable_eviction(self):
        manager = BlockManager(num_blocks=1, block_size=2)
        block = manager.blocks[0]
        token_ids = [10, 20]
        block_hash = manager.compute_hash(token_ids)
        block.update(block_hash, token_ids)
        manager.hash_to_block_id[block_hash] = block.block_id

        self.assertEqual(manager._allocate_block(), 0)

        self.assertEqual(manager.cached_block_reassignments, 1)
        self.assertEqual(manager.prefix_cache_evictions, 1)
        self.assertEqual(manager.num_cached_blocks, 0)
        self.assertEqual(manager.num_reachable_cached_blocks, 0)

    def test_duplicate_registration_tracks_physical_and_reachable_blocks(self):
        manager = BlockManager(num_blocks=4, block_size=2)
        first = Sequence([1, 2, 90], self.sampling)
        second = Sequence([1, 2, 91], self.sampling)
        manager.allocate(first, num_cached_blocks=0)
        manager.allocate(second, num_cached_blocks=0)

        first.num_scheduled_tokens = 2
        second.num_scheduled_tokens = 2
        manager.hash_blocks(first)
        manager.hash_blocks(second)

        self.assertEqual(manager.duplicate_cache_registrations, 1)
        self.assertEqual(manager.num_cached_blocks, 2)
        self.assertEqual(manager.num_reachable_cached_blocks, 1)
        self.assertEqual(
            manager.hash_to_block_id[manager.compute_hash([1, 2])],
            second.block_table[0],
        )

    def test_reset_metrics_preserves_cache_state(self):
        manager = BlockManager(num_blocks=1, block_size=2)
        block = manager.blocks[0]
        token_ids = [3, 4]
        block_hash = manager.compute_hash(token_ids)
        block.update(block_hash, token_ids)
        manager.hash_to_block_id[block_hash] = block.block_id
        manager.duplicate_cache_registrations = 3

        manager.reset_metrics()

        self.assertEqual(manager.duplicate_cache_registrations, 0)
        self.assertEqual(manager.num_cached_blocks, 1)
        self.assertEqual(manager.num_reachable_cached_blocks, 1)


if __name__ == "__main__":
    unittest.main()
