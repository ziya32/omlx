# SPDX-License-Identifier: Apache-2.0
"""
Tests for BlockAwarePrefixCache and related components.

This module tests the block-aware prefix caching system that uses
PagedCacheManager for block-based storage with SSD persistence.
"""

import time
from unittest.mock import MagicMock

import pytest

from omlx.cache.paged_cache import (
    BlockTable,
    PagedCacheManager,
    compute_block_hash,
)
from omlx.cache.prefix_cache import BlockAwarePrefixCache, BlockCacheEntry
from omlx.cache.stats import PrefixCacheStats


class MockModel:
    """Mock model for testing."""

    def __init__(self, num_layers: int = 32):
        self._num_layers = num_layers
        self.layers = [MagicMock() for _ in range(num_layers)]

    @property
    def args(self):
        mock_args = MagicMock()
        mock_args.num_hidden_layers = self._num_layers
        return mock_args


class TestBlockCacheEntry:
    """Tests for BlockCacheEntry dataclass."""

    def test_creation(self):
        """Test creating a cache entry."""
        block_table = BlockTable(request_id="req-001")
        entry = BlockCacheEntry(
            block_table=block_table,
            last_access=time.time(),
        )

        assert entry.block_table is block_table
        assert entry.last_access > 0


class TestBlockAwarePrefixCache:
    """Tests for BlockAwarePrefixCache."""

    @pytest.fixture
    def paged_cache(self):
        """Create a PagedCacheManager for testing."""
        return PagedCacheManager(
            block_size=4,  # Small block size for testing
            max_blocks=100,
            model_name="test-model",
            initial_blocks=100,
        )

    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        return MockModel(num_layers=4)

    @pytest.fixture
    def prefix_cache(self, mock_model, paged_cache):
        """Create a BlockAwarePrefixCache for testing."""
        return BlockAwarePrefixCache(
            model=mock_model,
            paged_cache_manager=paged_cache,
            paged_ssd_cache_manager=None,
        )

    def test_initialization(self, prefix_cache, mock_model, paged_cache):
        """Test cache initialization."""
        assert prefix_cache.model is mock_model
        assert prefix_cache.paged_cache is paged_cache
        assert prefix_cache.block_size == 4
        assert prefix_cache.expected_num_layers == 4

    def test_get_model_num_layers(self, paged_cache):
        """Test getting model num_layers from various sources."""
        # Model with layers attribute
        model1 = MockModel(num_layers=16)
        cache1 = BlockAwarePrefixCache(model1, paged_cache)
        assert cache1.expected_num_layers == 16

        # Model with args.num_hidden_layers
        model2 = MagicMock()
        model2.layers = None
        model2.args.num_hidden_layers = 24
        delattr(model2, 'layers')
        cache2 = BlockAwarePrefixCache(model2, paged_cache)
        assert cache2.expected_num_layers == 24

    def test_get_model_num_layers_with_make_cache(self, paged_cache):
        """Test make_cache() takes priority over model.layers for hybrid models."""
        # Hybrid model: 88 architectural layers but only 48 produce cache
        model = MockModel(num_layers=88)
        model.make_cache = lambda: [MagicMock() for _ in range(48)]
        cache = BlockAwarePrefixCache(model, paged_cache)
        assert cache.expected_num_layers == 48

        # Non-hybrid model: make_cache() returns same count as model.layers
        model2 = MockModel(num_layers=32)
        model2.make_cache = lambda: [MagicMock() for _ in range(32)]
        cache2 = BlockAwarePrefixCache(model2, paged_cache)
        assert cache2.expected_num_layers == 32

    def test_get_model_num_layers_make_cache_exception(self, paged_cache):
        """Test fallback to model.layers when make_cache() raises."""
        model = MockModel(num_layers=88)

        def bad_make_cache():
            raise RuntimeError("model not initialized")

        model.make_cache = bad_make_cache
        cache = BlockAwarePrefixCache(model, paged_cache)
        assert cache.expected_num_layers == 88

    def test_get_model_num_layers_make_cache_empty(self, paged_cache):
        """Test fallback when make_cache() returns empty list."""
        model = MockModel(num_layers=32)
        model.make_cache = lambda: []
        cache = BlockAwarePrefixCache(model, paged_cache)
        assert cache.expected_num_layers == 32

    def test_get_model_num_layers_make_cache_non_list(self, paged_cache):
        """Test fallback when make_cache() returns non-list."""
        model = MockModel(num_layers=32)
        model.make_cache = lambda: None
        cache = BlockAwarePrefixCache(model, paged_cache)
        assert cache.expected_num_layers == 32

    def test_fetch_cache_miss(self, prefix_cache):
        """Test fetch_cache returns miss when no cache exists."""
        tokens = [1, 2, 3, 4, 5, 6, 7, 8]

        block_table, remaining = prefix_cache.fetch_cache("req-001", tokens)

        assert block_table is None
        assert remaining == tokens
        assert prefix_cache._misses == 1

    def test_fetch_cache_empty_tokens(self, prefix_cache):
        """Test fetch_cache with empty tokens."""
        block_table, remaining = prefix_cache.fetch_cache("req-001", [])

        assert block_table is None
        assert remaining == []

    def test_store_cache_empty_tokens(self, prefix_cache):
        """Test store_cache with empty tokens returns None."""
        result = prefix_cache.store_cache("req-001", [], [])
        assert result is None

    def test_store_cache_creates_block_table(self, prefix_cache):
        """Test store_cache creates a block table."""
        tokens = [1, 2, 3, 4]  # One full block
        # Mock cache data (tensor data format)
        cache_data = [
            {"state": (MagicMock(), MagicMock()), "cache_type": "KVCache"}
            for _ in range(4)
        ]

        result = prefix_cache.store_cache("req-001", tokens, cache_data)

        assert result is not None
        assert result.request_id == "req-001"
        assert "req-001" in prefix_cache._request_tables

    def test_release_cache(self, prefix_cache, paged_cache):
        """Test releasing cache for a request."""
        # First create a block table with blocks
        table = paged_cache.create_block_table("req-001")
        block = paged_cache.allocate_block()
        table.block_ids.append(block.block_id)

        # Add to prefix cache tracking
        prefix_cache._request_tables["req-001"] = BlockCacheEntry(
            block_table=table,
            last_access=time.time(),
        )

        initial_free = paged_cache.free_blocks

        prefix_cache.release_cache("req-001")

        assert "req-001" not in prefix_cache._request_tables
        assert paged_cache.free_blocks == initial_free + 1

    def test_clear_request_entry(self, prefix_cache, paged_cache):
        """Test clearing request entry without freeing blocks."""
        table = paged_cache.create_block_table("req-001")
        prefix_cache._request_tables["req-001"] = BlockCacheEntry(
            block_table=table,
            last_access=time.time(),
        )

        prefix_cache.clear_request_entry("req-001")

        assert "req-001" not in prefix_cache._request_tables
        # Blocks should still be tracked in paged_cache
        assert "req-001" in paged_cache.request_tables

    def test_fork_cache(self, prefix_cache, paged_cache):
        """Test forking cache from one request to another."""
        # Create source with blocks
        source_table = paged_cache.create_block_table("req-source")
        block = paged_cache.allocate_block()
        source_table.block_ids.append(block.block_id)

        prefix_cache._request_tables["req-source"] = BlockCacheEntry(
            block_table=source_table,
            last_access=time.time(),
        )

        forked = prefix_cache.fork_cache("req-source", "req-forked")

        assert forked is not None
        assert forked.request_id == "req-forked"
        assert "req-forked" in prefix_cache._request_tables
        # Ref count should be incremented
        assert block.ref_count == 2

    def test_fork_cache_nonexistent(self, prefix_cache):
        """Test forking nonexistent cache returns None."""
        result = prefix_cache.fork_cache("nonexistent", "new-req")
        assert result is None

    def test_get_stats(self, prefix_cache):
        """Test getting statistics."""
        prefix_cache._hits = 5
        prefix_cache._misses = 3
        prefix_cache._tokens_saved = 100

        stats = prefix_cache.get_stats()

        assert isinstance(stats, PrefixCacheStats)
        assert stats.hits == 5
        assert stats.misses == 3
        assert stats.tokens_saved == 100
        assert stats.partial_block_skips == 0
        assert stats.partial_tokens_skipped == 0
        assert stats.block_size == prefix_cache.block_size

    def test_get_stats_dict(self, prefix_cache):
        """Test getting statistics as dictionary."""
        prefix_cache._hits = 10
        prefix_cache._misses = 5

        stats_dict = prefix_cache.get_stats_dict()

        assert "hits" in stats_dict
        assert "misses" in stats_dict
        assert "hit_rate" in stats_dict
        assert "partial_block_skips" in stats_dict
        assert "partial_tokens_skipped" in stats_dict
        assert "last_tokens_to_next_block" in stats_dict
        assert stats_dict["hit_rate"] == pytest.approx(10 / 15)

    def test_reset_stats(self, prefix_cache):
        """Test resetting statistics."""
        prefix_cache._hits = 10
        prefix_cache._misses = 5
        prefix_cache._tokens_saved = 500
        prefix_cache._partial_block_skips = 3
        prefix_cache._partial_tokens_skipped = 42
        prefix_cache._last_partial_tokens_skipped = 2
        prefix_cache._last_tokens_to_next_block = 254

        prefix_cache.reset_stats()

        assert prefix_cache._hits == 0
        assert prefix_cache._misses == 0
        assert prefix_cache._tokens_saved == 0
        assert prefix_cache._partial_block_skips == 0
        assert prefix_cache._partial_tokens_skipped == 0
        assert prefix_cache._last_partial_tokens_skipped == 0
        assert prefix_cache._last_tokens_to_next_block == 0

    def test_clear(self, prefix_cache, paged_cache):
        """Test clearing all cache data."""
        # Add some data
        table = paged_cache.create_block_table("req-001")
        prefix_cache._request_tables["req-001"] = BlockCacheEntry(
            block_table=table,
            last_access=time.time(),
        )
        prefix_cache._prefix_index[b"test_hash"] = (10, [1, 2], 2)
        prefix_cache._hits = 10

        cleared = prefix_cache.clear()

        assert cleared > 0
        assert len(prefix_cache._request_tables) == 0
        assert len(prefix_cache._prefix_index) == 0
        assert prefix_cache._hits == 0

    def test_len(self, prefix_cache):
        """Test __len__ returns number of request entries."""
        assert len(prefix_cache) == 0

        table = BlockTable(request_id="req-001")
        prefix_cache._request_tables["req-001"] = BlockCacheEntry(
            block_table=table,
            last_access=time.time(),
        )
        prefix_cache._request_tables["req-002"] = BlockCacheEntry(
            block_table=BlockTable(request_id="req-002"),
            last_access=time.time(),
        )

        assert len(prefix_cache) == 2

    def test_cache_manager_interface_fetch(self, prefix_cache):
        """Test CacheManager ABC fetch interface."""
        # Invalid key format
        value, hit = prefix_cache.fetch("invalid")
        assert hit is False
        assert value is None

        # Valid key format but miss
        value, hit = prefix_cache.fetch(("req-001", [1, 2, 3, 4]))
        assert hit is False

    def test_cache_manager_interface_store(self, prefix_cache):
        """Test CacheManager ABC store interface."""
        # Invalid key format
        result = prefix_cache.store("invalid", [])
        assert result is False

    def test_cache_manager_interface_evict(self, prefix_cache, paged_cache):
        """Test CacheManager ABC evict interface."""
        # Evict nonexistent
        result = prefix_cache.evict("req-nonexistent")
        assert result is False

        # Create entry then evict
        table = paged_cache.create_block_table("req-001")
        prefix_cache._request_tables["req-001"] = BlockCacheEntry(
            block_table=table,
            last_access=time.time(),
        )

        result = prefix_cache.evict("req-001")
        assert result is True
        assert "req-001" not in prefix_cache._request_tables

    def test_cache_manager_interface_properties(self, prefix_cache):
        """Test CacheManager ABC property interface."""
        assert prefix_cache.size == 0
        assert prefix_cache.max_size == 100  # max_blocks

    def test_set_paged_ssd_cache_manager(self, prefix_cache):
        """Test setting SSD cache manager."""
        mock_ssd_cache = MagicMock()

        prefix_cache.set_paged_ssd_cache_manager(mock_ssd_cache)

        assert prefix_cache.paged_ssd_cache is mock_ssd_cache

    def test_set_cold_restore_callback(self, prefix_cache):
        """Test setting cold restore callback."""

        def restore_callback(block_id: int, block_hash: bytes) -> bool:
            return True

        prefix_cache.set_cold_restore_callback(restore_callback)
        assert prefix_cache._cold_restore_callback is restore_callback


class TestBlockAwarePrefixCacheWithSSD:
    """Tests for BlockAwarePrefixCache with SSD cache manager."""

    @pytest.fixture
    def paged_cache(self):
        """Create a PagedCacheManager."""
        return PagedCacheManager(
            block_size=4,
            max_blocks=100,
            model_name="test-model",
            initial_blocks=100,
        )

    @pytest.fixture
    def mock_ssd_cache(self):
        """Create a mock SSD cache manager."""
        mock = MagicMock()
        mock.save_block.return_value = True
        mock.load_block.return_value = None
        mock.load_block_with_metadata.return_value = (None, None)
        mock.has_block.return_value = False
        return mock

    @pytest.fixture
    def prefix_cache_with_ssd(self, paged_cache, mock_ssd_cache):
        """Create a BlockAwarePrefixCache with SSD manager."""
        model = MockModel(num_layers=4)
        return BlockAwarePrefixCache(
            model=model,
            paged_cache_manager=paged_cache,
            paged_ssd_cache_manager=mock_ssd_cache,
        )

    def test_get_cache_for_generation_no_entry(self, prefix_cache_with_ssd):
        """Test get_cache_for_generation with no entry."""
        result, was_loaded = prefix_cache_with_ssd.get_cache_for_generation("req-001")

        assert result is None
        assert was_loaded is False

    def test_reconstruct_cache_empty_table(self, prefix_cache_with_ssd):
        """Test reconstruct_cache with empty block table."""
        table = BlockTable(request_id="req-001")

        result = prefix_cache_with_ssd.reconstruct_cache(table)

        assert result is None

    def test_reconstruct_cache_no_ssd_manager(self, paged_cache):
        """Test reconstruct_cache without SSD manager."""
        model = MockModel(num_layers=4)
        cache = BlockAwarePrefixCache(
            model=model,
            paged_cache_manager=paged_cache,
            paged_ssd_cache_manager=None,
        )

        table = BlockTable(request_id="req-001", block_ids=[1, 2])

        result = cache.reconstruct_cache(table)

        assert result is None


class TestPrefixIndexOperations:
    """Tests for prefix index operations."""

    @pytest.fixture
    def paged_cache(self):
        """Create a PagedCacheManager."""
        return PagedCacheManager(
            block_size=4,
            max_blocks=100,
            model_name="test-model",
            initial_blocks=100,
        )

    @pytest.fixture
    def prefix_cache(self, paged_cache):
        """Create a BlockAwarePrefixCache."""
        model = MockModel(num_layers=4)
        return BlockAwarePrefixCache(
            model=model,
            paged_cache_manager=paged_cache,
        )

    def test_update_prefix_index(self, prefix_cache, paged_cache):
        """Test _update_prefix_index method."""
        tokens = [1, 2, 3, 4, 5, 6, 7, 8]  # 2 blocks
        block_ids = [1, 2]

        # Allocate blocks and set them up
        blocks = paged_cache.get_new_blocks(2)
        block_ids = [b.block_id for b in blocks]

        prefix_cache._update_prefix_index(tokens, block_ids)

        # Prefix index should have entries
        assert len(prefix_cache._prefix_index) > 0

    def test_find_best_prefix_match_no_match(self, prefix_cache):
        """Test _find_best_prefix_match with no matching prefix."""
        tokens = [1, 2, 3, 4]

        result = prefix_cache._find_best_prefix_match(tokens)

        assert result is None

    def test_find_best_prefix_match_with_match(self, prefix_cache, paged_cache):
        """Test _find_best_prefix_match finding a matching prefix."""
        tokens = [1, 2, 3, 4]
        block_ids = [1, 2]

        # Manually add to prefix index
        block_hash = compute_block_hash(
            b"", tokens, model_name=paged_cache.model_name
        )
        prefix_cache._prefix_index[block_hash] = (4, block_ids, 1)

        result = prefix_cache._find_best_prefix_match(tokens)

        assert result is not None
        prefix_len, matched_ids, num_blocks = result
        assert prefix_len == 4

    def test_prefix_index_immutable_after_store(self, prefix_cache, paged_cache):
        """Test that _prefix_index entries are not affected by later mutations
        of the original block_ids list (e.g., from CoW or block reallocation).

        Regression test for: storing a mutable list reference in _prefix_index
        allows CoW operations to silently corrupt the index.
        """
        tokens = [1, 2, 3, 4, 5, 6, 7, 8]  # 2 blocks (block_size=4)

        # Allocate blocks
        blocks = paged_cache.get_new_blocks(2)
        block_ids = [b.block_id for b in blocks]
        original_ids = list(block_ids)  # snapshot for assertion

        # Store into prefix index
        prefix_cache._update_prefix_index(tokens, block_ids)

        # Simulate CoW: mutate the original list in-place
        block_ids[0] = 9999

        # Verify: prefix_index must still contain the original block IDs
        result = prefix_cache._find_best_prefix_match(tokens)
        assert result is not None
        _, matched_ids, num_blocks = result
        assert list(matched_ids[:num_blocks]) == original_ids[:num_blocks]


class TestValidateBlockCacheData:
    """Tests for _validate_block_cache_data method."""

    @pytest.fixture
    def prefix_cache(self):
        """Create a BlockAwarePrefixCache."""
        paged_cache = PagedCacheManager(
            block_size=4,
            max_blocks=100,
            model_name="test-model",
            initial_blocks=100,
        )
        model = MockModel(num_layers=4)
        return BlockAwarePrefixCache(
            model=model,
            paged_cache_manager=paged_cache,
        )

    def test_validate_empty_data(self, prefix_cache):
        """Test validation of empty cache data."""
        assert prefix_cache._validate_block_cache_data([]) is False

    def test_validate_with_none_values(self, prefix_cache):
        """Test validation fails with None keys/values."""
        cache_data = [(None, MagicMock())]
        assert prefix_cache._validate_block_cache_data(cache_data) is False

        cache_data = [(MagicMock(), None)]
        assert prefix_cache._validate_block_cache_data(cache_data) is False

    def test_validate_with_arrays_cache_skipped(self, prefix_cache):
        """Test validation skips ArraysCache layers for seq_len check."""
        # Create mock tensors with shape
        mock_keys1 = MagicMock()
        mock_keys1.shape = (1, 8, 64, 64)  # KVCache shape
        mock_values1 = MagicMock()

        mock_keys2 = MagicMock()
        mock_keys2.shape = (1, 128)  # ArraysCache has different shape
        mock_values2 = MagicMock()

        cache_data = [
            (mock_keys1, mock_values1),
            (mock_keys2, mock_values2),
        ]
        layer_cache_types = ["KVCache", "ArraysCache"]

        # Should pass because ArraysCache is skipped in validation
        result = prefix_cache._validate_block_cache_data(
            cache_data, layer_cache_types
        )
        assert result is True

    def test_validate_seq_len_mismatch(self, prefix_cache):
        """Test validation fails on seq_len mismatch."""
        mock_keys1 = MagicMock()
        mock_keys1.shape = (1, 8, 64, 64)
        mock_values1 = MagicMock()

        mock_keys2 = MagicMock()
        mock_keys2.shape = (1, 8, 32, 64)  # Different seq_len
        mock_values2 = MagicMock()

        cache_data = [
            (mock_keys1, mock_values1),
            (mock_keys2, mock_values2),
        ]
        layer_cache_types = ["KVCache", "KVCache"]

        result = prefix_cache._validate_block_cache_data(
            cache_data, layer_cache_types
        )
        assert result is False


class TestArraysCacheLastBlockOnly:
    """Tests for ArraysCache last-block-only storage and partial match rejection."""

    @pytest.fixture
    def mx(self):
        """Import MLX or skip."""
        try:
            import mlx.core as mx
            return mx
        except ImportError:
            pytest.skip("MLX not available")

    @pytest.fixture
    def paged_cache(self):
        """Create a PagedCacheManager."""
        return PagedCacheManager(
            block_size=4,
            max_blocks=100,
            model_name="test-model",
            initial_blocks=100,
        )

    @pytest.fixture
    def prefix_cache(self, paged_cache):
        """Create a BlockAwarePrefixCache."""
        model = MockModel(num_layers=2)
        return BlockAwarePrefixCache(
            model=model,
            paged_cache_manager=paged_cache,
        )

    def test_extract_block_arrays_cache_last_block_stores_full_state(
        self, prefix_cache, mx
    ):
        """Last block should store the full ArraysCache state."""
        conv_state = mx.ones((1, 3, 64))
        ssm_state = mx.ones((1, 32, 128, 128))
        cache_data = [
            {
                "state": (conv_state, ssm_state),
                "cache_type": "ArraysCache",
                "class_name": "ArraysCache",
            },
        ]

        result = prefix_cache._extract_block_tensor_slice(
            cache_data, 0, 4, model_cache_config=None, is_last_block=True,
        )

        assert result is not None
        assert len(result) == 1
        keys, values = result[0]
        # Should be full state, not placeholder
        assert keys.shape == (1, 3, 64)
        assert values.shape == (1, 32, 128, 128)

    def test_extract_block_arrays_cache_non_last_block_stores_placeholder(
        self, prefix_cache, mx
    ):
        """Non-last block should store placeholder (1,) for ArraysCache."""
        conv_state = mx.ones((1, 3, 64))
        ssm_state = mx.ones((1, 32, 128, 128))
        cache_data = [
            {
                "state": (conv_state, ssm_state),
                "cache_type": "ArraysCache",
                "class_name": "ArraysCache",
            },
        ]

        result = prefix_cache._extract_block_tensor_slice(
            cache_data, 0, 4, model_cache_config=None, is_last_block=False,
        )

        assert result is not None
        assert len(result) == 1
        keys, values = result[0]
        # Should be placeholder
        assert keys.shape == (1,)
        assert values.shape == (1,)

    def test_extract_block_hybrid_model_arrays_cache_and_kvcache(
        self, prefix_cache, mx
    ):
        """Hybrid model: KVCache sliced normally, ArraysCache last-block-only."""
        from omlx.cache.hybrid_cache import ModelCacheConfig

        kv_keys = mx.ones((1, 8, 8, 64))
        kv_values = mx.ones((1, 8, 8, 64))
        conv_state = mx.ones((1, 3, 64))
        ssm_state = mx.ones((1, 32, 128, 128))

        cache_data = [
            {
                "state": (kv_keys, kv_values),
                "cache_type": "KVCache",
                "class_name": "KVCache",
            },
            {
                "state": (conv_state, ssm_state),
                "cache_type": "ArraysCache",
                "class_name": "ArraysCache",
            },
        ]

        config = ModelCacheConfig.from_type_list(
            ["KVCache", "ArraysCache"], model_name="test"
        )

        # Non-last block
        result = prefix_cache._extract_block_tensor_slice(
            cache_data, 0, 4, model_cache_config=config, is_last_block=False,
        )
        assert result is not None
        assert len(result) == 2
        # KVCache layer should be sliced normally
        assert result[0][0].shape[2] == 4  # seq_len slice
        # ArraysCache layer should be placeholder
        assert result[1][0].shape == (1,)

        # Last block
        result = prefix_cache._extract_block_tensor_slice(
            cache_data, 4, 8, model_cache_config=config, is_last_block=True,
        )
        assert result is not None
        assert len(result) == 2
        # KVCache layer should be sliced
        assert result[0][0].shape[2] == 4
        # ArraysCache layer should have full state
        assert result[1][0].shape == (1, 3, 64)

    def test_reconstruct_arrays_cache_partial_match_returns_none(
        self, prefix_cache, mx
    ):
        """Partial match (placeholder in last block) should return None."""
        from omlx.cache.paged_ssd_cache import PagedSSDCacheManager

        # Create mock SSD cache
        mock_ssd = MagicMock(spec=PagedSSDCacheManager)

        model = MockModel(num_layers=2)
        paged_cache = PagedCacheManager(
            block_size=4, max_blocks=100, model_name="test-model",
            initial_blocks=100,
        )
        cache = BlockAwarePrefixCache(
            model=model,
            paged_cache_manager=paged_cache,
            paged_ssd_cache_manager=mock_ssd,
        )

        # Allocate blocks
        block1 = paged_cache.allocate_block()
        block1.block_hash = b"hash1"
        block1.token_count = 4
        block2 = paged_cache.allocate_block()
        block2.block_hash = b"hash2"
        block2.token_count = 4

        block_table = BlockTable(
            request_id="req-001",
            block_ids=[block1.block_id, block2.block_id],
            num_tokens=8,
        )

        # Block data: KVCache layer sliced, ArraysCache layer placeholder
        kv_slice = (mx.ones((1, 8, 4, 64)), mx.ones((1, 8, 4, 64)))
        placeholder = (mx.zeros((1,)), mx.zeros((1,)))

        block1_data = [kv_slice, placeholder]  # ArraysCache = placeholder (non-last)
        block2_data = [kv_slice, placeholder]  # ArraysCache = placeholder (still non-last in original)

        mock_ssd.load_block_with_metadata.side_effect = [
            (block1_data, {
                "model_name": "test-model",
                "num_layers": 2,
                "layer_cache_types": ["KVCache", "ArraysCache"],
                "layer_meta_states": [(), ()],
            }),
            (block2_data, {
                "model_name": "test-model",
                "num_layers": 2,
                "layer_cache_types": ["KVCache", "ArraysCache"],
                "layer_meta_states": [(), ()],
            }),
        ]

        result = cache.reconstruct_cache(block_table)

        # Should return None because ArraysCache layer has placeholder
        assert result is None

    def test_reconstruct_arrays_cache_exact_match_succeeds(
        self, prefix_cache, mx
    ):
        """Exact match (full state in last block) should reconstruct successfully."""
        from omlx.cache.paged_ssd_cache import PagedSSDCacheManager

        mock_ssd = MagicMock(spec=PagedSSDCacheManager)

        model = MockModel(num_layers=1)
        paged_cache = PagedCacheManager(
            block_size=4, max_blocks=100, model_name="test-model",
            initial_blocks=100,
        )
        cache = BlockAwarePrefixCache(
            model=model,
            paged_cache_manager=paged_cache,
            paged_ssd_cache_manager=mock_ssd,
        )

        block1 = paged_cache.allocate_block()
        block1.block_hash = b"hash1"
        block1.token_count = 4
        block2 = paged_cache.allocate_block()
        block2.block_hash = b"hash2"
        block2.token_count = 4

        block_table = BlockTable(
            request_id="req-001",
            block_ids=[block1.block_id, block2.block_id],
            num_tokens=8,
        )

        # First block: placeholder, Second block (last): full state
        conv_state = mx.ones((1, 3, 64))
        ssm_state = mx.ones((1, 32, 128, 128))

        block1_data = [(mx.zeros((1,)), mx.zeros((1,)))]  # placeholder
        block2_data = [(conv_state, ssm_state)]  # full state

        mock_ssd.load_block_with_metadata.side_effect = [
            (block1_data, {
                "model_name": "test-model",
                "num_layers": 1,
                "layer_cache_types": ["ArraysCache"],
                "layer_meta_states": [()],
            }),
            (block2_data, {
                "model_name": "test-model",
                "num_layers": 1,
                "layer_cache_types": ["ArraysCache"],
                "layer_meta_states": [()],
            }),
        ]

        result = cache.reconstruct_cache(block_table)

        # Should succeed because last block has full state
        assert result is not None
        assert len(result) == 1

    def test_store_cache_skips_partial_blocks(self, mx):
        """store_cache should only create full blocks, skipping partial trailing tokens.

        get_computed_blocks() matches full blocks only (floor division), so
        partial blocks are never matched. Skipping them ensures is_last_block
        points to the last full block, which is critical for ArraysCache/
        RotatingKVCache last-block-only storage.
        """
        block_size = 4
        paged_cache = PagedCacheManager(
            block_size=block_size,
            max_blocks=100,
            model_name="test-model",
            initial_blocks=100,
        )
        model = MockModel(num_layers=1)
        cache = BlockAwarePrefixCache(
            model=model,
            paged_cache_manager=paged_cache,
        )

        # 10 tokens = 2 full blocks (8 tokens) + 2 partial tokens
        tokens = list(range(10))
        cache_data = [
            {"state": (mx.ones((1, 8, 10, 64)), mx.ones((1, 8, 10, 64))),
             "cache_type": "KVCache", "class_name": "KVCache"}
        ]

        result = cache.store_cache("req-001", tokens, cache_data)

        assert result is not None
        # Should create exactly 2 blocks (10 // 4 = 2), not 3
        assert len(result.block_ids) == 2
        # num_tokens should reflect only full blocks
        assert result.num_tokens == 8  # 2 blocks * 4 tokens

        stats = cache.get_stats()
        assert stats.partial_block_skips == 1
        assert stats.partial_tokens_skipped == 2
        assert stats.last_partial_tokens_skipped == 2
        assert stats.last_tokens_to_next_block == 2

    def test_store_cache_arrayscache_partial_trailing_uses_last_full_block_state(self, mx):
        """ArraysCache with trailing partial tokens stores only full blocks safely."""
        from omlx.cache.hybrid_cache import ModelCacheConfig

        block_size = 4
        paged_cache = PagedCacheManager(
            block_size=block_size,
            max_blocks=100,
            model_name="test-model",
            initial_blocks=100,
        )
        mock_ssd = MagicMock()
        mock_ssd.save_block.return_value = True

        model = MockModel(num_layers=1)
        cache = BlockAwarePrefixCache(
            model=model,
            paged_cache_manager=paged_cache,
            paged_ssd_cache_manager=mock_ssd,
        )

        # 7 tokens = 1 full block (4) + 3 trailing partial tokens
        tokens = list(range(7))
        conv_state = mx.ones((1, 3, 64))
        ssm_state = mx.ones((1, 32, 128, 128))
        cache_data = [
            {
                "state": (conv_state, ssm_state),
                "cache_type": "ArraysCache",
                "class_name": "ArraysCache",
            }
        ]
        model_cache_config = ModelCacheConfig.from_type_list(
            ["ArraysCache"], model_name="test-model"
        )

        result = cache.store_cache(
            "req-001",
            tokens,
            cache_data,
            model_cache_config=model_cache_config,
        )

        assert result is not None
        assert len(result.block_ids) == 1
        assert result.num_tokens == 4
        mock_ssd.save_block.assert_called_once()

        saved_data = mock_ssd.save_block.call_args.kwargs["cache_data"]
        saved_conv_state, saved_ssm_state = saved_data[0]
        assert saved_conv_state.shape == conv_state.shape
        assert saved_ssm_state.shape == ssm_state.shape

    def test_store_cache_all_partial_creates_no_blocks(self, mx):
        """Tokens fewer than block_size should create no blocks."""
        block_size = 4
        paged_cache = PagedCacheManager(
            block_size=block_size,
            max_blocks=100,
            model_name="test-model",
            initial_blocks=100,
        )
        model = MockModel(num_layers=1)
        cache = BlockAwarePrefixCache(
            model=model,
            paged_cache_manager=paged_cache,
        )

        # 3 tokens < block_size=4 -> 0 full blocks
        tokens = [1, 2, 3]
        cache_data = [
            {"state": (mx.ones((1, 8, 3, 64)), mx.ones((1, 8, 3, 64))),
             "cache_type": "KVCache", "class_name": "KVCache"}
        ]

        result = cache.store_cache("req-001", tokens, cache_data)

        assert result is not None
        assert len(result.block_ids) == 0
        assert result.num_tokens == 0

        stats = cache.get_stats()
        assert stats.partial_block_skips == 1
        assert stats.partial_tokens_skipped == 3
        assert stats.last_partial_tokens_skipped == 3
        assert stats.last_tokens_to_next_block == 1

    def test_store_cache_exact_multiple_creates_all_blocks(self, mx):
        """Tokens exactly divisible by block_size should create all blocks."""
        block_size = 4
        paged_cache = PagedCacheManager(
            block_size=block_size,
            max_blocks=100,
            model_name="test-model",
            initial_blocks=100,
        )
        model = MockModel(num_layers=1)
        cache = BlockAwarePrefixCache(
            model=model,
            paged_cache_manager=paged_cache,
        )

        # 8 tokens = exactly 2 blocks
        tokens = list(range(8))
        cache_data = [
            {"state": (mx.ones((1, 8, 8, 64)), mx.ones((1, 8, 8, 64))),
             "cache_type": "KVCache", "class_name": "KVCache"}
        ]

        result = cache.store_cache("req-001", tokens, cache_data)

        assert result is not None
        assert len(result.block_ids) == 2
        assert result.num_tokens == 8

    def test_fetch_cache_with_segmented_extra_key_ranges(self):
        """Later image changes should preserve reuse before their boundary."""
        block_size = 4
        paged_cache = PagedCacheManager(
            block_size=block_size,
            max_blocks=100,
            model_name="test-model",
            initial_blocks=100,
        )
        model = MockModel(num_layers=1)
        cache = BlockAwarePrefixCache(
            model=model,
            paged_cache_manager=paged_cache,
        )

        tokens = list(range(12))
        original_ranges = [
            (5, ("image-1",)),
            (9, ("image-1", "image-2")),
        ]

        stored = cache.store_cache(
            "req-store",
            tokens,
            [],
            extra_key_ranges=original_ranges,
        )
        assert stored is not None
        assert stored.num_tokens == 12

        exact_table, exact_remaining = cache.fetch_cache(
            "req-exact",
            tokens,
            extra_key_ranges=original_ranges,
        )
        assert exact_table is not None
        assert exact_table.num_tokens == 12
        assert exact_remaining == []

        changed_later_image_table, changed_later_image_remaining = cache.fetch_cache(
            "req-later-image",
            tokens,
            extra_key_ranges=[
                (5, ("image-1",)),
                (9, ("image-1", "image-3")),
            ],
        )
        assert changed_later_image_table is not None
        assert changed_later_image_table.num_tokens == 8
        assert changed_later_image_remaining == tokens[8:]

        changed_earlier_image_table, changed_earlier_image_remaining = cache.fetch_cache(
            "req-earlier-image",
            tokens,
            extra_key_ranges=[
                (5, ("image-x",)),
                (9, ("image-x", "image-2")),
            ],
        )
        assert changed_earlier_image_table is not None
        assert changed_earlier_image_table.num_tokens == 4
        assert changed_earlier_image_remaining == tokens[4:]

    def test_store_cache_with_existing_prefix_uses_global_cache_indices(self, mx):
        """Store new blocks from full-sequence cache slices after cache hit.

        When a request reuses prefix blocks, extracted cache data includes the
        full sequence (prefix + newly processed suffix). New block slicing must
        use global token indices, otherwise wrong KV ranges are persisted.
        """
        block_size = 4
        paged_cache = PagedCacheManager(
            block_size=block_size,
            max_blocks=100,
            model_name="test-model",
            initial_blocks=100,
        )
        mock_ssd = MagicMock()
        mock_ssd.save_block.return_value = True

        model = MockModel(num_layers=1)
        cache = BlockAwarePrefixCache(
            model=model,
            paged_cache_manager=paged_cache,
            paged_ssd_cache_manager=mock_ssd,
        )

        # Existing cached prefix (1 full block).
        existing_tokens = [10, 11, 12, 13]
        block_table = paged_cache.create_block_table("req-001")
        existing_block = paged_cache.allocate_block()
        assert existing_block is not None
        existing_hash = compute_block_hash(None, existing_tokens, model_name="test-model")
        existing_block.block_hash = existing_hash
        existing_block.token_count = block_size
        block_table.block_ids.append(existing_block.block_id)
        block_table.num_tokens = block_size
        paged_cache.register_block_hash(existing_block, existing_tokens, None)

        # Full sequence and full-sequence cache tensors.
        full_tokens = existing_tokens + [20, 21, 22, 23]
        keys = mx.arange(8, dtype=mx.float32).reshape(1, 1, 8, 1)
        values = (mx.arange(8, dtype=mx.float32) + 100).reshape(1, 1, 8, 1)
        cache_data = [
            {"state": (keys, values), "cache_type": "KVCache", "class_name": "KVCache"}
        ]

        result = cache.store_cache("req-001", full_tokens, cache_data)

        assert result is not None
        # Existing 1 block + newly stored 1 block
        assert len(result.block_ids) == 2
        assert mock_ssd.save_block.call_count == 1

        saved_block_data = mock_ssd.save_block.call_args.kwargs["cache_data"]
        saved_keys, saved_values = saved_block_data[0]

        # New block must use global slice [4:8], not [0:4].
        expected_keys = keys[:, :, 4:8, :]
        expected_values = values[:, :, 4:8, :]
        assert saved_keys.tolist() == expected_keys.tolist()
        assert saved_values.tolist() == expected_values.tolist()

    def test_store_cache_rolls_back_when_ssd_save_fails(self, mx):
        """Failed SSD save should not retain block metadata in paged cache."""
        block_size = 4
        paged_cache = PagedCacheManager(
            block_size=block_size,
            max_blocks=100,
            model_name="test-model",
            initial_blocks=100,
        )
        mock_ssd = MagicMock()
        mock_ssd.save_block.return_value = False

        model = MockModel(num_layers=1)
        cache = BlockAwarePrefixCache(
            model=model,
            paged_cache_manager=paged_cache,
            paged_ssd_cache_manager=mock_ssd,
        )

        tokens = [1, 2, 3, 4]  # exactly one full block
        keys = mx.ones((1, 8, 4, 64))
        values = mx.ones((1, 8, 4, 64))
        cache_data = [
            {"state": (keys, values), "cache_type": "KVCache", "class_name": "KVCache"}
        ]

        result = cache.store_cache("req-rollback", tokens, cache_data)

        assert result is not None
        # If persistence fails, block should be rolled back (not indexed/retained).
        assert len(result.block_ids) == 0
        assert result.num_tokens == 0
        assert paged_cache.stats.allocated_blocks == 1  # null block only

        failed_hash = compute_block_hash(None, tokens, model_name="test-model")
        assert paged_cache.cached_block_hash_to_block.get_block(failed_hash) is None

    def test_store_cache_keeps_valid_prefix_when_later_ssd_save_fails(self, mx):
        """A later SSD save failure should roll back only the failed block."""
        block_size = 4
        paged_cache = PagedCacheManager(
            block_size=block_size,
            max_blocks=100,
            model_name="test-model",
            initial_blocks=100,
        )
        mock_ssd = MagicMock()
        mock_ssd.save_block.side_effect = [True, False]

        model = MockModel(num_layers=1)
        cache = BlockAwarePrefixCache(
            model=model,
            paged_cache_manager=paged_cache,
            paged_ssd_cache_manager=mock_ssd,
        )

        # Seed request with one already-cached prefix block.
        existing_tokens = [10, 11, 12, 13]
        existing_block = paged_cache.allocate_block()
        assert existing_block is not None
        existing_block.token_count = block_size
        block_table = paged_cache.create_block_table("req-partial-rollback")
        block_table.block_ids.append(existing_block.block_id)
        block_table.num_tokens = block_size
        paged_cache.register_block_hash(existing_block, existing_tokens, None)
        assert existing_block.block_hash is not None

        # Add two more blocks; first save succeeds, second save fails.
        tokens = existing_tokens + [20, 21, 22, 23, 30, 31, 32, 33]
        keys = mx.arange(12, dtype=mx.float32).reshape(1, 1, 12, 1)
        values = (mx.arange(12, dtype=mx.float32) + 100).reshape(1, 1, 12, 1)
        cache_data = [
            {"state": (keys, values), "cache_type": "KVCache", "class_name": "KVCache"}
        ]

        result = cache.store_cache("req-partial-rollback", tokens, cache_data)

        assert result is not None

        first_new_hash = compute_block_hash(
            existing_block.block_hash, tokens[4:8], model_name="test-model"
        )
        failed_hash = compute_block_hash(first_new_hash, tokens[8:12], model_name="test-model")

        # Keep existing block + first new block; drop only the failed second new block.
        assert len(result.block_ids) == 2
        assert result.num_tokens == 8
        assert result.block_ids[0] == existing_block.block_id

        first_new_block = paged_cache.cached_block_hash_to_block.get_block(first_new_hash)
        assert first_new_block is not None
        assert result.block_ids[1] == first_new_block.block_id
        assert result.block_ids == [existing_block.block_id, first_new_block.block_id]

        # save_block should attempt exactly two writes in this scenario.
        calls = mock_ssd.save_block.call_args_list
        assert len(calls) == 2
        attempted_hashes = [call.kwargs["block_hash"] for call in calls]
        assert attempted_hashes == [first_new_hash, failed_hash]
        assert [call.kwargs["token_count"] for call in calls] == [block_size, block_size]

        # Verify global-index slices were persisted for both attempted new blocks.
        first_saved_keys, first_saved_values = calls[0].kwargs["cache_data"][0]
        failed_saved_keys, failed_saved_values = calls[1].kwargs["cache_data"][0]
        assert first_saved_keys.tolist() == keys[:, :, 4:8, :].tolist()
        assert first_saved_values.tolist() == values[:, :, 4:8, :].tolist()
        assert failed_saved_keys.tolist() == keys[:, :, 8:12, :].tolist()
        assert failed_saved_values.tolist() == values[:, :, 8:12, :].tolist()

        assert paged_cache.cached_block_hash_to_block.get_block(first_new_hash) is not None
        assert paged_cache.cached_block_hash_to_block.get_block(failed_hash) is None
        # Failed block should be freed, not just removed from hash index.
        allocated_non_null_ids = {
            block.block_id
            for block in paged_cache.allocated_blocks.values()
            if not block.is_null
        }
        assert allocated_non_null_ids == {existing_block.block_id, first_new_block.block_id}
        assert all(
            b.block_hash != failed_hash for b in paged_cache.allocated_blocks.values()
        )

        # Public contract after partial failure: only valid prefix should be reused.
        expected_partial_ids = [existing_block.block_id, first_new_block.block_id]
        fetched_partial, remaining_partial = cache.fetch_cache(
            "req-partial-rollback-hit", tokens
        )
        assert fetched_partial is not None
        assert fetched_partial.block_ids == expected_partial_ids
        assert fetched_partial.num_tokens == 8
        assert remaining_partial == tokens[8:12]

    def test_store_cache_retry_after_partial_failure_saves_only_missing_tail(self, mx):
        """Retry should preserve valid prefix and only save the missing tail block."""
        block_size = 4
        paged_cache = PagedCacheManager(
            block_size=block_size,
            max_blocks=100,
            model_name="test-model",
            initial_blocks=100,
        )
        mock_ssd = MagicMock()
        mock_ssd.save_block.side_effect = [True, False, True]

        model = MockModel(num_layers=1)
        cache = BlockAwarePrefixCache(
            model=model,
            paged_cache_manager=paged_cache,
            paged_ssd_cache_manager=mock_ssd,
        )

        existing_tokens = [10, 11, 12, 13]
        existing_block = paged_cache.allocate_block()
        assert existing_block is not None
        existing_block.token_count = block_size
        block_table = paged_cache.create_block_table("req-retry")
        block_table.block_ids.append(existing_block.block_id)
        block_table.num_tokens = block_size
        paged_cache.register_block_hash(existing_block, existing_tokens, None)
        assert existing_block.block_hash is not None

        tokens = existing_tokens + [20, 21, 22, 23, 30, 31, 32, 33]
        keys = mx.arange(12, dtype=mx.float32).reshape(1, 1, 12, 1)
        values = (mx.arange(12, dtype=mx.float32) + 100).reshape(1, 1, 12, 1)
        cache_data = [
            {"state": (keys, values), "cache_type": "KVCache", "class_name": "KVCache"}
        ]

        first_result = cache.store_cache("req-retry", tokens, cache_data)
        assert first_result is not None

        first_new_hash = compute_block_hash(
            existing_block.block_hash, tokens[4:8], model_name="test-model"
        )
        tail_hash = compute_block_hash(first_new_hash, tokens[8:12], model_name="test-model")
        first_new_block = paged_cache.cached_block_hash_to_block.get_block(first_new_hash)
        assert first_new_block is not None
        retained_prefix_ids = first_result.block_ids.copy()
        assert retained_prefix_ids == [existing_block.block_id, first_new_block.block_id]

        retry_result = cache.store_cache("req-retry", tokens, cache_data)
        assert retry_result is not None

        calls = mock_ssd.save_block.call_args_list
        assert len(calls) == 3
        attempted_hashes = [call.kwargs["block_hash"] for call in calls]
        assert attempted_hashes == [first_new_hash, tail_hash, tail_hash]
        assert attempted_hashes.count(first_new_hash) == 1
        assert attempted_hashes.count(tail_hash) == 2
        assert [call.kwargs["token_count"] for call in calls] == [block_size, block_size, block_size]
        retry_saved_keys, retry_saved_values = calls[2].kwargs["cache_data"][0]
        assert retry_saved_keys.tolist() == keys[:, :, 8:12, :].tolist()
        assert retry_saved_values.tolist() == values[:, :, 8:12, :].tolist()

        assert len(retry_result.block_ids) == 3
        assert retry_result.num_tokens == 12
        assert retry_result.block_ids[:2] == retained_prefix_ids
        assert len(set(retry_result.block_ids)) == 3

        tail_block = paged_cache.cached_block_hash_to_block.get_block(tail_hash)
        assert tail_block is not None
        assert retry_result.block_ids[2] == tail_block.block_id

        # Reconstruct full cache and verify tensor content for retry flow.
        saved_by_hash = {
            existing_block.block_hash: [
                (keys[:, :, 0:4, :], values[:, :, 0:4, :]),
            ],
        }
        for call in calls:
            saved_by_hash[call.kwargs["block_hash"]] = call.kwargs["cache_data"]

        def load_block_with_metadata(block_hash):
            block_data = saved_by_hash.get(block_hash)
            if block_data is None:
                return None, None
            return (
                block_data,
                {
                    "model_name": "test-model",
                    "num_layers": 1,
                    "layer_cache_types": ["KVCache"],
                    "layer_meta_states": [()],
                },
            )

        mock_ssd.load_block_with_metadata.side_effect = load_block_with_metadata
        reconstructed = cache.reconstruct_cache(retry_result)
        assert reconstructed is not None
        assert len(reconstructed) == 1
        layer_cache = reconstructed[0]
        if hasattr(layer_cache, "state"):
            reconstructed_keys, reconstructed_values = layer_cache.state
        elif isinstance(layer_cache, (list, tuple)) and len(layer_cache) == 2:
            reconstructed_keys, reconstructed_values = layer_cache
        else:
            reconstructed_keys, reconstructed_values = layer_cache.keys, layer_cache.values

        assert reconstructed_keys.tolist() == keys.tolist()
        assert reconstructed_values.tolist() == values.tolist()

        # Force prefix-index fallback by removing chain-hash index entries.
        for block_id in retry_result.block_ids:
            block = paged_cache.allocated_blocks.get(block_id)
            assert block is not None
            assert block.block_hash is not None
            paged_cache.cached_block_hash_to_block.pop(block.block_hash, block.block_id)

        # Explicitly prove shared-hash path cannot succeed in this fixture.
        assert paged_cache._paged_ssd_cache_manager is None
        shared_block_ids, _ = paged_cache.find_shared_prefix(tokens)
        assert shared_block_ids == []

        expected_ids = retry_result.block_ids.copy()
        # Public contract via prefix-index fallback: full prefix hit, no remaining tokens.
        fetched_table, remaining = cache.fetch_cache("req-retry-prefix-index-hit", tokens)
        assert fetched_table is not None
        assert fetched_table.block_ids == expected_ids
        assert fetched_table.num_tokens == 12
        assert remaining == []


class TestPrefixCacheCacheList:
    """Tests for CacheList support in BlockAwarePrefixCache."""

    @pytest.fixture
    def mx(self):
        """Import MLX or skip."""
        try:
            import mlx.core as mx
            return mx
        except ImportError:
            pytest.skip("MLX not available")

    @pytest.fixture
    def paged_cache(self):
        """Create a PagedCacheManager for testing."""
        return PagedCacheManager(
            block_size=4,
            max_blocks=100,
            model_name="test-model",
            initial_blocks=100,
        )

    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        return MockModel(num_layers=4)

    @pytest.fixture
    def prefix_cache(self, mock_model, paged_cache):
        """Create a BlockAwarePrefixCache for testing."""
        return BlockAwarePrefixCache(
            model=mock_model,
            paged_cache_manager=paged_cache,
            paged_ssd_cache_manager=None,
        )

    def test_get_cache_seq_len_cache_list_only(self, prefix_cache, mx):
        """Test _get_cache_seq_len with all-CacheList model (e.g., deepseek_v32)."""
        # CacheList sub-states with 4D tensors
        sub_keys = mx.zeros((1, 8, 128, 64))
        sub_values = mx.zeros((1, 8, 128, 64))

        cache_data = [
            {
                'state': [(sub_keys, sub_values)],  # CacheList sub-states
                'cache_type': 'CacheList',
                'class_name': 'CacheList',
            },
            {
                'state': [(sub_keys, sub_values)],
                'cache_type': 'CacheList',
                'class_name': 'CacheList',
            },
        ]

        seq_len = prefix_cache._get_cache_seq_len(cache_data)
        assert seq_len == 128

    def test_get_cache_seq_len_mixed_kvcache_and_cache_list(self, prefix_cache, mx):
        """Test _get_cache_seq_len with mixed KVCache + CacheList model."""
        kv_keys = mx.zeros((1, 8, 256, 64))
        kv_values = mx.zeros((1, 8, 256, 64))
        sub_keys = mx.zeros((1, 4, 128, 64))

        cache_data = [
            {
                'state': (kv_keys, kv_values),
                'cache_type': 'KVCache',
                'class_name': 'KVCache',
            },
            {
                'state': [(sub_keys, MagicMock())],
                'cache_type': 'CacheList',
                'class_name': 'CacheList',
            },
        ]

        seq_len = prefix_cache._get_cache_seq_len(cache_data)
        # Should use KVCache layer (Step 1), not CacheList
        assert seq_len == 256

    def test_get_cache_seq_len_cache_list_skipped_in_step1(self, prefix_cache, mx):
        """Test CacheList is skipped in Step 1 and Step 2."""
        sub_keys = mx.zeros((1, 8, 64, 64))

        cache_data = [
            {
                'state': [(sub_keys, MagicMock())],
                'cache_type': 'CacheList',
                'class_name': 'CacheList',
            },
        ]

        # CacheList is in non_sliceable_types, so Step 1 & 2 skip it
        # Step 3 finds the sub-cache seq_len
        seq_len = prefix_cache._get_cache_seq_len(cache_data)
        assert seq_len == 64

    def test_get_cache_seq_len_pure_rotating_kvcache(self, prefix_cache, mx):
        """Test _get_cache_seq_len for pure RotatingKVCache model."""
        rot_keys = mx.zeros((1, 8, 96, 64))
        rot_values = mx.zeros((1, 8, 96, 64))

        cache_data = [
            {
                'state': (rot_keys, rot_values),
                'cache_type': 'RotatingKVCache',
                'class_name': 'RotatingKVCache',
            },
        ]

        # Step 1 skips RotatingKVCache; Step 2 must still recover seq_len.
        seq_len = prefix_cache._get_cache_seq_len(cache_data)
        assert seq_len == 96

    def test_extract_block_tensor_slice_cache_list_last_block(self, prefix_cache, mx):
        """Test _extract_block_tensor_slice for CacheList on last block."""
        from omlx.cache.hybrid_cache import ModelCacheConfig

        sub_keys = mx.zeros((1, 8, 32, 64))
        sub_values = mx.ones((1, 8, 32, 64))

        cache_data = [
            {
                'state': [(sub_keys, sub_values)],
                'cache_type': 'CacheList',
                'class_name': 'CacheList',
            },
        ]
        config = ModelCacheConfig.from_type_list(["CacheList"])

        result = prefix_cache._extract_block_tensor_slice(
            cache_data, 0, 32, model_cache_config=config, is_last_block=True,
        )

        assert result is not None
        assert len(result) == 1
        # CacheList marker format
        assert result[0][0] == '__cache_list__'
        assert len(result[0][1]) == 1  # One sub-cache
        assert result[0][1][0][0].shape == (1, 8, 32, 64)

    def test_extract_block_tensor_slice_cache_list_non_last_sliceable(self, prefix_cache, mx):
        """Test _extract_block_tensor_slice for CacheList with sliceable sub-caches on non-last block.

        When all sub-caches are 4D KVCache tensors, they should be sliced
        per-block instead of using last-block-only placeholder storage.
        """
        from omlx.cache.hybrid_cache import ModelCacheConfig

        sub_keys = mx.zeros((1, 8, 32, 64))
        sub_values = mx.ones((1, 8, 32, 64))

        cache_data = [
            {
                'state': [(sub_keys, sub_values)],
                'cache_type': 'CacheList',
                'class_name': 'CacheList',
            },
        ]
        config = ModelCacheConfig.from_type_list(["CacheList"])

        result = prefix_cache._extract_block_tensor_slice(
            cache_data, 0, 16, model_cache_config=config, is_last_block=False,
        )

        assert result is not None
        assert len(result) == 1
        # Sliceable sub-caches: per-block sliced data, not placeholder
        assert result[0][0] == '__cache_list__'
        assert len(result[0][1]) == 1
        assert result[0][1][0][0].shape == (1, 8, 16, 64)
        assert result[0][1][0][1].shape == (1, 8, 16, 64)

    def test_extract_block_tensor_slice_cache_list_zero_dim_values(self, prefix_cache, mx):
        """Test per-block slicing for CacheList with zero-dim values (DSA indexer)."""
        from omlx.cache.hybrid_cache import ModelCacheConfig

        # GLM-5 style: main attention + indexer with zero head_dim
        sub_keys1 = mx.zeros((1, 1, 64, 512))
        sub_values1 = mx.zeros((1, 1, 64, 64))
        sub_keys2 = mx.zeros((1, 1, 64, 128))
        sub_values2 = mx.zeros((1, 1, 64, 0))  # zero head_dim

        cache_data = [
            {
                'state': [(sub_keys1, sub_values1), (sub_keys2, sub_values2)],
                'cache_type': 'CacheList',
                'class_name': 'CacheList',
            },
        ]
        config = ModelCacheConfig.from_type_list(["CacheList"])

        result = prefix_cache._extract_block_tensor_slice(
            cache_data, 0, 32, model_cache_config=config, is_last_block=False,
        )

        assert result is not None
        assert result[0][0] == '__cache_list__'
        assert len(result[0][1]) == 2
        # Sub-cache 0: sliced normally
        assert result[0][1][0][0].shape == (1, 1, 32, 512)
        assert result[0][1][0][1].shape == (1, 1, 32, 64)
        # Sub-cache 1: sliced, values remain zero-dim
        assert result[0][1][1][0].shape == (1, 1, 32, 128)
        assert result[0][1][1][1].shape == (1, 1, 32, 0)

    def test_validate_block_cache_data_cache_list(self, prefix_cache, mx):
        """Test _validate_block_cache_data with CacheList layers."""
        # CacheList as list format (last block)
        cache_data = [
            [(mx.zeros((1, 8, 32, 64)), mx.zeros((1, 8, 32, 64)))],  # CacheList sub-cache list
            (mx.zeros((1, 8, 32, 64)), mx.zeros((1, 8, 32, 64))),  # Standard KVCache
        ]
        layer_cache_types = ["CacheList", "KVCache"]

        result = prefix_cache._validate_block_cache_data(cache_data, layer_cache_types)
        assert result is True

    def test_validate_block_cache_data_cache_list_placeholder(self, prefix_cache, mx):
        """Test _validate_block_cache_data with CacheList placeholder."""
        cache_data = [
            (mx.zeros((1,)), mx.zeros((1,))),  # CacheList placeholder (falls through to tuple check)
            (mx.zeros((1, 8, 32, 64)), mx.zeros((1, 8, 32, 64))),  # Standard KVCache
        ]
        layer_cache_types = ["CacheList", "KVCache"]

        result = prefix_cache._validate_block_cache_data(cache_data, layer_cache_types)
        assert result is True

    def test_find_kv_shape_ref_skips_cache_list(self, prefix_cache, mx):
        """Test _find_kv_shape_ref skips CacheList layers."""
        all_block_data = [
            [
                [(mx.zeros((1, 8, 32, 64)), mx.zeros((1, 8, 32, 64)))],  # CacheList: List[Tuple]
                (mx.zeros((1, 4, 32, 128)), mx.zeros((1, 4, 32, 128))),  # KVCache
            ]
        ]
        layer_cache_types = ["CacheList", "KVCache"]

        result = prefix_cache._find_kv_shape_ref(all_block_data, layer_cache_types)
        assert result == (4, 128)  # From KVCache layer, not CacheList

    def test_reconstruct_cache_list_partial_match_reject(self, mx):
        """Test reconstruct_cache rejects CacheList with placeholder (partial match)."""
        from omlx.cache.paged_ssd_cache import PagedSSDCacheManager

        mock_ssd = MagicMock(spec=PagedSSDCacheManager)

        model = MockModel(num_layers=1)
        paged_cache = PagedCacheManager(
            block_size=4, max_blocks=100, model_name="test-model",
            initial_blocks=100,
        )
        cache = BlockAwarePrefixCache(
            model=model,
            paged_cache_manager=paged_cache,
            paged_ssd_cache_manager=mock_ssd,
        )

        # Allocate a block
        block1 = paged_cache.allocate_block()
        block1.block_hash = b"hash_cl1"
        block1.token_count = 4

        block_table = BlockTable(
            request_id="req-cl-partial",
            block_ids=[block1.block_id],
            num_tokens=4,
        )

        # Block data: CacheList placeholder (partial match)
        placeholder = (mx.zeros((1,)), mx.zeros((1,)))
        block1_data = [placeholder]

        mock_ssd.load_block_with_metadata.return_value = (
            block1_data,
            {
                "model_name": "test-model",
                "num_layers": 1,
                "layer_cache_types": ["CacheList"],
                "layer_meta_states": [()],
            },
        )

        result = cache.reconstruct_cache(block_table)
        # Should return None because CacheList layer has placeholder
        assert result is None


class TestWalkBackTruncation:
    """Tests for walk-back truncation of non-sliceable caches."""

    @pytest.fixture
    def mx(self):
        """Import MLX or skip."""
        try:
            import mlx.core as mx
            return mx
        except ImportError:
            pytest.skip("MLX not available")

    # ------------------------------------------------------------------
    # _is_placeholder_state
    # ------------------------------------------------------------------

    def test_is_placeholder_state_detects_placeholder(self, mx):
        """Placeholder tuple with shape (1,) should be detected."""
        placeholder = (mx.zeros((1,)), mx.zeros((1,)))
        assert BlockAwarePrefixCache._is_placeholder_state(placeholder) is True

    def test_is_placeholder_state_rejects_real_arrays_cache(self, mx):
        """Real ArraysCache state should not be flagged as placeholder."""
        real_state = (mx.ones((1, 3, 64)), mx.ones((1, 32, 128, 128)))
        assert BlockAwarePrefixCache._is_placeholder_state(real_state) is False

    def test_is_placeholder_state_rejects_kv_cache(self, mx):
        """Standard 4D KVCache tensors should not be flagged."""
        kv_state = (mx.ones((1, 8, 4, 64)), mx.ones((1, 8, 4, 64)))
        assert BlockAwarePrefixCache._is_placeholder_state(kv_state) is False

    def test_is_placeholder_state_rejects_list(self, mx):
        """CacheList real data (list format) should not be flagged."""
        cache_list_data = [
            (mx.ones((1, 8, 4, 64)), mx.ones((1, 8, 4, 64))),
        ]
        assert BlockAwarePrefixCache._is_placeholder_state(cache_list_data) is False

    # ------------------------------------------------------------------
    # _find_walk_back_truncation_point
    # ------------------------------------------------------------------

    def test_walk_back_no_truncation_when_last_block_valid(self, mx):
        """No truncation when the last block has real state."""
        model = MockModel(num_layers=2)
        paged_cache = PagedCacheManager(
            block_size=4, max_blocks=100, model_name="test-model",
            initial_blocks=100,
        )
        cache = BlockAwarePrefixCache(
            model=model,
            paged_cache_manager=paged_cache,
            paged_ssd_cache_manager=None,
        )

        placeholder = (mx.zeros((1,)), mx.zeros((1,)))
        real_state = (mx.ones((1, 3, 64)), mx.ones((1, 32, 128, 128)))
        kv = (mx.ones((1, 8, 4, 64)), mx.ones((1, 8, 4, 64)))

        all_block_data = [
            [kv, placeholder],  # block 0
            [kv, real_state],   # block 1 (last, valid)
        ]
        layer_cache_types = ["KVCache", "ArraysCache"]

        result = cache._find_walk_back_truncation_point(
            all_block_data, layer_cache_types
        )
        assert result is None  # No truncation needed

    def test_walk_back_multi_turn_pattern(self, mx):
        """Walk-back finds the latest block with valid ArraysCache state.

        Simulates multi-turn pattern:
        A[p] B[p] C[real] D[p] E[real] F[p]
        Should walk back to E (index 4).
        """
        model = MockModel(num_layers=2)
        paged_cache = PagedCacheManager(
            block_size=4, max_blocks=100, model_name="test-model",
            initial_blocks=100,
        )
        cache = BlockAwarePrefixCache(
            model=model,
            paged_cache_manager=paged_cache,
            paged_ssd_cache_manager=None,
        )

        placeholder = (mx.zeros((1,)), mx.zeros((1,)))
        real_state = (mx.ones((1, 3, 64)), mx.ones((1, 32, 128, 128)))
        kv = (mx.ones((1, 8, 4, 64)), mx.ones((1, 8, 4, 64)))

        # A[p] B[p] C[real] D[p] E[real] F[p]
        all_block_data = [
            [kv, placeholder],  # A
            [kv, placeholder],  # B
            [kv, real_state],   # C (turn 1 last block)
            [kv, placeholder],  # D
            [kv, real_state],   # E (turn 2 last block)
            [kv, placeholder],  # F (last loaded, placeholder)
        ]
        layer_cache_types = ["KVCache", "ArraysCache"]

        result = cache._find_walk_back_truncation_point(
            all_block_data, layer_cache_types
        )
        assert result == 4  # Block E (index 4)

    def test_walk_back_all_placeholders_returns_none(self, mx):
        """All blocks have placeholders -- no valid fallback exists."""
        model = MockModel(num_layers=2)
        paged_cache = PagedCacheManager(
            block_size=4, max_blocks=100, model_name="test-model",
            initial_blocks=100,
        )
        cache = BlockAwarePrefixCache(
            model=model,
            paged_cache_manager=paged_cache,
            paged_ssd_cache_manager=None,
        )

        placeholder = (mx.zeros((1,)), mx.zeros((1,)))
        kv = (mx.ones((1, 8, 4, 64)), mx.ones((1, 8, 4, 64)))

        all_block_data = [
            [kv, placeholder],
            [kv, placeholder],
            [kv, placeholder],
        ]
        layer_cache_types = ["KVCache", "ArraysCache"]

        result = cache._find_walk_back_truncation_point(
            all_block_data, layer_cache_types
        )
        assert result is None  # No valid block found

    def test_walk_back_includes_rotating_kv_cache(self, mx):
        """RotatingKVCache placeholders should walk back to latest valid block."""
        model = MockModel(num_layers=2)
        paged_cache = PagedCacheManager(
            block_size=4, max_blocks=100, model_name="test-model",
            initial_blocks=100,
        )
        cache = BlockAwarePrefixCache(
            model=model,
            paged_cache_manager=paged_cache,
            paged_ssd_cache_manager=None,
        )

        placeholder = (mx.zeros((1,)), mx.zeros((1,)))
        rotating_real = (mx.ones((1, 8, 4, 64)), mx.ones((1, 8, 4, 64)))
        kv = (mx.ones((1, 8, 4, 64)), mx.ones((1, 8, 4, 64)))

        # Last block placeholder, previous block valid rotating state.
        all_block_data = [
            [kv, rotating_real],
            [kv, placeholder],
        ]
        layer_cache_types = ["KVCache", "RotatingKVCache"]

        result = cache._find_walk_back_truncation_point(
            all_block_data, layer_cache_types
        )
        assert result == 0

    # ------------------------------------------------------------------
    # Full reconstruct_cache integration with walk-back
    # ------------------------------------------------------------------

    def test_reconstruct_arrays_cache_walks_back_to_valid_block(self, mx):
        """Partial match should walk back instead of rejecting entirely.

        3 blocks: block0[p] block1[real] block2[p]
        Should truncate to blocks 0-1, returning valid cache.
        """
        from omlx.cache.paged_ssd_cache import PagedSSDCacheManager

        mock_ssd = MagicMock(spec=PagedSSDCacheManager)

        model = MockModel(num_layers=2)
        paged_cache = PagedCacheManager(
            block_size=4, max_blocks=100, model_name="test-model",
            initial_blocks=100,
        )
        cache = BlockAwarePrefixCache(
            model=model,
            paged_cache_manager=paged_cache,
            paged_ssd_cache_manager=mock_ssd,
        )

        # Allocate 3 blocks
        blocks = []
        for i in range(3):
            b = paged_cache.allocate_block()
            b.block_hash = f"hash{i}".encode()
            b.token_count = 4
            b.ref_count = 2  # Simulate fetch_cache having incremented ref
            blocks.append(b)

        block_table = BlockTable(
            request_id="req-001",
            block_ids=[b.block_id for b in blocks],
            num_tokens=12,
        )

        kv_slice = (mx.ones((1, 8, 4, 64)), mx.ones((1, 8, 4, 64)))
        placeholder = (mx.zeros((1,)), mx.zeros((1,)))
        conv_state = mx.ones((1, 3, 64))
        ssm_state = mx.ones((1, 32, 128, 128))
        real_state = (conv_state, ssm_state)

        metadata = {
            "model_name": "test-model",
            "num_layers": 2,
            "layer_cache_types": ["KVCache", "ArraysCache"],
            "layer_meta_states": [(), ()],
        }

        mock_ssd.load_block_with_metadata.side_effect = [
            ([kv_slice, placeholder], metadata),   # block 0: placeholder
            ([kv_slice, real_state], metadata),     # block 1: real state (turn 1 last)
            ([kv_slice, placeholder], metadata),    # block 2: placeholder
        ]

        result = cache.reconstruct_cache(block_table)

        # Should NOT be None -- walk-back recovered blocks 0-1
        assert result is not None
        assert len(result) == 2  # 2 layers reconstructed

        # block_table should be truncated to 2 blocks
        assert len(block_table.block_ids) == 2
        assert block_table.num_tokens == 8

        # Block 2 ref_count should have been decremented (freed)
        assert blocks[2].ref_count == 1

    def test_reconstruct_rotating_cache_walks_back_to_valid_block(self, mx):
        """Rotating partial match should walk back to latest valid block."""
        from omlx.cache.paged_ssd_cache import PagedSSDCacheManager

        mock_ssd = MagicMock(spec=PagedSSDCacheManager)

        model = MockModel(num_layers=2)
        paged_cache = PagedCacheManager(
            block_size=4, max_blocks=100, model_name="test-model",
            initial_blocks=100,
        )
        cache = BlockAwarePrefixCache(
            model=model,
            paged_cache_manager=paged_cache,
            paged_ssd_cache_manager=mock_ssd,
        )

        # Allocate 3 blocks
        blocks = []
        for i in range(3):
            b = paged_cache.allocate_block()
            b.block_hash = f"hash{i}".encode()
            b.token_count = 4
            b.ref_count = 2  # Simulate fetch_cache having incremented ref
            blocks.append(b)

        block_table = BlockTable(
            request_id="req-001",
            block_ids=[b.block_id for b in blocks],
            num_tokens=12,
        )

        kv_slice = (mx.ones((1, 8, 4, 64)), mx.ones((1, 8, 4, 64)))
        placeholder = (mx.zeros((1,)), mx.zeros((1,)))
        rotating_real = (mx.ones((1, 8, 4, 64)), mx.ones((1, 8, 4, 64)))

        metadata = {
            "model_name": "test-model",
            "num_layers": 2,
            "layer_cache_types": ["KVCache", "RotatingKVCache"],
            "layer_meta_states": [(), (0, 4, 4, 4)],
        }

        mock_ssd.load_block_with_metadata.side_effect = [
            ([kv_slice, placeholder], metadata),     # block 0: placeholder
            ([kv_slice, rotating_real], metadata),   # block 1: real rotating state
            ([kv_slice, placeholder], metadata),     # block 2: placeholder
        ]

        result = cache.reconstruct_cache(block_table)

        # Should recover blocks 0-1 via walk-back.
        assert result is not None
        assert len(result) == 2
        assert len(block_table.block_ids) == 2
        assert block_table.num_tokens == 8
        assert blocks[2].ref_count == 1

        rotating_cache = result[1]
        assert hasattr(rotating_cache, "max_size")
        assert rotating_cache.max_size == 4

    def test_reconstruct_all_placeholders_still_rejects(self, mx):
        """When no block has valid state, walk-back finds nothing and
        the existing per-layer rejection returns None."""
        from omlx.cache.paged_ssd_cache import PagedSSDCacheManager

        mock_ssd = MagicMock(spec=PagedSSDCacheManager)

        model = MockModel(num_layers=2)
        paged_cache = PagedCacheManager(
            block_size=4, max_blocks=100, model_name="test-model",
            initial_blocks=100,
        )
        cache = BlockAwarePrefixCache(
            model=model,
            paged_cache_manager=paged_cache,
            paged_ssd_cache_manager=mock_ssd,
        )

        blocks = []
        for i in range(2):
            b = paged_cache.allocate_block()
            b.block_hash = f"hash{i}".encode()
            b.token_count = 4
            blocks.append(b)

        block_table = BlockTable(
            request_id="req-001",
            block_ids=[b.block_id for b in blocks],
            num_tokens=8,
        )

        kv_slice = (mx.ones((1, 8, 4, 64)), mx.ones((1, 8, 4, 64)))
        placeholder = (mx.zeros((1,)), mx.zeros((1,)))

        metadata = {
            "model_name": "test-model",
            "num_layers": 2,
            "layer_cache_types": ["KVCache", "ArraysCache"],
            "layer_meta_states": [(), ()],
        }

        mock_ssd.load_block_with_metadata.side_effect = [
            ([kv_slice, placeholder], metadata),
            ([kv_slice, placeholder], metadata),
        ]

        result = cache.reconstruct_cache(block_table)

        # Should still return None -- all placeholders, no walk-back target
        assert result is None

    def test_partial_reconstruction_frees_dropped_blocks(self, mx):
        """Blocks dropped during partial reconstruction should have
        their ref_counts decremented."""
        from omlx.cache.paged_ssd_cache import PagedSSDCacheManager

        mock_ssd = MagicMock(spec=PagedSSDCacheManager)

        model = MockModel(num_layers=1)
        paged_cache = PagedCacheManager(
            block_size=4, max_blocks=100, model_name="test-model",
            initial_blocks=100,
        )
        cache = BlockAwarePrefixCache(
            model=model,
            paged_cache_manager=paged_cache,
            paged_ssd_cache_manager=mock_ssd,
        )

        block1 = paged_cache.allocate_block()
        block1.block_hash = b"hash1"
        block1.token_count = 4
        block1.ref_count = 2  # Simulate fetch_cache increment

        block2 = paged_cache.allocate_block()
        block2.block_hash = b"hash2"
        block2.token_count = 4
        block2.ref_count = 2  # Simulate fetch_cache increment

        block_table = BlockTable(
            request_id="req-001",
            block_ids=[block1.block_id, block2.block_id],
            num_tokens=8,
        )

        kv_slice = (mx.ones((1, 8, 4, 64)), mx.ones((1, 8, 4, 64)))

        # First block loads fine, second fails
        mock_ssd.load_block_with_metadata.side_effect = [
            ([kv_slice], {
                "model_name": "test-model",
                "num_layers": 1,
                "layer_cache_types": ["KVCache"],
                "layer_meta_states": [()],
            }),
            (None, None),  # Second block fails to load
        ]

        result = cache.reconstruct_cache(block_table)

        # Should partially reconstruct with block1 only
        assert result is not None
        assert len(block_table.block_ids) == 1
        assert block_table.num_tokens == 4

        # block2 ref_count should have been decremented
        assert block2.ref_count == 1


class TestPerBlockMetaStates:
    """Tests for per-block meta_states in store_cache with boundary snapshots.

    Verifies that blocks stored with boundary snapshots use the snapshot's
    meta_state (correct per-boundary offset) rather than the shared final
    meta_state from _extract_cache_states.
    """

    @pytest.fixture
    def mx(self):
        """Import MLX or skip."""
        try:
            import mlx.core as mx
            return mx
        except ImportError:
            pytest.skip("MLX not available")

    def test_store_cache_uses_snapshot_meta_for_rotating_cache(self, mx):
        """Boundary snapshot meta_state should override shared meta for RotatingKVCache blocks."""
        from omlx.cache.hybrid_cache import ModelCacheConfig

        block_size = 4
        paged_cache = PagedCacheManager(
            block_size=block_size,
            max_blocks=100,
            model_name="test-model",
            initial_blocks=100,
        )
        mock_ssd = MagicMock()
        mock_ssd.save_block.return_value = True

        model = MockModel(num_layers=2)
        cache = BlockAwarePrefixCache(
            model=model,
            paged_cache_manager=paged_cache,
            paged_ssd_cache_manager=mock_ssd,
        )

        # 8 tokens = 2 full blocks of 4
        tokens = list(range(8))

        # Simulate a hybrid model: layer 0 = KVCache, layer 1 = RotatingKVCache
        # Final cache state has offset=8 (end of request)
        cache_data = [
            {
                "state": (mx.ones((1, 4, 8, 64)), mx.ones((1, 4, 8, 64))),
                "cache_type": "KVCache",
                "class_name": "KVCache",
                "meta_state": ("8",),
            },
            {
                "state": (mx.ones((1, 1, 4, 256)), mx.ones((1, 1, 4, 256))),
                "cache_type": "RotatingKVCache",
                "class_name": "RotatingKVCache",
                "meta_state": ("0", "4", "8", "4"),  # keep, max_size, offset=8 (final), _idx
            },
        ]
        model_cache_config = ModelCacheConfig.from_type_list(
            ["KVCache", "RotatingKVCache"], model_name="test-model"
        )

        # Boundary snapshot at token 4 (end of block 1) with correct offset=4
        boundary_snapshots = {
            4: [
                {
                    "state": (),
                    "meta_state": (),
                    "class_name": "KVCache",
                    "cache_type": "KVCache",
                },
                {
                    "state": (mx.ones((1, 1, 4, 256)), mx.ones((1, 1, 4, 256))),
                    "meta_state": ("0", "4", "4", "4"),  # offset=4 at boundary
                    "class_name": "RotatingKVCache",
                    "cache_type": "RotatingKVCache",
                },
            ],
        }

        result = cache.store_cache(
            "req-001",
            tokens,
            cache_data,
            model_cache_config=model_cache_config,
            boundary_snapshots=boundary_snapshots,
        )

        assert result is not None
        assert len(result.block_ids) == 2

        # Verify save_block was called twice (one per block)
        assert mock_ssd.save_block.call_count == 2

        # Block 1 (has boundary snapshot): should use snapshot meta for
        # RotatingKVCache layer (offset=4), not shared meta (offset=8)
        block1_call = mock_ssd.save_block.call_args_list[0]
        block1_meta = block1_call.kwargs["layer_meta_states"]
        # RotatingKVCache meta (layer 1): offset should be 4 from snapshot
        assert block1_meta[1] == ("0", "4", "4", "4"), (
            f"Block 1 RotatingKVCache meta should use snapshot offset=4, "
            f"got {block1_meta[1]}"
        )

        # Block 2 (last block, uses main state): should use shared meta
        block2_call = mock_ssd.save_block.call_args_list[1]
        block2_meta = block2_call.kwargs["layer_meta_states"]
        # Last block has no separate boundary snapshot override (boundary at
        # token 8 matches the request end), so it uses the shared meta
        assert block2_meta[1] == ("0", "4", "8", "4"), (
            f"Block 2 RotatingKVCache meta should use shared meta offset=8, "
            f"got {block2_meta[1]}"
        )

    def test_store_cache_kvcache_meta_falls_back_to_shared(self, mx):
        """KVCache layers in boundary snapshots have empty meta, should fall back to shared."""
        from omlx.cache.hybrid_cache import ModelCacheConfig

        block_size = 4
        paged_cache = PagedCacheManager(
            block_size=block_size,
            max_blocks=100,
            model_name="test-model",
            initial_blocks=100,
        )
        mock_ssd = MagicMock()
        mock_ssd.save_block.return_value = True

        model = MockModel(num_layers=1)
        cache = BlockAwarePrefixCache(
            model=model,
            paged_cache_manager=paged_cache,
            paged_ssd_cache_manager=mock_ssd,
        )

        tokens = list(range(8))
        cache_data = [
            {
                "state": (mx.ones((1, 4, 8, 64)), mx.ones((1, 4, 8, 64))),
                "cache_type": "KVCache",
                "class_name": "KVCache",
                "meta_state": ("8",),
            },
        ]
        model_cache_config = ModelCacheConfig.from_type_list(
            ["KVCache"], model_name="test-model"
        )

        # KVCache layers have empty meta in boundary snapshots
        boundary_snapshots = {
            4: [
                {
                    "state": (),
                    "meta_state": (),
                    "class_name": "KVCache",
                    "cache_type": "KVCache",
                },
            ],
        }

        result = cache.store_cache(
            "req-001",
            tokens,
            cache_data,
            model_cache_config=model_cache_config,
            boundary_snapshots=boundary_snapshots,
        )

        assert result is not None
        assert mock_ssd.save_block.call_count == 2

        # Block 1: KVCache meta should fall back to shared meta (empty snapshot meta)
        block1_call = mock_ssd.save_block.call_args_list[0]
        block1_meta = block1_call.kwargs["layer_meta_states"]
        assert block1_meta[0] == ("8",), (
            f"KVCache should fall back to shared meta, got {block1_meta[0]}"
        )

    def test_store_cache_no_snapshot_uses_shared_meta(self, mx):
        """Blocks without boundary snapshots should use shared meta (existing behavior)."""
        block_size = 4
        paged_cache = PagedCacheManager(
            block_size=block_size,
            max_blocks=100,
            model_name="test-model",
            initial_blocks=100,
        )
        mock_ssd = MagicMock()
        mock_ssd.save_block.return_value = True

        model = MockModel(num_layers=1)
        cache = BlockAwarePrefixCache(
            model=model,
            paged_cache_manager=paged_cache,
            paged_ssd_cache_manager=mock_ssd,
        )

        tokens = list(range(4))
        cache_data = [
            {
                "state": (mx.ones((1, 4, 4, 64)), mx.ones((1, 4, 4, 64))),
                "cache_type": "KVCache",
                "class_name": "KVCache",
                "meta_state": ("4",),
            },
        ]

        # No boundary snapshots
        result = cache.store_cache("req-001", tokens, cache_data)

        assert result is not None
        assert mock_ssd.save_block.call_count == 1

        block_call = mock_ssd.save_block.call_args_list[0]
        block_meta = block_call.kwargs["layer_meta_states"]
        assert block_meta[0] == ("4",)

    def test_store_cache_last_block_with_snapshot_uses_snapshot_meta(self, mx):
        """Last block should also prefer snapshot meta when a boundary snapshot exists."""
        from omlx.cache.hybrid_cache import ModelCacheConfig

        block_size = 4
        paged_cache = PagedCacheManager(
            block_size=block_size,
            max_blocks=100,
            model_name="test-model",
            initial_blocks=100,
        )
        mock_ssd = MagicMock()
        mock_ssd.save_block.return_value = True

        # Hybrid model: KVCache + RotatingKVCache (need KVCache so
        # _get_cache_seq_len can determine the full sequence length)
        model = MockModel(num_layers=2)
        cache = BlockAwarePrefixCache(
            model=model,
            paged_cache_manager=paged_cache,
            paged_ssd_cache_manager=mock_ssd,
        )

        # 8 tokens = 2 blocks, but cache_data reflects processing of
        # 11 total tokens (3 trailing partial not stored).
        # Shared meta has offset=11 (final state) for RotatingKVCache.
        tokens = list(range(8))
        cache_data = [
            {
                "state": (mx.ones((1, 4, 11, 64)), mx.ones((1, 4, 11, 64))),
                "cache_type": "KVCache",
                "class_name": "KVCache",
                "meta_state": ("11",),
            },
            {
                "state": (mx.ones((1, 1, 4, 256)), mx.ones((1, 1, 4, 256))),
                "cache_type": "RotatingKVCache",
                "class_name": "RotatingKVCache",
                "meta_state": ("0", "4", "11", "4"),  # offset=11 (final)
            },
        ]
        model_cache_config = ModelCacheConfig.from_type_list(
            ["KVCache", "RotatingKVCache"], model_name="test-model"
        )

        # Both blocks have boundary snapshots with correct per-boundary offsets
        boundary_snapshots = {
            4: [
                {
                    "state": (),
                    "meta_state": (),
                    "class_name": "KVCache",
                    "cache_type": "KVCache",
                },
                {
                    "state": (mx.ones((1, 1, 4, 256)), mx.ones((1, 1, 4, 256))),
                    "meta_state": ("0", "4", "4", "4"),
                    "class_name": "RotatingKVCache",
                    "cache_type": "RotatingKVCache",
                },
            ],
            8: [
                {
                    "state": (),
                    "meta_state": (),
                    "class_name": "KVCache",
                    "cache_type": "KVCache",
                },
                {
                    "state": (mx.ones((1, 1, 4, 256)), mx.ones((1, 1, 4, 256))),
                    "meta_state": ("0", "4", "8", "4"),  # offset=8 at boundary
                    "class_name": "RotatingKVCache",
                    "cache_type": "RotatingKVCache",
                },
            ],
        }

        result = cache.store_cache(
            "req-001",
            tokens,
            cache_data,
            model_cache_config=model_cache_config,
            boundary_snapshots=boundary_snapshots,
        )

        assert result is not None
        assert mock_ssd.save_block.call_count == 2

        # Block 1: RotatingKVCache offset=4 from snapshot
        b1_meta = mock_ssd.save_block.call_args_list[0].kwargs["layer_meta_states"]
        assert b1_meta[1] == ("0", "4", "4", "4")

        # Block 2 (last): RotatingKVCache offset=8 from snapshot,
        # NOT offset=11 from shared meta
        b2_meta = mock_ssd.save_block.call_args_list[1].kwargs["layer_meta_states"]
        assert b2_meta[1] == ("0", "4", "8", "4"), (
            f"Last block should use snapshot offset=8, not shared offset=11, "
            f"got {b2_meta[1]}"
        )


class TestSubmitStoreCacheAsync:
    """Tests for the two-phase async store_cache that fixes the
    cold-prefill cache miss when Turn N+1's prefix lookup races Turn N's
    deferred SSD write (omlx commit 22440be regression)."""

    @pytest.fixture
    def mx(self):
        try:
            import mlx.core as mx
            return mx
        except ImportError:
            pytest.skip("MLX not available")

    @pytest.fixture
    def paged_cache(self):
        return PagedCacheManager(
            block_size=4,
            max_blocks=100,
            model_name="test-model",
            initial_blocks=100,
        )

    def _build_cache(self, paged_cache, save_block_return=True):
        """Build a BlockAwarePrefixCache wired to a mock SSD."""
        from unittest.mock import MagicMock
        mock_ssd = MagicMock()
        mock_ssd.save_block.return_value = save_block_return
        mock_ssd.load_block_with_metadata.return_value = (
            None, {"model_name": "test-model", "num_layers": 1},
        )
        model = MockModel(num_layers=1)
        cache = BlockAwarePrefixCache(
            model=model,
            paged_cache_manager=paged_cache,
            paged_ssd_cache_manager=mock_ssd,
        )
        return cache, mock_ssd

    def _build_cache_data(self, mx, num_tokens: int):
        """Two-token-dim KVCache state large enough for slicing tests."""
        return [
            {
                "state": (
                    mx.ones((1, 8, num_tokens, 64)),
                    mx.ones((1, 8, num_tokens, 64)),
                ),
                "cache_type": "KVCache",
                "class_name": "KVCache",
            }
        ]

    def test_phase1_registers_blocks_synchronously(self, paged_cache, mx):
        """fetch_cache MUST hit on the same prefix immediately after
        submit_store_cache_async returns, even before Phase 2 finishes —
        this is the whole point of the fix."""
        import concurrent.futures
        import threading

        cache, mock_ssd = self._build_cache(paged_cache)
        # Block save_block until we say so so Phase 2 stays in flight.
        gate = threading.Event()
        def _slow_save(**kwargs):
            gate.wait(timeout=10)
            return True
        mock_ssd.save_block.side_effect = _slow_save

        tokens = list(range(8))  # 2 full blocks
        cache_data = self._build_cache_data(mx, num_tokens=8)

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            block_table, future = cache.submit_store_cache_async(
                "req-1", tokens, cache_data, executor=executor,
            )
            assert block_table is not None
            assert len(block_table.block_ids) == 2
            assert future is not None
            assert not future.done(), "Phase 2 should still be in flight"

            # Phase 1 effect: a fresh fetch on the same prefix finds the blocks.
            hit_table, remaining = cache.fetch_cache("req-2", tokens)
            assert hit_table is not None, "fetch_cache should hit immediately"
            assert hit_table.num_tokens == 8
            assert remaining == []

            # Each newly allocated block carries the future.
            for bid in block_table.block_ids:
                blk = paged_cache.allocated_blocks[bid]
                assert blk.write_future is future
                assert blk.write_failed is False
        finally:
            gate.set()
            executor.shutdown(wait=True)

    def test_reconstruct_waits_for_pending_write(self, paged_cache, mx):
        """reconstruct_cache must block on block.write_future before
        attempting load_block_with_metadata."""
        import concurrent.futures
        import threading

        cache, mock_ssd = self._build_cache(paged_cache)
        gate = threading.Event()
        save_observed_at = []
        load_observed_at = []

        def _slow_save(**kwargs):
            save_observed_at.append(time.monotonic())
            gate.wait(timeout=10)
            return True
        def _load(block_hash):
            load_observed_at.append(time.monotonic())
            # After save unblocks, return a minimal-but-valid block payload.
            # reconstruct_cache only checks "block_data is None" to decide
            # the load failed; non-None is enough for the wait-then-load
            # ordering assertion we care about here.
            return ([{"state": (mx.ones((1, 8, 4, 64)), mx.ones((1, 8, 4, 64))),
                       "cache_type": "KVCache"}],
                    {"model_name": "test-model", "num_layers": 1,
                     "layer_cache_types": ["KVCache"], "layer_meta_states": [()]})
        mock_ssd.save_block.side_effect = _slow_save
        mock_ssd.load_block_with_metadata.side_effect = _load

        tokens = list(range(8))
        cache_data = self._build_cache_data(mx, num_tokens=8)
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            block_table, future = cache.submit_store_cache_async(
                "req-1", tokens, cache_data, executor=executor,
            )
            assert future is not None

            # Kick off reconstruct on a background thread; release the save gate
            # after a short wait so we can prove reconstruct waited.
            recon_done = threading.Event()
            def _recon():
                cache.reconstruct_cache(block_table)
                recon_done.set()
            t = threading.Thread(target=_recon, daemon=True)
            t.start()

            # Give reconstruct a moment to enter the wait.
            time.sleep(0.1)
            assert not recon_done.is_set(), "reconstruct must not return until write completes"

            gate.set()
            t.join(timeout=5)
            assert recon_done.is_set(), "reconstruct should finish once write is done"

            # save_block was observed before any load_block_with_metadata —
            # i.e. the wait actually held off the load.
            assert save_observed_at, "save_block should have been called"
            assert load_observed_at, "load_block_with_metadata should have been called"
            assert load_observed_at[0] >= save_observed_at[0], (
                "reconstruct loaded bytes before the save started — wait broke"
            )
        finally:
            gate.set()
            executor.shutdown(wait=True)

    def test_reconstruct_times_out_on_stuck_writer(self, paged_cache, mx, monkeypatch):
        """If the write_future never completes, reconstruct must hit the
        timeout, log loud, and treat the block as a missing prefix break."""
        import concurrent.futures
        from omlx.cache import prefix_cache as pc_mod

        # Shorten the wait so the test runs in <1s.
        monkeypatch.setattr(pc_mod, "WRITE_FUTURE_WAIT_S", 0.2)

        cache, mock_ssd = self._build_cache(paged_cache)

        # Build a block_table with one block tagged with a future that
        # never completes.
        tokens = list(range(4))
        block_table = paged_cache.create_block_table("req-stuck")
        block = paged_cache.allocate_block()
        block.token_count = 4
        block.block_hash = compute_block_hash(None, tokens, model_name="test-model")
        paged_cache.register_block_hash(block, tokens, None)
        block_table.block_ids.append(block.block_id)
        block_table.num_tokens = 4

        never_done = concurrent.futures.Future()
        block.write_future = never_done

        result = cache.reconstruct_cache(block_table)

        # Reconstruct breaks on the stuck block with no valid prefix preceding
        # it — same outcome as a missing block. ``load_block_with_metadata``
        # must NOT have been called (we'd have served stale/empty bytes).
        assert result is None, "stuck writer should not produce a reconstructed cache"
        mock_ssd.load_block_with_metadata.assert_not_called()
        # The future is still pending (we did not cancel it); cleanup.
        never_done.cancel()

    def test_reconstruct_treats_write_failed_as_missing(self, paged_cache, mx):
        """When Phase 2 marks block.write_failed=True (e.g. save_block
        returned False), reconstruct breaks at that block."""
        import concurrent.futures

        cache, mock_ssd = self._build_cache(paged_cache, save_block_return=False)
        tokens = list(range(8))
        cache_data = self._build_cache_data(mx, num_tokens=8)
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            block_table, future = cache.submit_store_cache_async(
                "req-fail", tokens, cache_data, executor=executor,
            )
            assert future is not None
            future.result(timeout=5)  # let Phase 2 finish

            for bid in block_table.block_ids:
                blk = paged_cache.allocated_blocks[bid]
                assert blk.write_failed is True

            # First failed block aborts reconstruction; with no preceding
            # valid blocks, reconstruct returns None.
            result = cache.reconstruct_cache(block_table)
            assert result is None
            mock_ssd.load_block_with_metadata.assert_not_called()
        finally:
            executor.shutdown(wait=True)

    def test_reconstruct_skips_wait_when_future_done(self, paged_cache, mx):
        """When write_future.done() is True, reconstruct must not call
        result() — steady-state lookups pay zero latency from this fix."""
        import concurrent.futures

        cache, mock_ssd = self._build_cache(paged_cache)
        tokens = list(range(8))
        cache_data = self._build_cache_data(mx, num_tokens=8)

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            block_table, future = cache.submit_store_cache_async(
                "req-1", tokens, cache_data, executor=executor,
            )
            assert future is not None
            future.result(timeout=5)  # Phase 2 done; future.done() == True

            # Replace each block's future with a sentinel whose .result()
            # would explode if called. .done() is the only legal check on
            # the wait path when state is settled — exercising .result()
            # would mean the fast-path short-circuit broke.
            class _ExplodingFuture:
                def done(self): return True
                def result(self, timeout=None):
                    raise AssertionError(
                        "reconstruct should not call result() on a done future"
                    )
            for bid in block_table.block_ids:
                paged_cache.allocated_blocks[bid].write_future = _ExplodingFuture()

            # Don't care whether downstream reconstruction succeeds (mocked
            # block bytes won't survive the real concatenate code path) —
            # only that reconstruct made it past the wait into the loader.
            cache.reconstruct_cache(block_table)
            assert mock_ssd.load_block_with_metadata.call_count >= 1, (
                "reconstruct never reached load_block_with_metadata — "
                "the .done() short-circuit appears to have broken"
            )
        finally:
            executor.shutdown(wait=True)

    def test_no_async_work_returns_none_future(self, paged_cache, mx):
        """When tokens fit in trailing-partial blocks only, no Phase 2
        work is scheduled and the returned future is None."""
        import concurrent.futures

        cache, _mock_ssd = self._build_cache(paged_cache)
        tokens = [1, 2, 3]  # < block_size=4 -> 0 full blocks
        cache_data = self._build_cache_data(mx, num_tokens=3)
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            block_table, future = cache.submit_store_cache_async(
                "req-1", tokens, cache_data, executor=executor,
            )
            assert block_table is not None
            assert len(block_table.block_ids) == 0
            assert future is None
        finally:
            executor.shutdown(wait=True)


class TestRegression_22440be_BackToBackPrefixCacheHit:
    """Regression test for the async store_cache race introduced by 22440be.

    Background: 22440be ("perf(cache): Async store_cache and SSD evict
    unlink off inference thread") moved the post-finish KV write to a
    worker thread — a real win on its own, ~96% reduction in main-thread
    store_cache time. But it also moved the IN-MEMORY block-hash
    registration to the worker thread. The next request's fetch_cache
    then raced the in-flight worker: if Request B started before Request
    A's worker finished, B's fetch_cache MISSED on a prefix that was
    logically written but not yet visible. B re-paid full cold prefill.

    The integration symptom was a 77% regression on the nanobot cold-chat
    e2e (197s → 348s). Both Turn 1 and Turn 2 of "hello" paid cold
    prefill cost because Turn 2's admission ran before Turn 1's worker
    had registered the cache — see commit message of the fix for details.

    The fix splits store_cache into Phase 1 (sync metadata register,
    runs on inference thread) and Phase 2 (async byte write, runs on
    executor). Phase 1 makes the prefix immediately visible to subsequent
    fetch_cache; Phase 2's per-block future is awaited inside
    reconstruct_cache before bytes are read.

    This test pins the post-fix contract end-to-end. It WILL FAIL if a
    future change moves the metadata register back behind the executor,
    skips the per-block wait in reconstruct, or otherwise regresses the
    "back-to-back requests hit prefix cache" guarantee.
    """

    @pytest.fixture
    def mx(self):
        try:
            import mlx.core as mx
            return mx
        except ImportError:
            pytest.skip("MLX not available")

    def test_request_b_hits_prefix_cache_while_request_a_write_inflight(self, mx):
        """The smoking-gun scenario: A finishes, A's bytes are still being
        written, B arrives with the same prefix, B must hit the cache."""
        import concurrent.futures
        import threading
        from unittest.mock import MagicMock

        block_size = 4
        paged_cache = PagedCacheManager(
            block_size=block_size,
            max_blocks=100,
            model_name="test-model",
            initial_blocks=100,
        )

        # A controllable SSD: writes block on a gate (simulates slow
        # worker), reads return realistic-ish payload metadata so we can
        # observe the load-vs-write ordering.
        write_gate = threading.Event()
        write_observed_at: list = []
        load_observed_at: list = []

        ssd = MagicMock()
        def _slow_save(**_kwargs):
            write_observed_at.append(time.monotonic())
            # Block until the test releases — simulates a worker that
            # hasn't finished writing bytes when B arrives.
            assert write_gate.wait(timeout=10), "test bug: gate never set"
            return True
        def _load(block_hash):
            load_observed_at.append(time.monotonic())
            # Returning None makes reconstruct_cache treat as missing —
            # which is exactly the failure mode we want to NOT see.
            # Return a non-None payload so reconstruct can complete.
            return (
                [
                    {
                        "state": (
                            mx.ones((1, 8, block_size, 64)),
                            mx.ones((1, 8, block_size, 64)),
                        ),
                        "cache_type": "KVCache",
                    }
                ],
                {
                    "model_name": "test-model",
                    "num_layers": 1,
                    "layer_cache_types": ["KVCache"],
                    "layer_meta_states": [()],
                },
            )
        ssd.save_block.side_effect = _slow_save
        ssd.load_block_with_metadata.side_effect = _load

        cache = BlockAwarePrefixCache(
            model=MockModel(num_layers=1),
            paged_cache_manager=paged_cache,
            paged_ssd_cache_manager=ssd,
        )

        # Request A finishes, hands its prefill cache to the worker.
        # Use a real ThreadPoolExecutor (max_workers=1, mirroring the
        # production scheduler) so Phase 2 lands on a different thread
        # and sits behind the gate.
        tokens = list(range(block_size * 2))  # 2 full blocks
        cache_data = [
            {
                "state": (
                    mx.ones((1, 8, len(tokens), 64)),
                    mx.ones((1, 8, len(tokens), 64)),
                ),
                "cache_type": "KVCache",
                "class_name": "KVCache",
            }
        ]
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="omlx-store-cache-test",
        )
        try:
            block_table_a, future_a = cache.submit_store_cache_async(
                "req-a", tokens, cache_data, executor=executor,
            )
            assert future_a is not None
            assert not future_a.done(), (
                "Phase 2 worker should be sitting on the write gate; if it "
                "already finished, the test no longer exercises the race"
            )

            # === Regression assertion #1 ===
            # Request B's admission. fetch_cache MUST hit on the same
            # prefix even though A's bytes haven't landed. Pre-fix code
            # missed here because metadata register was deferred to the
            # worker.
            block_table_b, remaining = cache.fetch_cache("req-b", tokens)
            assert block_table_b is not None, (
                "REGRESSION: fetch_cache missed the prefix that Request A "
                "wrote. Phase 1 metadata register is no longer synchronous — "
                "back-to-back chats will pay duplicate cold prefill."
            )
            assert block_table_b.num_tokens == len(tokens)
            assert remaining == []

            # === Regression assertion #2 ===
            # reconstruct_cache for B must wait for A's worker before
            # reading bytes. Run reconstruct in a background thread, prove
            # it's blocked on the gate, then release.
            recon_done = threading.Event()
            recon_result: list = []
            def _recon():
                recon_result.append(cache.reconstruct_cache(block_table_b))
                recon_done.set()
            t = threading.Thread(target=_recon, daemon=True)
            t.start()

            # Give reconstruct a moment to enter the wait. If the wait
            # was skipped, recon_done would already be set.
            time.sleep(0.1)
            assert not recon_done.is_set(), (
                "REGRESSION: reconstruct_cache returned before the in-flight "
                "write finished. Per-block wait on block.write_future is "
                "missing — Request B will read torn / empty bytes."
            )

            # Release the worker and let reconstruct complete.
            write_gate.set()
            t.join(timeout=5)
            assert recon_done.is_set(), "reconstruct should finish once write completes"

            # === Regression assertion #3 ===
            # Ordering: save happened before load. If load_observed_at[0]
            # < write_observed_at[0], the wait broke and reconstruct
            # racy-read the SSD before bytes were there. We don't assert
            # the final reconstructed cache is non-None — the mocked
            # block payload doesn't survive MLX's real concatenate, and
            # the load ordering above is the actual race signal.
            assert write_observed_at, "save_block was never called"
            assert load_observed_at, "load_block_with_metadata was never called"
            assert load_observed_at[0] >= write_observed_at[0], (
                "REGRESSION: reconstruct loaded SSD bytes BEFORE the write "
                "started. The wait on block.write_future is broken or skipped."
            )
        finally:
            write_gate.set()
            executor.shutdown(wait=True)

    def test_legacy_store_cache_via_executor_misses_fetch_window(self, mx):
        """Demonstrate that the PRE-FIX pattern (store_cache submitted
        whole-hog to a worker thread) WOULD lose the fetch_cache window
        — confirms this regression test class actually catches the bug.

        Replays the historical broken behavior: submit ``store_cache``
        (the original method, single function) to the executor; while it
        sits behind a gate, fetch_cache for the same prefix returns a
        MISS. This is exactly the failure mode 22440be introduced and
        ``submit_store_cache_async`` fixed.

        If a future change tries to "simplify" by routing back through
        the single-function store_cache on the executor, the regression
        test above catches the resulting cache miss.
        """
        import concurrent.futures
        import threading
        from unittest.mock import MagicMock

        block_size = 4
        paged_cache = PagedCacheManager(
            block_size=block_size,
            max_blocks=100,
            model_name="test-model",
            initial_blocks=100,
        )
        write_gate = threading.Event()
        ssd = MagicMock()
        def _slow_save(**_kwargs):
            assert write_gate.wait(timeout=10)
            return True
        ssd.save_block.side_effect = _slow_save
        cache = BlockAwarePrefixCache(
            model=MockModel(num_layers=1),
            paged_cache_manager=paged_cache,
            paged_ssd_cache_manager=ssd,
        )

        tokens = list(range(block_size * 2))
        cache_data = [
            {
                "state": (
                    mx.ones((1, 8, len(tokens), 64)),
                    mx.ones((1, 8, len(tokens), 64)),
                ),
                "cache_type": "KVCache",
                "class_name": "KVCache",
            }
        ]
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            # Pre-fix pattern: whole-hog submit. Metadata register is
            # blocked behind the gate inside the worker.
            future = executor.submit(
                cache.store_cache, "req-a", tokens, cache_data,
            )
            # Give the worker a moment to enter the gate.
            time.sleep(0.05)
            assert not future.done(), "worker should be sitting on the gate"

            # Simulate Request B's admission: same prefix, different ID.
            block_table_b, remaining_b = cache.fetch_cache("req-b", tokens)

            # This is the bug 22440be introduced. store_cache walks
            # blocks sequentially: register hash 1 -> save 1 -> register
            # hash 2 -> save 2 -> ... With save 1 stuck on the gate,
            # only block 1's hash made it into the index. fetch_cache
            # then partial-hits block 1 and misses everything after,
            # so Request B has to re-prefill most of the prefix. The
            # post-fix submit_store_cache_async registers ALL hashes up
            # front before any save runs — fetch_cache hits the full
            # prefix even mid-write.
            hit_tokens = block_table_b.num_tokens if block_table_b else 0
            assert hit_tokens < len(tokens), (
                f"Sanity check: legacy pattern should NOT hit the full "
                f"prefix while the worker is still mid-write. Got {hit_tokens}"
                f"/{len(tokens)} tokens — if this is the full prefix, the "
                f"executor scheduled both register+save eagerly enough that "
                f"the race window closed; adjust the gate timing."
            )
        finally:
            write_gate.set()
            executor.shutdown(wait=True)
