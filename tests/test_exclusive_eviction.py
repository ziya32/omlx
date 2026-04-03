# SPDX-License-Identifier: Apache-2.0
"""Tests for exclusive pinned model eviction in EnginePool.

Covers: _clear_for_exclusive(), _refresh_vision_limits(), _signal_exclusive_idle(),
acquire_engine(), release_engine(), exclusive_max_hold expiry, and the event
capture race condition.
"""

import asyncio
import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omlx.engine_pool import EngineEntry, EnginePool, EngineState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_mock_model_dir(tmp_path):
    """Create a mock model directory with two small models."""
    model_a = tmp_path / "model-a"
    model_a.mkdir()
    (model_a / "config.json").write_text(json.dumps({"model_type": "llama"}))
    (model_a / "model.safetensors").write_bytes(b"0" * 1024)

    model_b = tmp_path / "model-b"
    model_b.mkdir()
    (model_b / "config.json").write_text(json.dumps({"model_type": "qwen"}))
    (model_b / "model.safetensors").write_bytes(b"0" * 2048)

    return tmp_path


@pytest.fixture
def pool_ab(small_mock_model_dir):
    """EnginePool with model-a and model-b discovered (nothing loaded)."""
    pool = EnginePool(max_model_memory=10 * 1024**3)
    pool.discover_models(str(small_mock_model_dir))
    return pool


def _make_mock_engine(*, has_active_work: bool = False):
    """Create a lightweight mock engine that looks real enough for the pool."""
    engine = MagicMock()
    engine.start = AsyncMock()
    engine.stop = AsyncMock()
    # _engine_has_active_work checks _engine._engine.engine._output_collectors
    inner = MagicMock()
    inner._output_collectors = {"fake": True} if has_active_work else {}
    inner._step_in_flight = False
    core = MagicMock()
    core.engine = inner
    engine._engine = core
    return engine


def _activate_entry(entry: EngineEntry, *, engine=None, pinned=False,
                    exclusive=False, exclusive_max_hold=0):
    """Put an entry into ACTIVE state with the given flags."""
    entry.engine = engine or _make_mock_engine()
    entry.state = EngineState.ACTIVE
    entry.last_access = time.time()
    entry.is_pinned = pinned
    entry.exclusive = exclusive
    entry.exclusive_max_hold = exclusive_max_hold


# ---------------------------------------------------------------------------
# Test 1: Idle eviction
# ---------------------------------------------------------------------------


class TestIdleEviction:
    """When an exclusive+pinned model gets a request, idle non-pinned models
    are immediately unloaded."""

    async def test_get_engine_evicts_idle_non_pinned(self, pool_ab):
        """Design test #1: idle non-pinned model B is unloaded when
        exclusive model A gets a request via get_engine()."""
        pool = pool_ab
        entry_a = pool.get_entry("model-a")
        entry_b = pool.get_entry("model-b")

        _activate_entry(entry_a, pinned=True, exclusive=True)
        _activate_entry(entry_b, pinned=False)

        # get_engine("model-a") should call _clear_for_exclusive which
        # evicts model-b (idle, no active work).
        engine = await pool.get_engine("model-a")

        assert engine is entry_a.engine
        # model-b should be evicted (UNLOADING or UNLOADED, engine is None)
        assert entry_b.engine is None
        assert entry_b.state in (EngineState.UNLOADING, EngineState.UNLOADED)


# ---------------------------------------------------------------------------
# Test 2: Drain eviction
# ---------------------------------------------------------------------------


class TestDrainEviction:
    """When a non-pinned model is busy (active_uses > 0), _clear_for_exclusive
    starts a drain instead of immediate unload."""

    async def test_clear_for_exclusive_drains_busy_model(self, pool_ab):
        """Design test #2: busy model B enters DRAINING when exclusive A
        calls _clear_for_exclusive()."""
        pool = pool_ab
        entry_a = pool.get_entry("model-a")
        entry_b = pool.get_entry("model-b")

        _activate_entry(entry_a, pinned=True, exclusive=True)
        _activate_entry(entry_b, pinned=False)
        entry_b.active_uses = 1  # simulate busy model

        async with pool._tracked_lock("test"):
            result = await pool._clear_for_exclusive(entry_a)

        assert result is not None, "Should return a drain event"
        assert entry_b.state == EngineState.DRAINING
        assert entry_b.drain_complete is not None
        assert result is entry_b.drain_complete


# ---------------------------------------------------------------------------
# Test 3: Non-exclusive pinned does NOT evict
# ---------------------------------------------------------------------------


class TestNonExclusivePinnedNoEvict:
    """A pinned model that is NOT exclusive should not evict other models."""

    async def test_non_exclusive_pinned_no_eviction(self, pool_ab):
        """Design test #7: pinned but NOT exclusive model A does not
        evict model B on get_engine()."""
        pool = pool_ab
        entry_a = pool.get_entry("model-a")
        entry_b = pool.get_entry("model-b")

        _activate_entry(entry_a, pinned=True, exclusive=False)
        _activate_entry(entry_b, pinned=False)

        saved_engine_b = entry_b.engine

        engine = await pool.get_engine("model-a")

        assert engine is entry_a.engine
        # model-b should still be ACTIVE with its engine intact
        assert entry_b.state == EngineState.ACTIVE
        assert entry_b.engine is saved_engine_b


# ---------------------------------------------------------------------------
# Test 4: No-op when clean
# ---------------------------------------------------------------------------


class TestNoOpWhenClean:
    """When there are no non-pinned models to evict, _clear_for_exclusive
    returns None immediately."""

    async def test_clear_for_exclusive_noop(self, pool_ab):
        """Design test #6: only model A (pinned+exclusive) is ACTIVE,
        model B is UNLOADED. _clear_for_exclusive returns None."""
        pool = pool_ab
        entry_a = pool.get_entry("model-a")
        entry_b = pool.get_entry("model-b")

        _activate_entry(entry_a, pinned=True, exclusive=True)
        # entry_b stays UNLOADED (default)

        async with pool._tracked_lock("test"):
            result = await pool._clear_for_exclusive(entry_a)

        assert result is None


# ---------------------------------------------------------------------------
# Test 5: Wait on exclusive_idle in _check_process_memory
# ---------------------------------------------------------------------------


class TestExclusiveIdleWait:
    """_check_process_memory should find and return the exclusive_idle event
    when an exclusive model has active_uses > 0."""

    async def test_check_process_memory_returns_exclusive_idle(self, pool_ab):
        """Design test #8: _check_process_memory finds exclusive model A's
        exclusive_idle event when memory is tight."""
        pool = pool_ab
        entry_a = pool.get_entry("model-a")
        entry_b = pool.get_entry("model-b")

        _activate_entry(entry_a, pinned=True, exclusive=True)
        entry_a.active_uses = 1
        entry_a.exclusive_idle = asyncio.Event()

        # entry_b is UNLOADED but we want to "load" it — need a process
        # memory enforcer that says memory is too tight.
        enforcer = MagicMock()
        enforcer.max_bytes = 500  # artificially small
        pool._process_memory_enforcer = enforcer

        # Mock mx.get_active_memory to report memory that exceeds the limit
        with patch("omlx.engine_pool.mx") as mock_mx:
            mock_mx.get_active_memory.return_value = 400
            # entry_b has estimated_size > 0 (from safetensors), so projected > 500

            async with pool._tracked_lock("test"):
                result = await pool._check_process_memory(entry_b)

        assert result is entry_a.exclusive_idle


# ---------------------------------------------------------------------------
# Test 6: Event capture race (THE MOST CRITICAL TEST)
# ---------------------------------------------------------------------------


class TestEventCaptureRace:
    """release_engine() must capture the exclusive_idle event at call time,
    not at signal time. A new acquire_engine() may replace the event before
    the async signal task runs."""

    async def test_release_captures_event_before_new_acquire(self, pool_ab):
        """Design test #19: verify old event is signaled, not the new one."""
        pool = pool_ab
        entry_a = pool.get_entry("model-a")
        _activate_entry(entry_a, pinned=True, exclusive=True)

        # Simulate: model A has 1 active request with an exclusive_idle event
        old_event = asyncio.Event()
        entry_a.exclusive_idle = old_event
        entry_a.active_uses = 1
        entry_a._exclusive_hold_start = time.time()

        # Patch _signal_exclusive_idle to just set the event (skip Metal ops)
        async def _mock_signal(event):
            if event is not None:
                event.set()

        with patch.object(pool, "_signal_exclusive_idle", side_effect=_mock_signal):
            # release_engine decrements active_uses to 0 and fires async signal
            pool.release_engine("model-a")

            # IMMEDIATELY acquire again — this creates a NEW event
            pool.acquire_engine("model-a")
            new_event = entry_a.exclusive_idle

        assert new_event is not old_event, "acquire should create a fresh event"

        # Let the signal task run
        await asyncio.sleep(0.05)

        # The OLD event should be signaled (the one captured by release_engine)
        assert old_event.is_set(), "Old event should be signaled"
        # The NEW event should NOT be set (created by the subsequent acquire)
        assert not new_event.is_set(), "New event should NOT be signaled"


# ---------------------------------------------------------------------------
# Test 7: exclusive_max_hold expiry
# ---------------------------------------------------------------------------


class TestExclusiveMaxHoldExpiry:
    """When exclusive_max_hold is set and the hold time has expired,
    get_engine should NOT call _clear_for_exclusive."""

    async def test_expired_hold_skips_eviction(self, pool_ab):
        """Design test #13: after max_hold expires, non-pinned model B
        is NOT evicted."""
        pool = pool_ab
        entry_a = pool.get_entry("model-a")
        entry_b = pool.get_entry("model-b")

        _activate_entry(entry_a, pinned=True, exclusive=True, exclusive_max_hold=1)
        _activate_entry(entry_b, pinned=False)

        # Simulate: hold started more than 1 second ago
        entry_a._exclusive_hold_start = time.time() - 5.0
        entry_a.active_uses = 1  # currently in use (hold is active)

        saved_engine_b = entry_b.engine

        engine = await pool.get_engine("model-a")

        assert engine is entry_a.engine
        # model-b should NOT have been evicted
        assert entry_b.state == EngineState.ACTIVE
        assert entry_b.engine is saved_engine_b


# ---------------------------------------------------------------------------
# Test 8: Hold timer reset
# ---------------------------------------------------------------------------


class TestHoldTimerReset:
    """When active_uses drops to 0, _exclusive_hold_start should be reset."""

    async def test_release_resets_hold_start(self, pool_ab):
        """Design test #14: release_engine resets _exclusive_hold_start to 0."""
        pool = pool_ab
        entry_a = pool.get_entry("model-a")
        _activate_entry(entry_a, pinned=True, exclusive=True, exclusive_max_hold=60)

        # Simulate: model A has 1 active request
        entry_a.active_uses = 1
        entry_a._exclusive_hold_start = time.time()
        entry_a.exclusive_idle = asyncio.Event()

        assert entry_a._exclusive_hold_start > 0

        pool.release_engine("model-a")

        assert entry_a.active_uses == 0
        assert entry_a._exclusive_hold_start == 0.0


# ---------------------------------------------------------------------------
# Test 9: _refresh_vision_limits uses committed memory
# ---------------------------------------------------------------------------


class TestRefreshVisionLimits:
    """_refresh_vision_limits should compute headroom from _committed_memory(),
    not from mx.get_active_memory()."""

    async def test_refresh_uses_committed_memory(self, pool_ab):
        """Design test #4: vision_chunk_budget_pixels is set based on
        committed memory and the process memory enforcer's max_bytes."""
        pool = pool_ab
        entry_a = pool.get_entry("model-a")

        mock_engine = _make_mock_engine()
        mock_engine.vision_chunk_budget_pixels = 0  # will be updated
        _activate_entry(entry_a, pinned=True, exclusive=True, engine=mock_engine)

        # Set up process memory enforcer
        enforcer = MagicMock()
        enforcer.max_bytes = 10 * 1024**3  # 10GB
        pool._process_memory_enforcer = enforcer

        # committed_memory = entry_a.estimated_size (1024 bytes — tiny)
        # headroom = 10GB - 1024 ~= 10GB
        # vision_chunk_budget_pixels = headroom * 0.7 / 700
        async with pool._tracked_lock("test"):
            pool._refresh_vision_limits(entry_a)

        committed = pool._committed_memory()
        expected_headroom = enforcer.max_bytes - committed
        expected_pixels = int(
            expected_headroom * pool._VISION_SAFETY_FACTOR
            / pool._VISION_BYTES_PER_PIXEL
        )

        assert mock_engine.vision_chunk_budget_pixels == expected_pixels
        assert mock_engine.vision_chunk_budget_pixels > 0

    async def test_refresh_noop_for_non_vlm(self, pool_ab):
        """_refresh_vision_limits is a no-op if engine has no
        vision_chunk_budget_pixels attribute."""
        pool = pool_ab
        entry_a = pool.get_entry("model-a")

        mock_engine = _make_mock_engine()
        # Do NOT set vision_chunk_budget_pixels — simulate a non-VLM engine
        if hasattr(mock_engine, "vision_chunk_budget_pixels"):
            delattr(mock_engine, "vision_chunk_budget_pixels")
        _activate_entry(entry_a, pinned=True, exclusive=True, engine=mock_engine)

        enforcer = MagicMock()
        enforcer.max_bytes = 10 * 1024**3
        pool._process_memory_enforcer = enforcer

        async with pool._tracked_lock("test"):
            # Should not raise
            pool._refresh_vision_limits(entry_a)

    async def test_refresh_noop_without_enforcer(self, pool_ab):
        """_refresh_vision_limits is a no-op if no process memory enforcer."""
        pool = pool_ab
        entry_a = pool.get_entry("model-a")

        mock_engine = _make_mock_engine()
        mock_engine.vision_chunk_budget_pixels = 999
        _activate_entry(entry_a, pinned=True, exclusive=True, engine=mock_engine)

        # No enforcer set
        pool._process_memory_enforcer = None

        async with pool._tracked_lock("test"):
            pool._refresh_vision_limits(entry_a)

        # Should remain unchanged — no enforcer, no update
        assert mock_engine.vision_chunk_budget_pixels == 999


# ---------------------------------------------------------------------------
# Additional edge-case tests
# ---------------------------------------------------------------------------


class TestAcquireReleaseBasics:
    """Basic acquire_engine / release_engine semantics for exclusive models."""

    def test_acquire_creates_exclusive_idle_event(self, pool_ab):
        """acquire_engine on exclusive model creates exclusive_idle event
        when active_uses transitions 0 -> 1."""
        pool = pool_ab
        entry_a = pool.get_entry("model-a")
        _activate_entry(entry_a, pinned=True, exclusive=True)
        entry_a.active_uses = 0
        entry_a.exclusive_idle = None

        pool.acquire_engine("model-a")

        assert entry_a.active_uses == 1
        assert entry_a.exclusive_idle is not None
        assert not entry_a.exclusive_idle.is_set()
        assert entry_a._exclusive_hold_start > 0

    def test_acquire_does_not_replace_event_when_busy(self, pool_ab):
        """acquire_engine does NOT replace exclusive_idle when
        active_uses > 0 (already busy)."""
        pool = pool_ab
        entry_a = pool.get_entry("model-a")
        _activate_entry(entry_a, pinned=True, exclusive=True)

        entry_a.active_uses = 1
        original_event = asyncio.Event()
        entry_a.exclusive_idle = original_event

        pool.acquire_engine("model-a")

        assert entry_a.active_uses == 2
        assert entry_a.exclusive_idle is original_event

    def test_release_no_signal_when_still_busy(self, pool_ab):
        """release_engine does NOT signal exclusive_idle when
        active_uses > 0 after decrement."""
        pool = pool_ab
        entry_a = pool.get_entry("model-a")
        _activate_entry(entry_a, pinned=True, exclusive=True)

        entry_a.active_uses = 2
        entry_a._exclusive_hold_start = time.time()
        event = asyncio.Event()
        entry_a.exclusive_idle = event

        pool.release_engine("model-a")

        assert entry_a.active_uses == 1
        assert entry_a._exclusive_hold_start > 0  # not reset
        assert not event.is_set()

    def test_acquire_noop_for_non_exclusive(self, pool_ab):
        """acquire_engine on non-exclusive model does not create
        exclusive_idle event."""
        pool = pool_ab
        entry_a = pool.get_entry("model-a")
        _activate_entry(entry_a, pinned=True, exclusive=False)
        entry_a.active_uses = 0

        pool.acquire_engine("model-a")

        assert entry_a.active_uses == 1
        assert entry_a.exclusive_idle is None


class TestClearForExclusiveEdgeCases:
    """Edge cases for _clear_for_exclusive."""

    async def test_skips_other_pinned_models(self, pool_ab):
        """_clear_for_exclusive skips pinned models (even if not exclusive)."""
        pool = pool_ab
        entry_a = pool.get_entry("model-a")
        entry_b = pool.get_entry("model-b")

        _activate_entry(entry_a, pinned=True, exclusive=True)
        _activate_entry(entry_b, pinned=True, exclusive=False)

        async with pool._tracked_lock("test"):
            result = await pool._clear_for_exclusive(entry_a)

        assert result is None  # nothing to evict
        assert entry_b.state == EngineState.ACTIVE
        assert entry_b.engine is not None

    async def test_returns_draining_event_if_already_draining(self, pool_ab):
        """If a non-pinned model is already DRAINING, _clear_for_exclusive
        returns its drain_complete event."""
        pool = pool_ab
        entry_a = pool.get_entry("model-a")
        entry_b = pool.get_entry("model-b")

        _activate_entry(entry_a, pinned=True, exclusive=True)
        # Put entry_b in DRAINING state manually
        entry_b.state = EngineState.DRAINING
        entry_b.engine = _make_mock_engine()
        drain_event = asyncio.Event()
        entry_b.drain_complete = drain_event

        async with pool._tracked_lock("test"):
            result = await pool._clear_for_exclusive(entry_a)

        assert result is drain_event

    async def test_returns_unloading_event_if_already_unloading(self, pool_ab):
        """If a non-pinned model is already UNLOADING, _clear_for_exclusive
        returns its unload_complete event."""
        pool = pool_ab
        entry_a = pool.get_entry("model-a")
        entry_b = pool.get_entry("model-b")

        _activate_entry(entry_a, pinned=True, exclusive=True)
        # Put entry_b in UNLOADING state manually
        entry_b.state = EngineState.UNLOADING
        unload_event = asyncio.Event()
        entry_b.unload_complete = unload_event

        async with pool._tracked_lock("test"):
            result = await pool._clear_for_exclusive(entry_a)

        assert result is unload_event


class TestSetExclusive:
    """Tests for the set_exclusive() method."""

    def test_set_exclusive_success(self, pool_ab):
        """set_exclusive sets the exclusive flag and max_hold."""
        pool = pool_ab
        assert pool.set_exclusive("model-a", True, max_hold=120)
        entry = pool.get_entry("model-a")
        assert entry.exclusive is True
        assert entry.exclusive_max_hold == 120

    def test_set_exclusive_disable(self, pool_ab):
        """set_exclusive(False) clears the exclusive flag."""
        pool = pool_ab
        pool.set_exclusive("model-a", True, max_hold=60)
        pool.set_exclusive("model-a", False)
        entry = pool.get_entry("model-a")
        assert entry.exclusive is False
        assert entry.exclusive_max_hold == 0

    def test_set_exclusive_missing_model(self, pool_ab):
        """set_exclusive returns False for unknown model."""
        pool = pool_ab
        assert pool.set_exclusive("nonexistent", True) is False


# ---------------------------------------------------------------------------
# Gap test 1: apply_settings_overrides wiring
# ---------------------------------------------------------------------------


class TestApplySettingsOverridesWiring:
    """Verify apply_settings_overrides sets and clears exclusive fields."""

    def test_apply_exclusive_true(self, pool_ab):
        """apply_settings_overrides with exclusive=True sets entry fields."""
        from omlx.model_settings import ModelSettings, ModelSettingsManager

        pool = pool_ab
        entry_a = pool.get_entry("model-a")
        assert entry_a.exclusive is False
        assert entry_a.exclusive_max_hold == 0

        # Create a mock settings manager that returns exclusive settings
        mgr = MagicMock(spec=ModelSettingsManager)
        settings = ModelSettings(exclusive=True, exclusive_max_hold=30)
        mgr.get_settings.return_value = settings

        pool.apply_settings_overrides(mgr)

        assert entry_a.exclusive is True
        assert entry_a.exclusive_max_hold == 30

    def test_apply_exclusive_false_clears(self, pool_ab):
        """apply_settings_overrides with exclusive=False clears entry fields."""
        from omlx.model_settings import ModelSettings, ModelSettingsManager

        pool = pool_ab
        entry_a = pool.get_entry("model-a")

        # First set exclusive=True
        entry_a.exclusive = True
        entry_a.exclusive_max_hold = 60

        # Now apply settings with exclusive=False
        mgr = MagicMock(spec=ModelSettingsManager)
        settings = ModelSettings(exclusive=False, exclusive_max_hold=0)
        mgr.get_settings.return_value = settings

        pool.apply_settings_overrides(mgr)

        assert entry_a.exclusive is False
        assert entry_a.exclusive_max_hold == 0


# ---------------------------------------------------------------------------
# Gap test 2: Mixed idle + busy models in _clear_for_exclusive
# ---------------------------------------------------------------------------


class TestClearForExclusiveMixedIdleBusy:
    """_clear_for_exclusive with both idle and busy non-pinned models."""

    async def test_idle_evicted_busy_drained(self, small_mock_model_dir):
        """B (idle) is evicted immediately, C (busy) is drained."""
        # Need a pool with 3 models
        model_c = small_mock_model_dir / "model-c"
        model_c.mkdir()
        (model_c / "config.json").write_text(json.dumps({"model_type": "llama"}))
        (model_c / "model.safetensors").write_bytes(b"0" * 512)

        pool = EnginePool(max_model_memory=10 * 1024**3)
        pool.discover_models(str(small_mock_model_dir))

        entry_a = pool.get_entry("model-a")
        entry_b = pool.get_entry("model-b")
        entry_c = pool.get_entry("model-c")

        _activate_entry(entry_a, pinned=True, exclusive=True)
        _activate_entry(entry_b, pinned=False)  # idle
        _activate_entry(entry_c, pinned=False)  # will be busy
        entry_c.active_uses = 1

        async with pool._tracked_lock("test"):
            result = await pool._clear_for_exclusive(entry_a)

        # B should be evicted (idle -> UNLOADING or UNLOADED, engine=None)
        assert entry_b.engine is None
        assert entry_b.state in (EngineState.UNLOADING, EngineState.UNLOADED)

        # C should be draining (busy -> DRAINING)
        assert entry_c.state == EngineState.DRAINING

        # The method returns C's drain_complete event
        assert result is entry_c.drain_complete


# ---------------------------------------------------------------------------
# Gap test 3: Wake and load scenario (design test #9)
# ---------------------------------------------------------------------------


class TestWakeAndLoad:
    """Exclusive model A finishes, non-pinned B wakes and can proceed."""

    async def test_exclusive_idle_event_allows_waiter(self, pool_ab):
        """_check_process_memory returns exclusive_idle; signaling it
        unblocks the waiter."""
        pool = pool_ab
        entry_a = pool.get_entry("model-a")
        entry_b = pool.get_entry("model-b")

        _activate_entry(entry_a, pinned=True, exclusive=True)
        entry_a.active_uses = 1
        entry_a.exclusive_idle = asyncio.Event()

        # Set up process memory enforcer with tight limits
        enforcer = MagicMock()
        enforcer.max_bytes = 500  # artificially small
        pool._process_memory_enforcer = enforcer

        with patch("omlx.engine_pool.mx") as mock_mx:
            mock_mx.get_active_memory.return_value = 400

            async with pool._tracked_lock("test"):
                result = await pool._check_process_memory(entry_b)

        # Should return A's exclusive_idle event
        assert result is entry_a.exclusive_idle
        assert not result.is_set()

        # Signal: A finishes its request
        entry_a.exclusive_idle.set()

        # The waiter can now proceed
        assert result.is_set()


# ---------------------------------------------------------------------------
# Gap test 4: Timeout gives proper error type (design test #12)
# ---------------------------------------------------------------------------


class TestTimeoutErrorType:
    """When exclusive_idle wait times out, the error is ModelLoadingError
    (HTTP 504), not InsufficientMemoryError (HTTP 507)."""

    async def test_exclusive_wait_timeout_raises_model_loading_error(
        self, pool_ab
    ):
        """get_engine() timeout on exclusive_idle raises ModelLoadingError."""
        from omlx.exceptions import ModelLoadingError

        pool = pool_ab
        pool._max_wait_timeout = 0.1  # very short timeout

        entry_a = pool.get_entry("model-a")
        entry_b = pool.get_entry("model-b")

        _activate_entry(entry_a, pinned=True, exclusive=True)
        entry_a.active_uses = 1
        entry_a.exclusive_idle = asyncio.Event()  # never set

        # Set up process memory enforcer with tight limits so
        # _check_process_memory returns the exclusive_idle event
        enforcer = MagicMock()
        enforcer.max_bytes = 500
        pool._process_memory_enforcer = enforcer

        with patch("omlx.engine_pool.mx") as mock_mx:
            mock_mx.get_active_memory.return_value = 400

            with pytest.raises(ModelLoadingError):
                await pool.get_engine("model-b")


# ---------------------------------------------------------------------------
# Gap test 5: set_exclusive(False) cleans up state
# ---------------------------------------------------------------------------


class TestSetExclusiveCleansUpState:
    """set_exclusive(False) clears exclusive_idle and _exclusive_hold_start."""

    def test_clears_orphaned_state(self, pool_ab):
        """Calling set_exclusive(model_id, False) clears exclusive_idle
        and _exclusive_hold_start."""
        pool = pool_ab
        entry_a = pool.get_entry("model-a")
        _activate_entry(entry_a, pinned=True, exclusive=True)

        # Simulate mid-use state
        entry_a.exclusive_idle = asyncio.Event()
        entry_a._exclusive_hold_start = time.time()

        pool.set_exclusive("model-a", False)

        assert entry_a.exclusive is False
        assert entry_a.exclusive_max_hold == 0
        assert entry_a.exclusive_idle is None
        assert entry_a._exclusive_hold_start == 0.0


# ---------------------------------------------------------------------------
# Gap test 6: _signal_exclusive_idle calls mx operations before signaling
# ---------------------------------------------------------------------------


class TestSignalExclusiveIdleMxOps:
    """_signal_exclusive_idle calls mx.synchronize and mx.clear_cache,
    then sets the event."""

    async def test_mx_ops_called_and_event_set(self, pool_ab):
        """mx.synchronize and mx.clear_cache are called, and the event
        is set."""
        pool = pool_ab
        event = asyncio.Event()

        with patch("omlx.engine_pool.mx") as mock_mx, \
             patch("omlx.engine_pool.get_mlx_executor") as mock_get_exec:
            # Make run_in_executor actually run the lambda
            mock_executor = MagicMock()

            async def fake_run_in_executor(executor, fn):
                fn()

            loop = asyncio.get_running_loop()
            original_run_in_executor = loop.run_in_executor

            async def patched_run_in_executor(executor, fn):
                fn()

            with patch.object(loop, "run_in_executor", side_effect=patched_run_in_executor):
                await pool._signal_exclusive_idle(event)

        # mx.synchronize and mx.clear_cache should have been called
        assert mock_mx.synchronize.called
        assert mock_mx.clear_cache.called

        # Event should be set
        assert event.is_set()

    async def test_event_set_even_on_mx_failure(self, pool_ab):
        """Event is set even if mx operations raise an exception
        (due to the finally block)."""
        pool = pool_ab
        event = asyncio.Event()

        with patch("omlx.engine_pool.mx") as mock_mx, \
             patch("omlx.engine_pool.get_mlx_executor") as mock_get_exec:

            loop = asyncio.get_running_loop()

            async def failing_run_in_executor(executor, fn):
                raise RuntimeError("Metal error")

            with patch.object(loop, "run_in_executor", side_effect=failing_run_in_executor):
                await pool._signal_exclusive_idle(event)

        # Event should still be set (finally block)
        assert event.is_set()
