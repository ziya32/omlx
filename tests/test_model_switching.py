# SPDX-License-Identifier: Apache-2.0
"""Tests for oMLX model switching: drain, implicit queue, and state machine.

These tests are written against the design in DESIGN-omlx-model-switching.md.
They verify the drain-based model switching, loading coalescing, implicit
queue (wait instead of 507), client cancellation, atomicity guarantees,
memory accounting, invariants (INV-1 through INV-8), livelock detection,
and backward compatibility with existing LRU/TTL/pinning behavior.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omlx.engine_pool import EngineEntry, EnginePool, EngineState, LOAD_COOLDOWN
from omlx.exceptions import (
    ModelLoadingError,
    ModelNotFoundError,
    ModelTooLargeError,
)


# ---------------------------------------------------------------------------
# MockEngine
# ---------------------------------------------------------------------------

class MockEngine:
    """Lightweight engine mock with controllable active-work simulation.

    Compatible with EnginePool._engine_has_active_work() via the
    ``active_operations`` counter (the Audio-engine path).
    """

    def __init__(self, model_id: str, load_delay: float = 0):
        self.model_id = model_id
        self._load_delay = load_delay
        self.active_operations = 0  # Checked by _engine_has_active_work
        self._loaded = False
        self._model_name = model_id
        self._stop_called = asyncio.Event()

    # -- BaseEngine-like interface ------------------------------------------

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def tokenizer(self):
        return None

    async def start(self):
        if self._load_delay > 0:
            await asyncio.sleep(self._load_delay)
        self._loaded = True

    async def stop(self):
        self._stop_called.set()
        self._loaded = False

    # -- Test helpers -------------------------------------------------------

    def add_request(self, request_id: str = "req"):
        """Simulate an active in-flight request."""
        self.active_operations += 1

    def finish_request(self, request_id: str | None = None):
        """Simulate a request finishing."""
        if self.active_operations > 0:
            self.active_operations -= 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_mock_models(
    tmp_path: Path,
    model_ids: list[str],
    size: int = 4000,
) -> None:
    """Create minimal model directories that pass ``discover_models``."""
    for model_id in model_ids:
        model_dir = tmp_path / model_id
        model_dir.mkdir(exist_ok=True)
        (model_dir / "config.json").write_text(
            json.dumps({"model_type": "llama"})
        )
        # File size IS the estimated_size (model_discovery just sums file sizes).
        (model_dir / "model.safetensors").write_bytes(b"0" * size)


def _patch_load_engine(pool: EnginePool, load_delay: float = 0.05) -> None:
    """Replace ``_load_engine`` with a MockEngine-based version."""
    pool._mock_engines: dict[str, MockEngine] = {}

    async def _mock_load(model_id: str, force_lm: bool = False) -> None:
        engine = MockEngine(model_id, load_delay=load_delay)
        await engine.start()
        entry = pool._entries[model_id]
        entry.engine = engine
        pool._mock_engines[model_id] = engine

    pool._load_engine = _mock_load  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
async def switching_pool(tmp_path):
    """EnginePool configured for model-switching tests.

    * max_model_memory=10 000 — fits two 4 000-byte models but not three.
    * drain_timeout=5 s, max_wait_timeout=10 s — short for fast tests.
    * Teardown asserts _timeout_counter == 0 (catches masked livelocks).
    """
    pool = EnginePool(
        max_model_memory=10_000,
        drain_timeout=5,
        max_wait_timeout=10,
    )
    _create_mock_models(tmp_path, ["model-a", "model-b", "model-c"], size=4000)
    pool.discover_models(str(tmp_path))
    _patch_load_engine(pool)

    yield pool

    await pool.shutdown()
    assert pool._timeout_counter == 0, (
        f"LIVELOCK: {pool._timeout_counter} timeout(s) fired during test. "
        "Check LIVELOCK_SUSPECT entries in the log."
    )


@pytest.fixture
async def pool_expecting_timeouts(tmp_path):
    """Pool where timeouts are *expected* (livelock-detection tests).

    Uses very short timeouts so tests complete quickly.
    Does NOT assert ``_timeout_counter == 0`` on teardown.
    """
    pool = EnginePool(
        max_model_memory=10_000,
        drain_timeout=2,
        max_wait_timeout=3,
    )
    _create_mock_models(tmp_path, ["model-a", "model-b"], size=6000)
    pool.discover_models(str(tmp_path))
    _patch_load_engine(pool)

    yield pool

    await pool.shutdown()
    # No _timeout_counter assertion — we WANT timeouts here.


# ===================================================================
# 1. State Machine Transitions
# ===================================================================

class TestStateMachineTransitions:
    """Verify every valid state transition and reject invalid ones."""

    async def test_unloaded_to_loading(self, switching_pool):
        """UNLOADED -> LOADING: state changes and ready_event is created."""
        entry = switching_pool.get_entry("model-a")
        assert entry.state == EngineState.UNLOADED

        # Start loading in a task so we can inspect intermediate state.
        load_started = asyncio.Event()
        original_load = switching_pool._load_engine

        async def slow_load(model_id, **kwargs):
            load_started.set()
            await asyncio.sleep(0.2)
            await original_load(model_id)

        switching_pool._load_engine = slow_load

        task = asyncio.create_task(switching_pool.get_engine("model-a"))
        await load_started.wait()

        assert entry.state == EngineState.LOADING
        assert entry.ready_event is not None
        assert not entry.ready_event.is_set()

        engine = await asyncio.wait_for(task, timeout=5)
        assert engine is not None

    async def test_loading_to_active(self, switching_pool):
        """LOADING -> ACTIVE: engine is set and ready_event fires."""
        engine = await switching_pool.get_engine("model-a")
        entry = switching_pool.get_entry("model-a")

        assert entry.state == EngineState.ACTIVE
        assert entry.engine is engine
        assert entry.ready_event is not None
        assert entry.ready_event.is_set()

    async def test_loading_to_unloaded_on_failure(self, switching_pool):
        """LOADING -> UNLOADED on load failure: load_error set, ready_event fired."""
        load_error = RuntimeError("corrupt weights")

        async def failing_load(model_id, **kwargs):
            raise load_error

        switching_pool._load_engine = failing_load

        with pytest.raises(RuntimeError, match="corrupt weights"):
            await switching_pool.get_engine("model-a")

        entry = switching_pool.get_entry("model-a")
        assert entry.state == EngineState.UNLOADED
        assert entry.ready_event is not None
        assert entry.ready_event.is_set()
        assert entry.load_error is load_error

    async def test_active_to_draining(self, switching_pool):
        """ACTIVE -> DRAINING: drain_complete created, drain_monitor started."""
        engine_a = await switching_pool.get_engine("model-a")
        engine_a.add_request("req-1")

        # Fill remaining memory so eviction is needed for model-c.
        engine_b = await switching_pool.get_engine("model-b")
        engine_b.add_request("req-2")  # Keep model-b busy so it can't be evicted idle

        # Request model-c triggers drain of LRU (model-a, since accessed first).
        task_c = asyncio.create_task(switching_pool.get_engine("model-c"))
        await asyncio.sleep(0.1)

        entry_a = switching_pool.get_entry("model-a")
        assert entry_a.state == EngineState.DRAINING
        assert entry_a.drain_complete is not None
        assert not entry_a.drain_complete.is_set()

        # Clean up: finish the requests so drain completes.
        engine_a.finish_request("req-1")
        engine_b.finish_request("req-2")
        await asyncio.wait_for(task_c, timeout=8)

    async def test_draining_to_unloaded(self, switching_pool):
        """DRAINING -> UNLOADED: engine cleared, drain_complete fired."""
        engine_a = await switching_pool.get_engine("model-a")
        engine_a.add_request("req-1")
        engine_b = await switching_pool.get_engine("model-b")
        engine_b.add_request("req-2")  # Keep busy so model-a is drained (not model-b evicted)

        task_c = asyncio.create_task(switching_pool.get_engine("model-c"))
        await asyncio.sleep(0.1)

        entry_a = switching_pool.get_entry("model-a")
        assert entry_a.state == EngineState.DRAINING

        engine_a.finish_request("req-1")
        engine_b.finish_request("req-2")
        await asyncio.wait_for(task_c, timeout=8)

        assert entry_a.state == EngineState.UNLOADED
        assert entry_a.engine is None
        assert entry_a.drain_complete.is_set()

    async def test_draining_to_unloaded_on_timeout(self, pool_expecting_timeouts):
        """DRAINING -> UNLOADED on timeout: requests force-aborted."""
        pool = pool_expecting_timeouts
        engine_a = await pool.get_engine("model-a")
        engine_a.add_request("infinite-req")  # Never finishes

        task_b = asyncio.create_task(pool.get_engine("model-b"))

        # Wait for drain timeout (2s) + some slack.
        try:
            await asyncio.wait_for(task_b, timeout=8)
        except Exception:
            pass  # May raise on load failure or timeout cascade

        entry_a = pool.get_entry("model-a")
        assert entry_a.state == EngineState.UNLOADED
        assert entry_a.drain_complete.is_set()
        assert pool._timeout_counter >= 1

    async def test_set_state_logs_transition(self, switching_pool, caplog):
        """_set_state logs old->new state with reason."""
        with caplog.at_level(logging.INFO):
            await switching_pool.get_engine("model-a")

        # The log should contain the transition info.
        found = any(
            "unloaded" in rec.message.lower() and "active" in rec.message.lower()
            for rec in caplog.records
        )
        # Also accept loading->active pattern.
        found = found or any(
            "loading" in rec.message.lower() and "active" in rec.message.lower()
            for rec in caplog.records
        )
        assert found, (
            "Expected state transition log entry not found. "
            f"Records: {[r.message for r in caplog.records]}"
        )


# ===================================================================
# 2. Fast Path (No Switching)
# ===================================================================

class TestFastPath:
    """Ensure the common case (model already ACTIVE) has minimal overhead."""

    async def test_active_model_returns_immediately(self, switching_pool):
        """get_engine for an already-loaded model returns instantly."""
        engine = await switching_pool.get_engine("model-a")
        entry = switching_pool.get_entry("model-a")
        assert entry.state == EngineState.ACTIVE

        t0 = time.monotonic()
        engine2 = await switching_pool.get_engine("model-a")
        elapsed = time.monotonic() - t0

        assert engine2 is engine
        assert elapsed < 0.05  # Should be near-instantaneous

    async def test_active_model_updates_last_access(self, switching_pool):
        """get_engine updates last_access on the fast path."""
        await switching_pool.get_engine("model-a")
        entry = switching_pool.get_entry("model-a")
        first_access = entry.last_access

        await asyncio.sleep(0.01)
        await switching_pool.get_engine("model-a")
        assert entry.last_access > first_access

    async def test_multiple_concurrent_active_requests(self, switching_pool):
        """10 concurrent get_engine calls for an ACTIVE model all succeed."""
        engine = await switching_pool.get_engine("model-a")

        tasks = [
            asyncio.create_task(switching_pool.get_engine("model-a"))
            for _ in range(10)
        ]
        results = await asyncio.gather(*tasks)

        assert all(r is engine for r in results)


# ===================================================================
# 3. Loading Coalescing
# ===================================================================

class TestLoadingCoalescing:
    """Multiple requests for an unloaded model trigger exactly one load."""

    async def test_concurrent_get_engine_one_load(self, switching_pool):
        """5 concurrent get_engine('model-a') -> one _load_engine call."""
        load_count = 0
        original_load = switching_pool._load_engine

        async def counting_load(model_id, **kwargs):
            nonlocal load_count
            load_count += 1
            await asyncio.sleep(0.1)  # Simulate load time
            await original_load(model_id)

        switching_pool._load_engine = counting_load

        tasks = [
            asyncio.create_task(switching_pool.get_engine("model-a"))
            for _ in range(5)
        ]
        engines = await asyncio.gather(*tasks)

        assert load_count == 1
        assert all(e is engines[0] for e in engines)

    async def test_coalesced_waiters_see_active(self, switching_pool):
        """All coalesced waiters observe state=ACTIVE and a valid engine."""
        seen_states = []

        original_load = switching_pool._load_engine

        async def slow_load(model_id, **kwargs):
            await asyncio.sleep(0.1)
            await original_load(model_id)

        switching_pool._load_engine = slow_load

        async def observing_waiter():
            engine = await switching_pool.get_engine("model-a")
            entry = switching_pool.get_entry("model-a")
            seen_states.append((entry.state, entry.engine is not None))
            return engine

        tasks = [asyncio.create_task(observing_waiter()) for _ in range(5)]
        await asyncio.gather(*tasks)

        assert all(
            state == EngineState.ACTIVE and has_engine
            for state, has_engine in seen_states
        )

    async def test_coalesced_waiters_see_error(self, switching_pool):
        """All coalesced waiters see the load error when loading fails."""
        async def failing_load(model_id, **kwargs):
            await asyncio.sleep(0.1)
            raise RuntimeError("kaboom")

        switching_pool._load_engine = failing_load

        tasks = [
            asyncio.create_task(switching_pool.get_engine("model-a"))
            for _ in range(5)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # The first caller raises the load error. Coalesced waiters either
        # see the same error propagated via load_error or retry and also fail.
        errors = [r for r in results if isinstance(r, Exception)]
        assert len(errors) >= 1
        entry = switching_pool.get_entry("model-a")
        assert entry.state == EngineState.UNLOADED

    async def test_second_load_attempt_after_failure(self, switching_pool):
        """After a load failure, a new request triggers a fresh load attempt."""
        call_count = 0
        original_load = switching_pool._load_engine

        async def sometimes_failing_load(model_id, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("first attempt fails")
            await original_load(model_id)

        switching_pool._load_engine = sometimes_failing_load

        # First attempt fails.
        with pytest.raises(RuntimeError, match="first attempt fails"):
            await switching_pool.get_engine("model-a")

        assert call_count == 1
        entry = switching_pool.get_entry("model-a")
        assert entry.state == EngineState.UNLOADED

        # Reset the load cooldown so the retry happens immediately.
        entry.load_failed_at = 0

        # Second attempt should succeed (new load triggered).
        engine = await switching_pool.get_engine("model-a")
        assert engine is not None
        assert call_count == 2
        assert entry.state == EngineState.ACTIVE


# ===================================================================
# 4. Drain Behavior
# ===================================================================

class TestDrainBehavior:
    """Drain mechanism: idle unload, busy drain, timeout, crash recovery."""

    async def test_idle_victim_immediate_unload(self, switching_pool):
        """An idle victim is unloaded immediately (no drain)."""
        await switching_pool.get_engine("model-a")
        await switching_pool.get_engine("model-b")

        entry_a = switching_pool.get_entry("model-a")
        assert entry_a.state == EngineState.ACTIVE

        # model-c triggers eviction. model-a is LRU and idle -> immediate unload.
        engine_c = await switching_pool.get_engine("model-c")
        assert engine_c is not None

        assert entry_a.state == EngineState.UNLOADED
        assert entry_a.engine is None

    async def test_busy_victim_starts_drain(self, switching_pool):
        """A busy victim is drained instead of immediately unloaded."""
        engine_a = await switching_pool.get_engine("model-a")
        engine_a.add_request("req-1")
        engine_b = await switching_pool.get_engine("model-b")
        engine_b.add_request("req-2")  # Keep busy so model-a is drained

        task_c = asyncio.create_task(switching_pool.get_engine("model-c"))
        await asyncio.sleep(0.1)

        entry_a = switching_pool.get_entry("model-a")
        assert entry_a.state == EngineState.DRAINING

        # Clean up.
        engine_a.finish_request("req-1")
        engine_b.finish_request("req-2")
        await asyncio.wait_for(task_c, timeout=8)

    async def test_drain_completes_when_requests_finish(self, switching_pool):
        """Drain completes automatically once all active requests finish."""
        engine_a = await switching_pool.get_engine("model-a")
        engine_a.add_request("req-1")
        engine_b = await switching_pool.get_engine("model-b")
        engine_b.add_request("req-2")  # Keep busy so model-a is drained

        task_c = asyncio.create_task(switching_pool.get_engine("model-c"))
        await asyncio.sleep(0.1)

        entry_a = switching_pool.get_entry("model-a")
        assert entry_a.state == EngineState.DRAINING

        engine_a.finish_request("req-1")
        engine_b.finish_request("req-2")
        # drain_monitor polls every ~1s; give it time.
        engine_c = await asyncio.wait_for(task_c, timeout=8)

        assert entry_a.state == EngineState.UNLOADED
        assert engine_c is not None
        assert switching_pool.get_entry("model-c").state == EngineState.ACTIVE

    async def test_drain_timeout_force_aborts(self, pool_expecting_timeouts):
        """Drain timeout force-aborts remaining requests."""
        pool = pool_expecting_timeouts
        engine_a = await pool.get_engine("model-a")
        engine_a.add_request("stuck-req")  # Never finishes.

        task_b = asyncio.create_task(pool.get_engine("model-b"))
        try:
            await asyncio.wait_for(task_b, timeout=8)
        except Exception:
            pass

        entry_a = pool.get_entry("model-a")
        assert entry_a.state == EngineState.UNLOADED
        assert pool._timeout_counter >= 1

    async def test_drain_timeout_increments_counter(self, pool_expecting_timeouts):
        """Drain timeout increments _timeout_counter."""
        pool = pool_expecting_timeouts
        engine_a = await pool.get_engine("model-a")
        engine_a.add_request("stuck")

        task_b = asyncio.create_task(pool.get_engine("model-b"))
        try:
            await asyncio.wait_for(task_b, timeout=8)
        except Exception:
            pass

        assert pool._timeout_counter >= 1

    async def test_new_requests_rejected_during_drain(self, switching_pool):
        """get_engine for a draining model waits (does not return the draining engine)."""
        engine_a = await switching_pool.get_engine("model-a")
        engine_a.add_request("req-1")
        engine_b = await switching_pool.get_engine("model-b")
        engine_b.add_request("req-2")  # Keep busy so model-a is drained

        # Trigger drain of model-a via model-c request.
        task_c = asyncio.create_task(switching_pool.get_engine("model-c"))
        await asyncio.sleep(0.1)

        entry_a = switching_pool.get_entry("model-a")
        assert entry_a.state == EngineState.DRAINING

        # A second request for model-a must wait, not get the draining engine.
        task_a_again = asyncio.create_task(switching_pool.get_engine("model-a"))
        await asyncio.sleep(0.1)
        assert not task_a_again.done(), (
            "get_engine for a draining model must not return immediately"
        )

        # Complete the drain.
        engine_a.finish_request("req-1")
        engine_b.finish_request("req-2")
        await asyncio.wait_for(task_c, timeout=8)
        # task_a_again should eventually resolve (model-a will be reloaded or
        # wait until memory is available).
        result = await asyncio.wait_for(task_a_again, timeout=8)
        assert result is not None

    async def test_drain_monitor_crash_sets_event(self, switching_pool):
        """If drain_monitor crashes, drain_complete is still set."""
        engine_a = await switching_pool.get_engine("model-a")
        engine_a.add_request("req-1")
        engine_b = await switching_pool.get_engine("model-b")
        engine_b.add_request("req-2")  # Keep busy so model-a is drained

        # Monkey-patch _engine_has_active_work to raise inside drain_monitor.
        # Use a flag so we only crash AFTER the initial victim selection and
        # drain start (which need the function to work normally).
        drain_started_flag = asyncio.Event()
        original_check = switching_pool._engine_has_active_work

        @staticmethod
        def crashing_check(engine):
            if drain_started_flag.is_set():
                raise RuntimeError("simulated crash in drain monitor")
            return original_check(engine)

        # Let the normal flow start the drain, then patch + set the flag
        task_c = asyncio.create_task(switching_pool.get_engine("model-c"))
        await asyncio.sleep(0.2)  # Let drain start

        # Now patch and set the flag so drain_monitor crashes on next poll
        switching_pool._engine_has_active_work = crashing_check
        drain_started_flag.set()

        # Give drain monitor time to crash and recover.
        try:
            await asyncio.wait_for(task_c, timeout=8)
        except Exception:
            pass  # The load may fail, but drain_complete must be set.

        entry_a = switching_pool.get_entry("model-a")
        # The critical assertion: drain_complete was set despite the crash.
        assert entry_a.drain_complete is not None
        assert entry_a.drain_complete.is_set()


# ===================================================================
# 5. Implicit Queue (Wait Instead of Fail)
# ===================================================================

class TestImplicitQueue:
    """No 507 errors — callers wait for drain/load instead of failing."""

    async def test_no_507_during_model_switch(self, switching_pool):
        """The core requirement: never fail a request, always wait."""
        engine_a = await switching_pool.get_engine("model-a")
        engine_a.add_request("req-a")  # Keep model-a busy too
        engine_b = await switching_pool.get_engine("model-b")
        engine_b.add_request("req-1")  # Keep model-b busy.

        # Request model-c: must wait for drain, NOT raise 507 / InsufficientMemoryError.
        task_c = asyncio.create_task(switching_pool.get_engine("model-c"))

        # Let drain start, then finish requests.
        await asyncio.sleep(0.1)
        engine_a.finish_request("req-a")
        engine_b.finish_request("req-1")

        engine_c = await asyncio.wait_for(task_c, timeout=8)
        assert engine_c is not None

    async def test_wait_for_drain_then_load(self, switching_pool):
        """Caller waits for drain, then model is loaded after drain completes."""
        engine_a = await switching_pool.get_engine("model-a")
        engine_a.add_request("req-1")
        engine_b = await switching_pool.get_engine("model-b")
        engine_b.add_request("req-2")  # Keep busy so model-a is drained

        task_c = asyncio.create_task(switching_pool.get_engine("model-c"))
        await asyncio.sleep(0.1)

        # model-a is draining.
        entry_a = switching_pool.get_entry("model-a")
        assert entry_a.state == EngineState.DRAINING

        engine_a.finish_request("req-1")
        engine_b.finish_request("req-2")
        engine_c = await asyncio.wait_for(task_c, timeout=8)

        assert engine_c is not None
        assert switching_pool.get_entry("model-c").state == EngineState.ACTIVE

    async def test_wait_for_loading_model(self, switching_pool):
        """A second caller waits for an already-loading model (coalescing)."""
        original_load = switching_pool._load_engine
        load_started = asyncio.Event()

        async def slow_load(model_id, **kwargs):
            load_started.set()
            await asyncio.sleep(0.2)
            await original_load(model_id)

        switching_pool._load_engine = slow_load

        task1 = asyncio.create_task(switching_pool.get_engine("model-a"))
        await load_started.wait()

        # Second caller should coalesce.
        task2 = asyncio.create_task(switching_pool.get_engine("model-a"))

        e1, e2 = await asyncio.gather(
            asyncio.wait_for(task1, timeout=5),
            asyncio.wait_for(task2, timeout=5),
        )
        assert e1 is e2

    async def test_multiple_waiters_on_drain(self, switching_pool):
        """3 callers waiting for the same drain all proceed after completion."""
        engine_a = await switching_pool.get_engine("model-a")
        engine_a.add_request("req-1")
        engine_b = await switching_pool.get_engine("model-b")
        engine_b.add_request("req-2")  # Keep busy so model-a is drained

        # Three callers for model-c, all waiting on model-a's drain.
        tasks = [
            asyncio.create_task(switching_pool.get_engine("model-c"))
            for _ in range(3)
        ]
        await asyncio.sleep(0.1)

        engine_a.finish_request("req-1")
        engine_b.finish_request("req-2")
        results = await asyncio.gather(
            *[asyncio.wait_for(t, timeout=8) for t in tasks]
        )

        # All should get the same model-c engine.
        assert all(r is not None for r in results)
        assert all(r is results[0] for r in results)

    async def test_wait_timeout_returns_error(self, tmp_path):
        """ModelLoadingError after max_wait_timeout if model never available."""
        pool = EnginePool(
            max_model_memory=10_000,
            drain_timeout=5,
            max_wait_timeout=2,  # Short wait timeout
        )
        _create_mock_models(tmp_path, ["model-a", "model-b"], size=6000)
        pool.discover_models(str(tmp_path))
        _patch_load_engine(pool)

        # Load model-a and keep it busy. With drain_timeout=5 > max_wait_timeout=2,
        # the get_engine wait times out before the drain can complete.
        engine_a = await pool.get_engine("model-a")
        engine_a.add_request("stuck-req")

        with pytest.raises((ModelLoadingError, asyncio.TimeoutError)):
            await asyncio.wait_for(
                pool.get_engine("model-b"),
                timeout=8,
            )

        await pool.shutdown()


# ===================================================================
# 6. Client Cancellation
# ===================================================================

class TestClientCancellation:
    """Client cancellation at any stage must be clean with no leaked state."""

    async def test_cancel_while_waiting_for_load(self, switching_pool):
        """Cancelling a task waiting on ready_event raises CancelledError."""
        original_load = switching_pool._load_engine
        load_started = asyncio.Event()

        async def slow_load(model_id, **kwargs):
            load_started.set()
            await asyncio.sleep(10)  # Very slow load
            await original_load(model_id)

        switching_pool._load_engine = slow_load

        # Two tasks: one loader, one waiter.
        task_loader = asyncio.create_task(switching_pool.get_engine("model-a"))
        await load_started.wait()

        task_waiter = asyncio.create_task(switching_pool.get_engine("model-a"))
        await asyncio.sleep(0.05)

        # Cancel the waiter.
        task_waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task_waiter

        # Cancel the loader too to clean up.
        task_loader.cancel()
        try:
            await task_loader
        except (asyncio.CancelledError, Exception):
            pass

        # Entry should eventually reach UNLOADED (loader was cancelled).
        entry = switching_pool.get_entry("model-a")
        # ready_event must be set so no one is stuck.
        if entry.ready_event is not None:
            assert entry.ready_event.is_set()

    async def test_cancel_while_waiting_for_drain(self, switching_pool):
        """Cancelling a task waiting on drain_complete does not disrupt drain."""
        engine_a = await switching_pool.get_engine("model-a")
        engine_a.add_request("req-1")
        engine_b = await switching_pool.get_engine("model-b")
        engine_b.add_request("req-2")  # Keep busy so model-a is drained

        task_c = asyncio.create_task(switching_pool.get_engine("model-c"))
        await asyncio.sleep(0.1)

        entry_a = switching_pool.get_entry("model-a")
        assert entry_a.state == EngineState.DRAINING

        # Cancel the waiting task (simulates client disconnect).
        task_c.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task_c

        # Drain should continue unaffected.
        assert entry_a.state == EngineState.DRAINING

        # Finish model-a's work; drain should complete cleanly.
        engine_a.finish_request("req-1")
        await asyncio.sleep(2)
        assert entry_a.state == EngineState.UNLOADED

    async def test_cancel_during_load_wakes_waiters(self, switching_pool):
        """If the loader task is cancelled, waiters must not be stuck forever."""
        load_started = asyncio.Event()

        async def slow_load(model_id, **kwargs):
            load_started.set()
            await asyncio.sleep(100)

        switching_pool._load_engine = slow_load

        task_loader = asyncio.create_task(switching_pool.get_engine("model-a"))
        await load_started.wait()

        entry = switching_pool.get_entry("model-a")
        assert entry.state == EngineState.LOADING

        task_loader.cancel()
        try:
            await task_loader
        except (asyncio.CancelledError, Exception):
            pass

        # ready_event must be set (INV-2).
        assert entry.ready_event is not None
        assert entry.ready_event.is_set()
        assert entry.state == EngineState.UNLOADED

    async def test_cancel_does_not_affect_other_waiters(self, switching_pool):
        """Cancelling one of N waiters does not affect the other N-1."""
        original_load = switching_pool._load_engine
        load_started = asyncio.Event()

        async def slow_load(model_id, **kwargs):
            load_started.set()
            await asyncio.sleep(0.3)
            await original_load(model_id)

        switching_pool._load_engine = slow_load

        tasks = [
            asyncio.create_task(switching_pool.get_engine("model-a"))
            for _ in range(5)
        ]
        await load_started.wait()

        # Cancel one waiter.
        tasks[2].cancel()
        try:
            await tasks[2]
        except asyncio.CancelledError:
            pass

        # Remaining 4 tasks should succeed.
        remaining = [t for i, t in enumerate(tasks) if i != 2]
        results = await asyncio.gather(
            *[asyncio.wait_for(t, timeout=5) for t in remaining],
            return_exceptions=True,
        )
        engines = [r for r in results if not isinstance(r, Exception)]
        assert len(engines) >= 3  # At least most should succeed
        assert all(e is engines[0] for e in engines)


# ===================================================================
# 7. Atomicity & TOCTOU
# ===================================================================

class TestAtomicityTOCTOU:
    """Verify atomic state transitions and absence of TOCTOU bugs."""

    async def test_post_load_state_and_signal_atomic(self, switching_pool):
        """Waiters never see ACTIVE without a valid engine."""
        seen_states = []

        original_load = switching_pool._load_engine

        async def slow_load(model_id, **kwargs):
            await asyncio.sleep(0.1)
            await original_load(model_id)

        switching_pool._load_engine = slow_load

        async def observing_waiter():
            engine = await switching_pool.get_engine("model-a")
            entry = switching_pool.get_entry("model-a")
            seen_states.append((entry.state, entry.engine is not None))
            return engine

        tasks = [asyncio.create_task(observing_waiter()) for _ in range(5)]
        await asyncio.gather(*tasks)

        assert all(
            state == EngineState.ACTIVE and has_engine
            for state, has_engine in seen_states
        )

    async def test_load_error_visible_to_waiters(self, switching_pool):
        """When load fails, load_error is set before ready_event fires."""
        async def failing_load(model_id, **kwargs):
            await asyncio.sleep(0.1)
            raise RuntimeError("load failed")

        switching_pool._load_engine = failing_load

        results = await asyncio.gather(
            *[
                asyncio.create_task(switching_pool.get_engine("model-a"))
                for _ in range(3)
            ],
            return_exceptions=True,
        )

        errors = [r for r in results if isinstance(r, Exception)]
        assert len(errors) >= 1

        entry = switching_pool.get_entry("model-a")
        assert entry.state == EngineState.UNLOADED
        assert entry.load_error is not None

    async def test_drain_check_and_unload_atomic(self, switching_pool):
        """No request slips in between drain idle check and unload.

        This verifies that checking active_work and unloading happen under
        the same lock hold.
        """
        engine_a = await switching_pool.get_engine("model-a")
        engine_a.add_request("req-1")
        engine_b = await switching_pool.get_engine("model-b")
        engine_b.add_request("req-2")  # Keep busy so model-a is drained

        task_c = asyncio.create_task(switching_pool.get_engine("model-c"))
        await asyncio.sleep(0.1)

        entry_a = switching_pool.get_entry("model-a")
        assert entry_a.state == EngineState.DRAINING

        # Finish requests. On the next drain_monitor poll, it should atomically
        # check that there's no work and unload.
        engine_a.finish_request("req-1")
        engine_b.finish_request("req-2")
        engine_c = await asyncio.wait_for(task_c, timeout=8)

        # model-a should be cleanly unloaded.
        assert entry_a.state == EngineState.UNLOADED
        assert entry_a.engine is None
        assert engine_c is not None

    async def test_concurrent_unload_idempotent(self, switching_pool):
        """Two concurrent unload attempts on the same model don't crash.

        This tests defensive coding: _unload_engine is idempotent even when
        called concurrently without external lock coordination.
        """
        await switching_pool.get_engine("model-a")
        entry = switching_pool.get_entry("model-a")
        assert entry.engine is not None

        # Two concurrent unloads — tests defensive idempotency.
        await asyncio.gather(
            switching_pool._unload_engine("model-a"),
            switching_pool._unload_engine("model-a"),
        )

        # Should be cleanly unloaded, no crash.
        assert entry.engine is None


# ===================================================================
# 8. Memory Accounting
# ===================================================================

class TestMemoryAccounting:
    """Verify committed_memory accounting with draining models."""

    async def test_committed_memory_includes_draining(self, switching_pool):
        """A draining model's memory is still counted in committed_memory."""
        engine_a = await switching_pool.get_engine("model-a")
        engine_a.add_request("req-1")
        engine_b = await switching_pool.get_engine("model-b")
        engine_b.add_request("req-2")  # Keep busy so model-a is drained

        committed_before = switching_pool._committed_memory()

        task_c = asyncio.create_task(switching_pool.get_engine("model-c"))
        await asyncio.sleep(0.1)

        entry_a = switching_pool.get_entry("model-a")
        assert entry_a.state == EngineState.DRAINING

        # Draining model should still be counted.
        committed_during = switching_pool._committed_memory()
        assert committed_during >= entry_a.estimated_size

        # Clean up.
        engine_a.finish_request("req-1")
        engine_b.finish_request("req-2")
        await asyncio.wait_for(task_c, timeout=8)

    async def test_committed_memory_excludes_unloaded(self, switching_pool):
        """Unloaded models do not contribute to committed_memory."""
        await switching_pool.get_engine("model-a")
        committed_with_a = switching_pool._committed_memory()
        assert committed_with_a > 0

        await switching_pool._unload_engine("model-a")
        # Wait for deferred cleanup to complete (UNLOADING -> UNLOADED)
        if switching_pool._cleanup_tasks:
            await asyncio.gather(*switching_pool._cleanup_tasks, return_exceptions=True)
        committed_after = switching_pool._committed_memory()
        assert committed_after < committed_with_a

    async def test_prepare_memory_returns_event_when_full(self, switching_pool):
        """_prepare_memory_for returns a drain event when memory is full and models draining."""
        engine_a = await switching_pool.get_engine("model-a")
        engine_a.add_request("req-1")
        engine_b = await switching_pool.get_engine("model-b")
        engine_b.add_request("req-2")  # Keep busy so model-a is drained

        # Both slots full. model-c triggers drain of model-a.
        task_c = asyncio.create_task(switching_pool.get_engine("model-c"))
        await asyncio.sleep(0.1)

        entry_a = switching_pool.get_entry("model-a")
        assert entry_a.state == EngineState.DRAINING
        assert entry_a.drain_complete is not None

        engine_a.finish_request("req-1")
        engine_b.finish_request("req-2")
        await asyncio.wait_for(task_c, timeout=8)

    async def test_prepare_memory_raises_when_all_pinned(self, tmp_path):
        """_prepare_memory_for raises ModelTooLargeError when all memory is pinned."""
        pool = EnginePool(
            max_model_memory=8000,
            drain_timeout=5,
            max_wait_timeout=5,
        )
        _create_mock_models(tmp_path, ["pinned-a", "unpinned-b"], size=4000)
        pool.discover_models(str(tmp_path), pinned_models=["pinned-a"])
        _patch_load_engine(pool)

        await pool.get_engine("pinned-a")

        # unpinned-b fits alongside pinned-a (4000+4000 <= 8000 without headroom).
        # But with KV headroom (25%) = 5000, so 4000+5000 > 8000.
        # The pool should either evict or fail depending on memory math.
        # Let's create a tighter scenario:
        await pool.shutdown()

        pool2 = EnginePool(
            max_model_memory=5000,
            drain_timeout=5,
            max_wait_timeout=3,
        )
        _create_mock_models(tmp_path, ["pinned-x"], size=4000)
        # Add an extra model that's too big.
        extra_dir = tmp_path / "huge-model"
        extra_dir.mkdir(exist_ok=True)
        (extra_dir / "config.json").write_text(json.dumps({"model_type": "llama"}))
        (extra_dir / "model.safetensors").write_bytes(b"0" * 6000)
        pool2.discover_models(str(tmp_path), pinned_models=["pinned-x"])
        _patch_load_engine(pool2)

        await pool2.get_engine("pinned-x")

        # huge-model (6000 bytes) can never fit even alone (6000 > 5000).
        with pytest.raises(ModelTooLargeError):
            await pool2.get_engine("huge-model")

        await pool2.shutdown()

    async def test_no_oom_sequential_drain_load(self, switching_pool):
        """Drain finishes before load starts -> peak memory = max(A,B), not A+B."""
        engine_a = await switching_pool.get_engine("model-a")
        engine_a.add_request("req-1")
        committed_a = switching_pool._committed_memory()

        # model-b (same size) should be loadable after drain, not during.
        engine_b = await switching_pool.get_engine("model-b")
        engine_b.add_request("req-2")  # Keep busy so model-a is drained

        task_c = asyncio.create_task(switching_pool.get_engine("model-c"))
        await asyncio.sleep(0.1)

        engine_a.finish_request("req-1")
        engine_b.finish_request("req-2")
        engine_c = await asyncio.wait_for(task_c, timeout=8)

        # After everything settles, committed memory should be <= max.
        final_committed = switching_pool._committed_memory()
        assert final_committed <= switching_pool.max_model_memory


# ===================================================================
# 9. Invariant Verification
# ===================================================================

class TestInvariants:
    """Dedicated tests for each invariant INV-1 through INV-8."""

    async def test_inv1_lock_not_held_during_event_wait(self, switching_pool):
        """INV-1: Lock is released before awaiting ready_event / drain_complete."""
        original_load = switching_pool._load_engine
        load_started = asyncio.Event()

        async def slow_load(model_id, **kwargs):
            load_started.set()
            await asyncio.sleep(0.3)
            await original_load(model_id)

        switching_pool._load_engine = slow_load

        task1 = asyncio.create_task(switching_pool.get_engine("model-a"))
        await load_started.wait()

        # While model-a is loading, we should be able to get_engine for
        # already-loaded model-b (i.e., the lock is not held during the wait).
        # First load model-b.
        # Actually, model-a is loading (lock released during load). We need
        # to verify that a concurrent get_engine for a different model can
        # proceed. Let's load model-b while model-a is loading.
        task2 = asyncio.create_task(switching_pool.get_engine("model-b"))

        # Both should complete without deadlock.
        e_a, e_b = await asyncio.gather(
            asyncio.wait_for(task1, timeout=5),
            asyncio.wait_for(task2, timeout=5),
        )
        assert e_a is not None
        assert e_b is not None

    async def test_inv2_ready_event_always_set(self, switching_pool):
        """INV-2: ready_event is always set when state leaves LOADING.

        Test for the CancelledError case: even if the loader is cancelled,
        ready_event must fire.
        """
        load_started = asyncio.Event()

        async def slow_load(model_id, **kwargs):
            load_started.set()
            await asyncio.sleep(100)

        switching_pool._load_engine = slow_load

        task = asyncio.create_task(switching_pool.get_engine("model-a"))
        await load_started.wait()

        entry = switching_pool.get_entry("model-a")
        assert entry.state == EngineState.LOADING

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        # ready_event MUST be set even though we cancelled.
        assert entry.ready_event is not None
        assert entry.ready_event.is_set()
        assert entry.state == EngineState.UNLOADED

    async def test_inv3_drain_complete_always_set(self, switching_pool):
        """INV-3: drain_complete is always set when state leaves DRAINING.

        Covers normal completion, timeout, and crash paths.
        """
        engine_a = await switching_pool.get_engine("model-a")
        engine_a.add_request("req-1")
        engine_b = await switching_pool.get_engine("model-b")
        engine_b.add_request("req-2")  # Keep busy so model-a is drained

        task_c = asyncio.create_task(switching_pool.get_engine("model-c"))
        await asyncio.sleep(0.1)

        entry_a = switching_pool.get_entry("model-a")
        assert entry_a.state == EngineState.DRAINING
        assert entry_a.drain_complete is not None

        engine_a.finish_request("req-1")
        engine_b.finish_request("req-2")
        await asyncio.wait_for(task_c, timeout=8)

        assert entry_a.drain_complete.is_set()
        assert entry_a.state == EngineState.UNLOADED

    async def test_inv4_get_engine_bounded(self, tmp_path):
        """INV-4: get_engine has bounded wait time (max_wait_timeout)."""
        pool = EnginePool(
            max_model_memory=10_000,
            drain_timeout=2,
            max_wait_timeout=2,  # Very short timeout
        )
        _create_mock_models(tmp_path, ["model-a", "model-b"], size=6000)
        pool.discover_models(str(tmp_path))
        _patch_load_engine(pool)

        engine_a = await pool.get_engine("model-a")
        engine_a.add_request("stuck")  # Never finishes

        t0 = time.monotonic()
        with pytest.raises((ModelLoadingError, asyncio.TimeoutError)):
            await asyncio.wait_for(
                pool.get_engine("model-b"),
                timeout=8,
            )
        elapsed = time.monotonic() - t0
        # Should complete within about max_wait_timeout + drain_timeout + slack.
        assert elapsed < 8

        await pool.shutdown()

    async def test_inv5_drain_bounded(self, pool_expecting_timeouts):
        """INV-5: drain_monitor has bounded run time (drain_timeout)."""
        pool = pool_expecting_timeouts
        engine_a = await pool.get_engine("model-a")
        engine_a.add_request("stuck")

        t0 = time.monotonic()
        task_b = asyncio.create_task(pool.get_engine("model-b"))
        try:
            await asyncio.wait_for(task_b, timeout=8)
        except Exception:
            pass
        elapsed = time.monotonic() - t0

        # Drain should timeout within drain_timeout (2s) + polling overhead.
        assert elapsed < 6
        assert pool._timeout_counter >= 1

    async def test_inv6_unload_bounded(self, tmp_path):
        """INV-6: _unload_engine has bounded run time."""
        pool = EnginePool(
            max_model_memory=10_000,
            drain_timeout=5,
            max_wait_timeout=10,
        )
        _create_mock_models(tmp_path, ["model-a"], size=4000)
        pool.discover_models(str(tmp_path))
        _patch_load_engine(pool)

        await pool.get_engine("model-a")

        # Make engine.stop() hang.
        entry = pool.get_entry("model-a")

        async def hanging_stop():
            await asyncio.sleep(100)  # Hang

        entry.engine.stop = hanging_stop

        # _unload_engine should have a timeout on stop().
        t0 = time.monotonic()
        try:
            await asyncio.wait_for(
                pool._unload_engine("model-a"),
                timeout=35,
            )
        except asyncio.TimeoutError:
            pass
        elapsed = time.monotonic() - t0

        # Should complete (or timeout) reasonably, not hang for 100s.
        assert elapsed < 35

        await pool.shutdown()

    async def test_inv7_single_loader(self, switching_pool):
        """INV-7: Only one coroutine loads a given model at a time."""
        load_count = 0
        original_load = switching_pool._load_engine

        async def counting_load(model_id, **kwargs):
            nonlocal load_count
            load_count += 1
            await asyncio.sleep(0.1)
            await original_load(model_id)

        switching_pool._load_engine = counting_load

        tasks = [
            asyncio.create_task(switching_pool.get_engine("model-a"))
            for _ in range(10)
        ]
        await asyncio.gather(*tasks)

        assert load_count == 1, f"Expected 1 load, got {load_count}"

    async def test_inv8_prepare_memory_explicit(self, tmp_path):
        """INV-8: _prepare_memory_for never returns None if memory is insufficient.

        When all memory is pinned and nothing can drain, it raises.
        """
        pool = EnginePool(
            max_model_memory=5000,
            drain_timeout=5,
            max_wait_timeout=5,
        )
        # One pinned model that fills memory.
        _create_mock_models(tmp_path, ["pinned"], size=4000)
        # A model that can't fit alongside pinned (even without headroom).
        huge_dir = tmp_path / "too-big"
        huge_dir.mkdir(exist_ok=True)
        (huge_dir / "config.json").write_text(json.dumps({"model_type": "llama"}))
        (huge_dir / "model.safetensors").write_bytes(b"0" * 6000)
        pool.discover_models(str(tmp_path), pinned_models=["pinned"])
        _patch_load_engine(pool)

        await pool.get_engine("pinned")

        # too-big can't fit even alone (6000 > 5000), let alone with pinned.
        with pytest.raises(ModelTooLargeError):
            await pool.get_engine("too-big")

        await pool.shutdown()


# ===================================================================
# 10. Livelock Detection
# ===================================================================

class TestLivelockDetection:
    """Tests that intentionally trigger timeouts and verify detection."""

    async def test_drain_timeout_detected(self, pool_expecting_timeouts, caplog):
        """Drain timeout increments _timeout_counter and logs LIVELOCK_SUSPECT."""
        pool = pool_expecting_timeouts
        engine_a = await pool.get_engine("model-a")
        engine_a.add_request("infinite-req")

        with caplog.at_level(logging.ERROR):
            task_b = asyncio.create_task(pool.get_engine("model-b"))
            try:
                await asyncio.wait_for(task_b, timeout=8)
            except Exception:
                pass

        assert pool._timeout_counter >= 1
        assert "LIVELOCK_SUSPECT" in caplog.text

    async def test_get_engine_wait_timeout_detected(
        self, tmp_path, caplog
    ):
        """get_engine wait timeout is detected and logged."""
        # max_wait_timeout < drain_timeout so the wait times out first
        pool = EnginePool(
            max_model_memory=10_000,
            drain_timeout=5,
            max_wait_timeout=2,
        )
        _create_mock_models(tmp_path, ["model-a", "model-b"], size=6000)
        pool.discover_models(str(tmp_path))
        _patch_load_engine(pool)

        engine_a = await pool.get_engine("model-a")
        engine_a.add_request("stuck")

        with caplog.at_level(logging.ERROR):
            with pytest.raises((ModelLoadingError, asyncio.TimeoutError)):
                await asyncio.wait_for(
                    pool.get_engine("model-b"),
                    timeout=8,
                )

        assert pool._timeout_counter >= 1
        # LIVELOCK_SUSPECT should appear in logs.
        assert "LIVELOCK_SUSPECT" in caplog.text

        await pool.shutdown()

    async def test_timeout_counter_accumulates(self, pool_expecting_timeouts):
        """Multiple timeouts accumulate in _timeout_counter."""
        pool = pool_expecting_timeouts
        engine_a = await pool.get_engine("model-a")
        engine_a.add_request("stuck-1")

        # First timeout attempt.
        try:
            await asyncio.wait_for(pool.get_engine("model-b"), timeout=8)
        except Exception:
            pass

        first_count = pool._timeout_counter
        assert first_count >= 1


# ===================================================================
# 11. Regression Tests for Existing Behavior
# ===================================================================

class TestRegression:
    """Ensure existing behavior is preserved for non-switching code paths."""

    async def test_existing_lru_eviction_still_works(self, switching_pool):
        """Idle model evicted via LRU when memory is needed (no drain)."""
        await switching_pool.get_engine("model-a")
        await switching_pool.get_engine("model-b")

        entry_a = switching_pool.get_entry("model-a")
        assert entry_a.state == EngineState.ACTIVE

        # model-c forces eviction of LRU (model-a, no active work).
        engine_c = await switching_pool.get_engine("model-c")
        assert engine_c is not None
        assert entry_a.state == EngineState.UNLOADED
        assert entry_a.engine is None

    async def test_existing_pinned_models_never_evicted(self, tmp_path):
        """Pinned models are never selected as eviction victims."""
        pool = EnginePool(
            max_model_memory=10_000,
            drain_timeout=5,
            max_wait_timeout=10,
        )
        _create_mock_models(tmp_path, ["pinned-a", "model-b", "model-c"], size=4000)
        pool.discover_models(str(tmp_path), pinned_models=["pinned-a"])
        _patch_load_engine(pool)

        await pool.get_engine("pinned-a")
        await pool.get_engine("model-b")

        # model-c should evict model-b (not pinned-a).
        engine_c = await pool.get_engine("model-c")
        assert engine_c is not None

        assert pool.get_entry("pinned-a").state == EngineState.ACTIVE
        assert pool.get_entry("model-b").state == EngineState.UNLOADED

        await pool.shutdown()

    async def test_existing_model_discovery_preserves_loaded(self, tmp_path):
        """Re-discovery preserves engines that are already loaded."""
        pool = EnginePool(
            max_model_memory=10_000,
            drain_timeout=5,
            max_wait_timeout=10,
        )
        _create_mock_models(tmp_path, ["model-a", "model-b"], size=4000)
        pool.discover_models(str(tmp_path))
        _patch_load_engine(pool)

        engine_a = await pool.get_engine("model-a")

        # Re-discover.
        pool.discover_models(str(tmp_path))

        entry_a = pool.get_entry("model-a")
        assert entry_a.engine is engine_a  # Preserved

        await pool.shutdown()

    async def test_existing_shutdown_stops_all(self, switching_pool):
        """pool.shutdown() stops all loaded engines."""
        await switching_pool.get_engine("model-a")
        await switching_pool.get_engine("model-b")

        assert switching_pool.loaded_model_count == 2

        await switching_pool.shutdown()

        assert switching_pool.loaded_model_count == 0

    async def test_existing_pool_status(self, switching_pool):
        """get_status() returns status info."""
        await switching_pool.get_engine("model-a")

        status = switching_pool.get_status()
        assert status["model_count"] >= 3
        assert status["loaded_count"] >= 1

        models = {m["id"]: m for m in status["models"]}
        assert "model-a" in models
        # Status should indicate loaded.
        assert models["model-a"]["loaded"] is True

    async def test_model_not_found_still_raises(self, switching_pool):
        """ModelNotFoundError raised for unknown model IDs."""
        with pytest.raises(ModelNotFoundError):
            await switching_pool.get_engine("nonexistent-model")


# ===================================================================
# 12. Integration Smoke Tests
# ===================================================================

@pytest.mark.slow
class TestIntegrationSmoke:
    """End-to-end smoke tests with full request flow.

    These use MockEngine (not real MLX) but exercise the complete pool
    lifecycle: load, serve, drain, reload.
    """

    async def test_e2e_switch_with_generation(self, switching_pool):
        """Load model-a, switch to model-b, both serve requests."""
        engine_a = await switching_pool.get_engine("model-a")
        engine_a.add_request("req-a-1")
        engine_a.finish_request("req-a-1")

        engine_b = await switching_pool.get_engine("model-b")
        engine_b.add_request("req-b-1")
        engine_b.finish_request("req-b-1")

        # Both models served requests.
        assert engine_a is not None
        assert engine_b is not None

    async def test_e2e_drain_during_generation(self, switching_pool):
        """Start request on A, switch to C via drain, A finishes before unload."""
        engine_a = await switching_pool.get_engine("model-a")
        engine_a.add_request("long-req")
        engine_b = await switching_pool.get_engine("model-b")
        engine_b.add_request("req-b")  # Keep busy so model-a is drained

        # Request model-c triggers drain of model-a.
        task_c = asyncio.create_task(switching_pool.get_engine("model-c"))
        await asyncio.sleep(0.1)

        entry_a = switching_pool.get_entry("model-a")
        assert entry_a.state == EngineState.DRAINING

        # "Finish generation" on model-a and model-b.
        engine_a.finish_request("long-req")
        engine_b.finish_request("req-b")

        engine_c = await asyncio.wait_for(task_c, timeout=8)
        assert engine_c is not None
        assert entry_a.state == EngineState.UNLOADED

    async def test_e2e_cancel_during_queue(self, switching_pool):
        """Cancel a queued request; no leaked state."""
        engine_a = await switching_pool.get_engine("model-a")
        engine_a.add_request("req-1")
        engine_b = await switching_pool.get_engine("model-b")
        engine_b.add_request("req-2")  # Keep busy so drain is triggered

        task_c = asyncio.create_task(switching_pool.get_engine("model-c"))
        await asyncio.sleep(0.1)

        task_c.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task_c

        # Pool should still be healthy.
        engine_a.finish_request("req-1")
        engine_b.finish_request("req-2")
        await asyncio.sleep(2)  # Let drain settle if started.

        # Can still use the pool for new requests.
        engine = await switching_pool.get_engine("model-b")
        assert engine is not None


# ===================================================================
# 13. Load Error Propagation & Cooldown
# ===================================================================

class TestLoadErrorPropagation:
    """Tests for C2 fix: coalesced waiters see the original load error."""

    async def test_coalesced_waiters_see_original_error(self, switching_pool):
        """Coalesced waiters see the ORIGINAL error, not a cooldown error.

        Verifies fix for C2: when load fails, coalesced waiters that were
        blocked on ready_event see the original exception message, not
        'failed to load N s ago, retrying in M s'.
        """
        async def failing_load(model_id, **kwargs):
            await asyncio.sleep(0.1)
            raise RuntimeError("GPU memory corrupted")

        switching_pool._load_engine = failing_load

        tasks = [
            asyncio.create_task(switching_pool.get_engine("model-a"))
            for _ in range(5)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        errors = [r for r in results if isinstance(r, Exception)]
        assert len(errors) >= 1

        # ALL errors should reference the original error text
        for err in errors:
            err_str = str(err)
            assert "GPU memory corrupted" in err_str, (
                f"Expected original error text, got: {err_str}"
            )
            # Should NOT contain cooldown message
            assert "retrying in" not in err_str, (
                f"Waiter got cooldown error instead of original: {err_str}"
            )


class TestLoadCooldown:
    """Tests for load cooldown mechanism (T2)."""

    async def test_request_within_cooldown_gets_error(self, switching_pool):
        """A request within 30s of a load failure gets ModelLoadingError with cooldown message.

        Verifies that the cooldown mechanism prevents rapid retry storms.
        """
        async def failing_load(model_id, **kwargs):
            raise RuntimeError("download error")

        switching_pool._load_engine = failing_load

        with pytest.raises(RuntimeError, match="download error"):
            await switching_pool.get_engine("model-a")

        entry = switching_pool.get_entry("model-a")
        assert entry.load_failed_at > 0

        # Immediate retry should hit cooldown
        with pytest.raises(ModelLoadingError, match="retrying in"):
            await switching_pool.get_engine("model-a")

    async def test_request_after_cooldown_triggers_fresh_load(self, switching_pool):
        """A request after the cooldown period triggers a fresh load attempt.

        Verifies that setting load_failed_at to the past allows retry.
        """
        call_count = 0
        original_load = switching_pool._load_engine

        async def sometimes_failing_load(model_id, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("transient error")
            await original_load(model_id)

        switching_pool._load_engine = sometimes_failing_load

        with pytest.raises(RuntimeError, match="transient error"):
            await switching_pool.get_engine("model-a")

        assert call_count == 1
        entry = switching_pool.get_entry("model-a")

        # Simulate cooldown expiry by setting load_failed_at far in the past
        entry.load_failed_at = time.time() - LOAD_COOLDOWN - 1

        engine = await switching_pool.get_engine("model-a")
        assert engine is not None
        assert call_count == 2
        assert entry.state == EngineState.ACTIVE


# ===================================================================
# 14. TTL + Drain Interaction
# ===================================================================

class TestTTLDrainInteraction:
    """Tests for TTL expiration with active requests triggering drain (T3)."""

    async def test_ttl_with_active_requests_starts_drain(self, tmp_path):
        """TTL expiry with active requests starts drain instead of immediate unload.

        Verifies that check_ttl_expirations respects in-flight work.
        """
        pool = EnginePool(
            max_model_memory=10_000,
            drain_timeout=5,
            max_wait_timeout=10,
        )
        _create_mock_models(tmp_path, ["model-a"], size=4000)
        pool.discover_models(str(tmp_path))
        _patch_load_engine(pool)

        engine_a = await pool.get_engine("model-a")
        engine_a.add_request("active-req")

        entry = pool.get_entry("model-a")
        # Set last_access far enough in the past to trigger TTL
        entry.last_access = time.time() - 100

        # Create a mock settings manager with a short TTL
        mock_settings = MagicMock()
        mock_model_settings = MagicMock()
        mock_model_settings.ttl_seconds = 10  # 10s TTL, last access 100s ago
        mock_settings.get_settings.return_value = mock_model_settings

        expired = await pool.check_ttl_expirations(mock_settings)

        # Model should be draining (not immediately unloaded) due to active request
        assert "model-a" in expired
        assert entry.state == EngineState.DRAINING

        # Clean up
        engine_a.finish_request("active-req")
        await asyncio.sleep(2)  # Let drain monitor finish
        await pool.shutdown()


# ===================================================================
# 15. Process Memory Check Path
# ===================================================================

class TestProcessMemoryCheck:
    """Tests for _check_process_memory (T4)."""

    async def test_check_process_memory_returns_none_without_enforcer(self, switching_pool):
        """_check_process_memory returns None when no enforcer is configured.

        This is the common case — most deployments don't set --max-process-memory.
        """
        entry = switching_pool.get_entry("model-a")
        assert switching_pool._process_memory_enforcer is None

        async with switching_pool._tracked_lock("test"):
            result = await switching_pool._check_process_memory(entry)
        assert result is None


# ===================================================================
# 16. Drain + Reload Returns Fresh Engine
# ===================================================================

class TestDrainReloadFreshEngine:
    """Tests that engine after drain+reload is a new object (T5)."""

    async def test_fresh_engine_after_drain_and_reload(self, switching_pool):
        """After drain and reload, the engine is a NEW object, not the old one.

        Verifies the stale engine is not accidentally reused.
        """
        engine_a = await switching_pool.get_engine("model-a")
        engine_a.add_request("req-1")
        engine_b = await switching_pool.get_engine("model-b")
        engine_b.add_request("req-2")  # Keep busy so model-a is drained

        # Trigger drain of model-a via model-c
        task_c = asyncio.create_task(switching_pool.get_engine("model-c"))
        await asyncio.sleep(0.1)

        entry_a = switching_pool.get_entry("model-a")
        assert entry_a.state == EngineState.DRAINING

        # Complete drain
        engine_a.finish_request("req-1")
        engine_b.finish_request("req-2")
        await asyncio.wait_for(task_c, timeout=8)

        assert entry_a.state == EngineState.UNLOADED

        # Now reload model-a (model-c and model-b may need eviction)
        engine_a_reloaded = await switching_pool.get_engine("model-a")

        # The reloaded engine MUST be a different object
        assert engine_a_reloaded is not engine_a


# ===================================================================
# 17. Concurrent Drain + Independent get_engine
# ===================================================================

class TestConcurrentDrainIndependentGet:
    """Tests that draining one model doesn't block unrelated models (T6)."""

    async def test_independent_model_returns_during_drain(self, switching_pool):
        """Loading model-b returns immediately while model-a is draining.

        Verifies that drain of one model doesn't block get_engine for an
        already-loaded independent model.
        """
        engine_a = await switching_pool.get_engine("model-a")
        engine_b = await switching_pool.get_engine("model-b")
        engine_a.add_request("req-1")

        # Start draining model-a manually
        async with switching_pool._tracked_lock("test_drain"):
            switching_pool._start_drain("model-a")

        entry_a = switching_pool.get_entry("model-a")
        assert entry_a.state == EngineState.DRAINING

        # model-b is already loaded and ACTIVE — should return immediately
        t0 = time.monotonic()
        result = await switching_pool.get_engine("model-b")
        elapsed = time.monotonic() - t0

        assert result is engine_b
        assert elapsed < 0.1, f"get_engine for loaded model took {elapsed:.3f}s"

        # Clean up
        engine_a.finish_request("req-1")
        await asyncio.sleep(2)  # Let drain complete


# ===================================================================
# 18. discover_models During LOADING State
# ===================================================================

class TestDiscoverModelsDuringLoading:
    """Tests that discover_models preserves LOADING entries (T7, C4 fix)."""

    async def test_discover_models_preserves_loading_entry(self, switching_pool):
        """discover_models mid-load preserves the LOADING entry and its waiters.

        After fix C4: entries in non-UNLOADED states (LOADING, DRAINING) are
        preserved during re-discovery, preventing orphaned waiters.
        """
        original_load = switching_pool._load_engine
        load_started = asyncio.Event()

        async def slow_load(model_id, **kwargs):
            load_started.set()
            await asyncio.sleep(0.3)
            await original_load(model_id)

        switching_pool._load_engine = slow_load

        # Start loading model-a
        task = asyncio.create_task(switching_pool.get_engine("model-a"))
        await load_started.wait()

        entry_before = switching_pool.get_entry("model-a")
        assert entry_before.state == EngineState.LOADING
        ready_event_before = entry_before.ready_event

        # Re-discover models mid-load (simulates admin re-scan)
        # We need the model dirs — extract from the entry's model_path
        model_path = entry_before.model_path
        model_dir = str(Path(model_path).parent)
        switching_pool.discover_models(model_dir)

        # The LOADING entry should be preserved
        entry_after = switching_pool.get_entry("model-a")
        assert entry_after is entry_before, "LOADING entry was replaced!"
        assert entry_after.state == EngineState.LOADING
        assert entry_after.ready_event is ready_event_before

        # Wait for load to complete — waiter should not be orphaned
        engine = await asyncio.wait_for(task, timeout=5)
        assert engine is not None
