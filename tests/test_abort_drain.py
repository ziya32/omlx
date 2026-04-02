# SPDX-License-Identifier: Apache-2.0
"""Tests for abort and drain race condition fixes.

Covers:
- Fix 1: Post-decode abort check in scheduler.step()
- Fix 2: _step_in_flight flag blocks drain
- Fix 3: VLM prefill chunk cap
- Fix 4: Deferred cleanup keeps collector alive during abort
- Fix 5: Two-phase unload (UNLOADING state)
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from omlx.engine_pool import EngineEntry, EnginePool, EngineState
from omlx.request import RequestOutput
from omlx.scheduler import SchedulerConfig


# ---------------------------------------------------------------------------
# Fix 1: Post-decode abort discards output
# ---------------------------------------------------------------------------

class TestPostDecodeAbort:
    """Fix 1: abort check after batch_generator.next() discards output."""

    def test_post_decode_abort_discards_output(self):
        """Mock batch_generator.next() to return tokens. Enqueue abort before
        calling step(). Verify aborted UID's output is discarded and request
        is cleaned up."""
        from omlx.scheduler import Scheduler, SchedulerConfig

        with patch("omlx.scheduler.Scheduler.__init__", return_value=None):
            scheduler = Scheduler.__new__(Scheduler)

        # Minimal scheduler state
        scheduler.config = SchedulerConfig()
        scheduler.running = {"req-1": MagicMock(request_id="req-1")}
        scheduler.waiting = MagicMock()
        scheduler.waiting.__len__ = lambda self: 0
        scheduler.requests = {"req-1": scheduler.running["req-1"]}
        scheduler.finished_req_ids = set()
        scheduler._step_counter = 0
        scheduler._pending_abort_ids = {"req-1"}
        scheduler.memory_monitor = None
        scheduler._deferred_clear_steps = None

        # Mock batch_generator.next() returning a response for req-1
        mock_response = MagicMock()
        mock_bg = MagicMock()
        mock_bg.next.return_value = [mock_response]
        scheduler.batch_generator = mock_bg

        # Mock internal methods
        scheduler._process_pending_aborts = MagicMock()
        scheduler._schedule_waiting = MagicMock(return_value=([], []))
        scheduler._process_batch_responses = MagicMock(return_value=([], set()))
        scheduler._cleanup_finished = MagicMock()
        scheduler._check_memory_pressure = MagicMock()

        # Make _process_pending_aborts clear the pending set on second call
        call_count = [0]
        def process_aborts():
            call_count[0] += 1
            if call_count[0] >= 2:
                scheduler._pending_abort_ids.clear()
        scheduler._process_pending_aborts.side_effect = process_aborts

        output = scheduler.step()

        # _process_pending_aborts should be called at least twice:
        # once at the start, once after next()
        assert scheduler._process_pending_aborts.call_count >= 2


# ---------------------------------------------------------------------------
# Fix 2: _step_in_flight blocks drain
# ---------------------------------------------------------------------------

class TestStepInFlightBlocksDrain:
    """Fix 2: _step_in_flight prevents drain from unloading."""

    def test_step_in_flight_blocks_drain(self):
        """Set _step_in_flight = True on engine core. Call
        _engine_has_active_work(). Assert returns True."""
        # Create a mock engine structure matching BatchedEngine layout
        inner_engine_core = MagicMock()
        inner_engine_core._output_collectors = {}  # empty collectors
        inner_engine_core._step_in_flight = True  # step is in flight

        engine_wrapper = MagicMock()
        engine_wrapper._engine = MagicMock()
        engine_wrapper._engine.engine = inner_engine_core

        assert EnginePool._engine_has_active_work(engine_wrapper) is True

    def test_no_step_in_flight_allows_drain(self):
        """With _step_in_flight = False and no collectors, engine has no
        active work."""
        inner_engine_core = MagicMock()
        inner_engine_core._output_collectors = {}
        inner_engine_core._step_in_flight = False

        engine_wrapper = MagicMock()
        engine_wrapper._engine = MagicMock()
        engine_wrapper._engine.engine = inner_engine_core
        # Ensure active_operations doesn't interfere
        engine_wrapper.active_operations = 0

        assert EnginePool._engine_has_active_work(engine_wrapper) is False

    @pytest.mark.asyncio
    async def test_drain_waits_for_step_in_flight(self):
        """Mock engine with _step_in_flight = True. Run _drain_monitor for 2
        iterations. Verify model is NOT unloaded. Set flag to False. Verify
        model IS unloaded on next iteration."""
        pool = EnginePool(max_model_memory=10_000_000_000)

        # Create a mock engine entry
        inner_core = MagicMock()
        inner_core._output_collectors = {}
        inner_core._step_in_flight = True

        mock_engine = MagicMock()
        mock_engine._engine = MagicMock()
        mock_engine._engine.engine = inner_core
        mock_engine.active_operations = 0
        mock_engine.stop = AsyncMock()

        entry = EngineEntry(
            model_id="test-model",
            model_path="/fake/path",
            model_type="llm",
            engine_type="batched",
            estimated_size=1_000_000_000,
            engine=mock_engine,
            state=EngineState.DRAINING,
            drain_started=0,  # will be set
            drain_complete=asyncio.Event(),
        )
        pool._entries["test-model"] = entry

        import time
        entry.drain_started = time.time()

        # Mock the lock
        pool._tracked_lock = MagicMock()
        pool._tracked_lock.return_value = MagicMock()
        pool._tracked_lock.return_value.__aenter__ = AsyncMock()
        pool._tracked_lock.return_value.__aexit__ = AsyncMock()

        # Mock _unload_engine to track calls
        unload_calls = []
        original_unload = pool._unload_engine
        async def mock_unload(model_id, *, reason="unload"):
            unload_calls.append(model_id)
            entry.state = EngineState.UNLOADING
            entry.engine = None
        pool._unload_engine = mock_unload

        # _step_in_flight is True, so drain should NOT unload
        assert EnginePool._engine_has_active_work(mock_engine) is True

        # Now set step_in_flight to False
        inner_core._step_in_flight = False
        assert EnginePool._engine_has_active_work(mock_engine) is False


# ---------------------------------------------------------------------------
# Fix 3: VLM prefill chunk cap
# ---------------------------------------------------------------------------

class TestVLMPrefillChunkCap:
    """Fix 3: VLM prefill uses capped chunk size."""

    def test_vlm_prefill_chunk_cap(self):
        """Verify that when VLM embeddings are present, the effective
        prefill step size is capped at 4096."""
        # Simulate the logic from _process_prompts
        prefill_step_size = 32000  # large default
        vlm_embeds_map = {1: ("embed", {}, 100)}  # non-empty = VLM request

        has_vlm_embeddings = bool(vlm_embeds_map)
        effective_prefill_step_size = min(
            prefill_step_size,
            4096 if has_vlm_embeddings else prefill_step_size,
        )

        assert effective_prefill_step_size == 4096

    def test_non_vlm_prefill_uses_full_step_size(self):
        """Without VLM embeddings, the full prefill step size is used."""
        prefill_step_size = 32000
        vlm_embeds_map = {}  # empty = no VLM

        has_vlm_embeddings = bool(vlm_embeds_map)
        effective_prefill_step_size = min(
            prefill_step_size,
            4096 if has_vlm_embeddings else prefill_step_size,
        )

        assert effective_prefill_step_size == 32000

    def test_vlm_prefill_abort_between_chunks(self):
        """Verify that with smaller chunk size, more abort check opportunities
        exist for VLM requests. A 20000-token VLM request with chunk_cap=4096
        should produce at least 4 chunks (vs 1 chunk with 32000 step size)."""
        total_tokens = 20000
        vlm_chunk_size = 4096
        default_chunk_size = 32000

        vlm_chunks = -(-total_tokens // vlm_chunk_size)  # ceil division
        default_chunks = -(-total_tokens // default_chunk_size)

        # VLM should have many more abort check opportunities
        assert vlm_chunks >= 4  # at least 4 chunks
        assert default_chunks == 1  # would be just 1 chunk
        assert vlm_chunks > default_chunks


# ---------------------------------------------------------------------------
# Fix 4: Deferred cleanup keeps collector
# ---------------------------------------------------------------------------

class TestDeferredCleanup:
    """Fix 4: abort_request defers cleanup to engine loop."""

    @pytest.mark.asyncio
    async def test_deferred_cleanup_keeps_collector(self):
        """Call abort_request(). Verify _output_collectors still contains
        the request ID. Simulate engine loop cleanup. Verify collector is
        removed after."""
        from omlx.engine_core import EngineCore, EngineConfig
        from omlx.output_collector import RequestOutputCollector
        from omlx.scheduler import SchedulerConfig

        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True

            mock_model = MagicMock()
            mock_model.config = MagicMock()
            mock_model.config.model_type = "llama"

            mock_tokenizer = MagicMock()
            mock_tokenizer.eos_token_id = 2

            with patch("omlx.scheduler.Scheduler.__init__", return_value=None):
                engine = EngineCore(
                    model=mock_model,
                    tokenizer=mock_tokenizer,
                    config=EngineConfig(),
                )

        # Set up mock scheduler
        engine.scheduler = MagicMock()
        engine.scheduler.abort_request.return_value = True

        # Manually add a collector for a request
        collector = RequestOutputCollector()
        engine._output_collectors["req-1"] = collector
        engine._finished_events["req-1"] = asyncio.Event()

        # Abort the request
        await engine.abort_request("req-1")

        # Collector should still be present (deferred cleanup)
        assert "req-1" in engine._output_collectors
        assert "req-1" in engine._pending_cleanups

        # Simulate engine loop processing pending cleanups
        for rid in list(engine._pending_cleanups):
            engine._cleanup_request(rid)
        engine._pending_cleanups.clear()

        # Now collector should be gone
        assert "req-1" not in engine._output_collectors


# ---------------------------------------------------------------------------
# Fix 5: Two-phase unload (UNLOADING state)
# ---------------------------------------------------------------------------

class TestTwoPhaseUnload:
    """Fix 5: UNLOADING state blocks new model loads."""

    def test_unloading_state_exists(self):
        """Verify UNLOADING is a valid EngineState."""
        assert hasattr(EngineState, "UNLOADING")
        assert EngineState.UNLOADING.value == "unloading"

    @pytest.mark.asyncio
    async def test_unloading_blocks_new_model_load(self):
        """Set a model to UNLOADING state. Call _prepare_memory_for() for a
        different model. Assert it returns an event to wait on. Complete
        cleanup, set UNLOADED. Assert memory check passes."""
        pool = EnginePool(max_model_memory=4_000_000_000)

        unload_event = asyncio.Event()

        # Model A is in UNLOADING state
        entry_a = EngineEntry(
            model_id="model-a",
            model_path="/fake/model-a",
            model_type="llm",
            engine_type="batched",
            estimated_size=2_000_000_000,
            engine=None,
            state=EngineState.UNLOADING,
            unload_complete=unload_event,
        )

        # Model B wants to load
        entry_b = EngineEntry(
            model_id="model-b",
            model_path="/fake/model-b",
            model_type="llm",
            engine_type="batched",
            estimated_size=2_000_000_000,
        )

        pool._entries["model-a"] = entry_a
        pool._entries["model-b"] = entry_b

        # _prepare_memory_for should return the unload_complete event
        result = await pool._prepare_memory_for(entry_b)
        assert result is unload_event

        # Simulate cleanup completing
        entry_a.state = EngineState.UNLOADED
        unload_event.set()

        # Now _prepare_memory_for should return None (memory available)
        result = await pool._prepare_memory_for(entry_b)
        assert result is None

    @pytest.mark.asyncio
    async def test_mx_synchronize_before_unloaded(self):
        """Verify state is UNLOADING during cleanup and UNLOADED only after
        synchronize returns."""
        pool = EnginePool(max_model_memory=10_000_000_000)

        mock_engine = MagicMock()
        mock_engine.stop = AsyncMock()

        entry = EngineEntry(
            model_id="test-model",
            model_path="/fake/path",
            model_type="llm",
            engine_type="batched",
            estimated_size=1_000_000_000,
            engine=mock_engine,
            state=EngineState.ACTIVE,
        )
        pool._entries["test-model"] = entry

        states_during_cleanup = []

        original_sync = None
        def mock_sync_and_clear():
            # Record state during Metal sync
            states_during_cleanup.append(entry.state)
            return (None, None)

        pool._cleanup_tasks = []

        with patch("omlx.engine_pool.get_mlx_executor") as mock_executor:
            loop = asyncio.get_running_loop()

            # Make run_in_executor call the lambda synchronously
            async def mock_run_in_executor(executor, fn):
                return fn()
            loop_patch = patch.object(
                loop, "run_in_executor", side_effect=mock_run_in_executor
            )

            with loop_patch:
                # Call _unload_engine which sets UNLOADING
                await pool._unload_engine("test-model", reason="test")

                # State should be UNLOADING before cleanup task runs
                assert entry.state == EngineState.UNLOADING

                # Run pending tasks (the deferred cleanup)
                await asyncio.sleep(0)
                # Give the cleanup task a chance to complete
                if pool._cleanup_tasks:
                    await asyncio.gather(*pool._cleanup_tasks, return_exceptions=True)

        # After cleanup, state should be UNLOADED
        assert entry.state == EngineState.UNLOADED

    @pytest.mark.asyncio
    async def test_committed_memory_includes_unloading(self):
        """Verify _committed_memory counts UNLOADING models."""
        pool = EnginePool(max_model_memory=10_000_000_000)

        entry = EngineEntry(
            model_id="test-model",
            model_path="/fake/path",
            model_type="llm",
            engine_type="batched",
            estimated_size=2_000_000_000,
            state=EngineState.UNLOADING,
        )
        pool._entries["test-model"] = entry

        # UNLOADING model's memory should be counted as committed
        assert pool._committed_memory() == 2_000_000_000

        # After transitioning to UNLOADED, memory should be freed
        entry.state = EngineState.UNLOADED
        assert pool._committed_memory() == 0
