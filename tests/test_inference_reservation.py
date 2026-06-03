# SPDX-License-Identifier: Apache-2.0
"""Model-free tests for the in-flight inference working-set reservation.

Covers the OOM-crash fix (docs/design/fix-oom-crash-06--260602.md):
- §3a  EnginePool.reserve_inference admission loop + accounting helpers
- §3b  load admission made reservation-aware
- §3c  _pick_inference_victim self/pinned exclusion
- §3d  BaseNonStreamingEngine estimate hook + TTS estimates
- §3e  scheduler ↔ pool cross-visibility

All engines are mocked and ``get_effective_metal_cap_bytes`` /
``_committed_memory`` are monkeypatched — NO model is ever loaded and no
Metal weights are allocated (the 64 GB OOM-guard forbids real models in the
agent harness). Mirrors tests/test_engine_pool.py and
tests/test_process_memory_enforcer.py.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from omlx.engine.base import BaseNonStreamingEngine
from omlx.engine.tts import TTSEngine
from omlx.engine_pool import (
    _INFERENCE_MARGIN_BYTES,
    EngineEntry,
    EnginePool,
    EngineState,
)

GiB = 1024**3
MiB = 1024**2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pool(max_wait_timeout: float = 5.0) -> EnginePool:
    pool = EnginePool(max_wait_timeout=max_wait_timeout)
    # Pre-load admission ceiling: high enough that the load tests control it
    # explicitly. Inference admission uses the Metal wall, not this.
    pool._get_final_ceiling = lambda: 0
    return pool


def _add_entry(
    pool: EnginePool,
    model_id: str,
    *,
    size: int,
    state: EngineState = EngineState.ACTIVE,
    is_pinned: bool = False,
    busy: bool = False,
    last_access: float = 1.0,
) -> EngineEntry:
    """Register a loaded mock engine entry on the pool."""
    engine = MagicMock()
    engine.has_active_requests = MagicMock(return_value=busy)
    engine.stop = AsyncMock()
    entry = EngineEntry(
        model_id=model_id,
        model_path=f"/fake/{model_id}",
        model_type="audio_tts",
        engine_type="audio_tts",
        estimated_size=size,
        is_pinned=is_pinned,
        state=state,
    )
    entry.engine = engine if state == EngineState.ACTIVE else None
    entry.last_access = last_access
    entry.active_uses = 1 if busy else 0
    pool._entries[model_id] = entry
    return entry


def _patch_cap(monkeypatch, cap_bytes: int) -> None:
    """Monkeypatch the Metal wall accessor (imported lazily in _wall_budget)."""
    monkeypatch.setattr(
        "omlx.process_memory_enforcer.get_effective_metal_cap_bytes",
        lambda: cap_bytes,
    )


def _stub_phase1_unload(pool: EnginePool) -> None:
    """Replace _unload_engine with a model-free Phase-1-only transition.

    Real _unload_engine schedules _deferred_engine_cleanup, which calls
    mx.clear_cache (touches Metal). Here we faithfully reproduce ONLY the
    synchronous Phase-1 effects (engine=None, state=UNLOADING, fresh
    unload_complete) — exactly what reserve_inference observes — without any
    MLX work. Tests then drive the deferred completion explicitly.
    """

    async def _fake_unload(model_id: str, *, reason: str = "unload") -> None:
        entry = pool._entries.get(model_id)
        if entry is None or entry.engine is None:
            return
        entry.engine = None
        entry.last_access = 0.0
        entry.unload_complete = asyncio.Event()
        entry.state = EngineState.UNLOADING

    pool._unload_engine = AsyncMock(side_effect=_fake_unload)


def _complete_unload(pool: EnginePool, model_id: str) -> None:
    """Simulate the deferred cleanup finishing: drop weights, fire the event."""
    entry = pool._entries[model_id]
    entry.state = EngineState.UNLOADED
    assert entry.unload_complete is not None
    entry.unload_complete.set()


# ---------------------------------------------------------------------------
# §3a / §5 — accounting helpers + no-op path
# ---------------------------------------------------------------------------


class TestReservationAccounting:
    def test_committed_plus_reservations(self, monkeypatch):
        pool = _make_pool()
        _add_entry(pool, "a", size=10 * GiB)
        assert pool._committed_memory() == 10 * GiB
        pool._inflight_reservations = 3 * GiB
        assert pool._committed_plus_reservations() == 13 * GiB

    def test_wall_budget_subtracts_margin(self, monkeypatch):
        pool = _make_pool()
        _patch_cap(monkeypatch, 50 * GiB)
        assert pool._wall_budget() == 50 * GiB - _INFERENCE_MARGIN_BYTES

    def test_wall_budget_zero_when_no_cap(self, monkeypatch):
        pool = _make_pool()
        _patch_cap(monkeypatch, 0)
        assert pool._wall_budget() == 0

    def test_reservable_free(self, monkeypatch):
        pool = _make_pool()
        _patch_cap(monkeypatch, 50 * GiB)
        _add_entry(pool, "a", size=40 * GiB)
        pool._inflight_reservations = 2 * GiB
        # free = (cap - margin) - committed(40) - reservations(2)
        assert pool._reservable_free() == (
            50 * GiB - _INFERENCE_MARGIN_BYTES - 40 * GiB - 2 * GiB
        )

    @pytest.mark.asyncio
    async def test_noop_when_no_cap(self, monkeypatch):
        """No Metal cap ⇒ strict no-op: no lock, no state, just yield."""
        pool = _make_pool()
        _patch_cap(monkeypatch, 0)
        ran = False
        async with pool.reserve_inference("a", 4 * GiB):
            ran = True
            assert not pool._lock.locked()  # strict no-op took no lock
        assert ran
        assert pool._inflight_reservations == 0

    @pytest.mark.asyncio
    async def test_noop_when_est_zero(self, monkeypatch):
        """est<=0 ⇒ strict no-op even with a real cap."""
        pool = _make_pool()
        _patch_cap(monkeypatch, 50 * GiB)
        async with pool.reserve_inference("a", 0):
            assert not pool._lock.locked()
        assert pool._inflight_reservations == 0


# ---------------------------------------------------------------------------
# §3a / §8 — fit path + concurrency
# ---------------------------------------------------------------------------


class TestFitPath:
    @pytest.mark.asyncio
    async def test_single_op_fits(self, monkeypatch):
        pool = _make_pool()
        _patch_cap(monkeypatch, 50 * GiB)
        _add_entry(pool, "a", size=40 * GiB)
        async with pool.reserve_inference("a", 4 * GiB):
            assert pool._inflight_reservations == 4 * GiB
        assert pool._inflight_reservations == 0

    @pytest.mark.asyncio
    async def test_two_fitting_ops_concurrent_no_wait(self, monkeypatch):
        """Two ops that both fit run concurrently; neither evicts/waits."""
        pool = _make_pool()
        _patch_cap(monkeypatch, 60 * GiB)  # budget 58 GiB
        _add_entry(pool, "a", size=40 * GiB)
        # No victim must ever be picked on the healthy path.
        pool._unload_engine = AsyncMock(
            side_effect=AssertionError("must not evict on fit path")
        )
        pool._start_drain = MagicMock(
            side_effect=AssertionError("must not drain on fit path")
        )

        gate = asyncio.Event()
        peak = {"v": 0}

        async def _op(est):
            async with pool.reserve_inference("a", est):
                peak["v"] = max(peak["v"], pool._inflight_reservations)
                await gate.wait()

        t1 = asyncio.create_task(_op(4 * GiB))
        t2 = asyncio.create_task(_op(5 * GiB))
        await asyncio.sleep(0.05)
        # Both admitted concurrently: reservations stacked.
        assert pool._inflight_reservations == 9 * GiB
        gate.set()
        await asyncio.gather(t1, t2)
        assert peak["v"] == 9 * GiB
        assert pool._inflight_reservations == 0


# ---------------------------------------------------------------------------
# §3a — idle evict awaits the DEFERRED cleanup (the key correctness point)
# ---------------------------------------------------------------------------


class TestIdleEvictAwaitsCleanup:
    @pytest.mark.asyncio
    async def test_does_not_admit_until_unload_complete(self, monkeypatch):
        """B must NOT admit best-effort while the evicted victim's weights are
        still committed (UNLOADING). It awaits unload_complete, then re-loops.
        """
        pool = _make_pool()
        _patch_cap(monkeypatch, 56 * GiB)  # budget 54 GiB
        # 48.47 committed: pinned 'big' + idle 'embed' + 'tts'
        _add_entry(pool, "big", size=int(37.84 * GiB), is_pinned=True)
        _add_entry(pool, "embed", size=int(6.29 * GiB), last_access=1.0)
        _add_entry(pool, "tts", size=int(4.34 * GiB), last_access=5.0)
        _stub_phase1_unload(pool)

        # 'tts' reserves first and fits (free ≈ 5.19 GiB).
        a_cm = pool.reserve_inference("tts", 4 * GiB)
        await a_cm.__aenter__()
        assert pool._inflight_reservations == 4 * GiB

        # 'tts' op B now needs 4 GiB but only ~1.19 free → must evict 'embed'.
        admitted = asyncio.Event()

        async def _op_b():
            async with pool.reserve_inference("tts", 4 * GiB):
                admitted.set()
                await asyncio.sleep(0)

        tb = asyncio.create_task(_op_b())
        await asyncio.sleep(0.05)

        # B picked 'embed' as victim (Phase-1 unload ran), but its weights are
        # STILL committed (UNLOADING) → B must be blocked, NOT admitted.
        pool._unload_engine.assert_awaited()
        assert pool._entries["embed"].state == EngineState.UNLOADING
        assert not admitted.is_set()
        # Reservation count must still be just A's 4 GiB (B has not added).
        assert pool._inflight_reservations == 4 * GiB

        # Now the deferred cleanup completes → weights drop → B wakes, re-loops,
        # fits, and admits.
        _complete_unload(pool, "embed")
        await asyncio.wait_for(admitted.wait(), timeout=2)
        assert pool._inflight_reservations == 8 * GiB  # A + B

        await a_cm.__aexit__(None, None, None)
        await tb
        assert pool._inflight_reservations == 0
        assert pool._timeout_counter == 0


# ---------------------------------------------------------------------------
# §3b — load admission sees the reservation
# ---------------------------------------------------------------------------


class TestLoadSeesReservation:
    @pytest.mark.asyncio
    async def test_load_evicts_when_reservation_blocks_fit(self, monkeypatch):
        """A non-pinned load that fits WITHOUT but not WITH a 4 GiB inflight
        reservation must evict/wait (§3b), not commit weights.
        """
        pool = _make_pool()
        ceiling = 50 * GiB
        pool._get_final_ceiling = lambda: ceiling
        # committed weights = 44 GiB; a new 4 GiB load fits (48 <= 50)...
        _add_entry(pool, "resident", size=44 * GiB, last_access=1.0)
        new_entry = EngineEntry(
            model_id="loadme",
            model_path="/fake/loadme",
            model_type="audio_tts",  # no KV headroom branch
            engine_type="audio_tts",
            estimated_size=4 * GiB,
            state=EngineState.UNLOADED,
        )
        pool._entries["loadme"] = new_entry

        # Without reservation: load admits straight away (no victim).
        ev = await pool._prepare_memory_for(new_entry)
        assert ev is None

        # With a 4 GiB inflight reservation: 44 + 4(resv) + 4(load) = 52 > 50
        # → the while-gate must trigger eviction of 'resident'.
        pool._inflight_reservations = 4 * GiB
        _stub_phase1_unload(pool)
        ev = await pool._prepare_memory_for(new_entry)
        # 'resident' is idle → unloaded immediately (Phase-1); loop then fits
        # at committed=0(+UNLOADING weights still counted until cleanup). The
        # idle path unloads and re-loops; with our stub the UNLOADING weights
        # still count, so it returns the unload_complete event to wait on.
        pool._unload_engine.assert_awaited()
        assert pool._entries["resident"].state == EngineState.UNLOADING
        assert ev is pool._entries["resident"].unload_complete


# ---------------------------------------------------------------------------
# §3a / §6 — release on every path
# ---------------------------------------------------------------------------


class TestReleaseEveryPath:
    @pytest.mark.asyncio
    async def test_release_on_normal(self, monkeypatch):
        pool = _make_pool()
        _patch_cap(monkeypatch, 50 * GiB)
        _add_entry(pool, "a", size=40 * GiB)
        async with pool.reserve_inference("a", 4 * GiB):
            pass
        assert pool._inflight_reservations == 0

    @pytest.mark.asyncio
    async def test_release_on_exception(self, monkeypatch):
        pool = _make_pool()
        _patch_cap(monkeypatch, 50 * GiB)
        _add_entry(pool, "a", size=40 * GiB)
        with pytest.raises(ValueError):
            async with pool.reserve_inference("a", 4 * GiB):
                raise ValueError("boom")
        assert pool._inflight_reservations == 0

    @pytest.mark.asyncio
    async def test_release_on_cancel(self, monkeypatch):
        pool = _make_pool()
        _patch_cap(monkeypatch, 50 * GiB)
        _add_entry(pool, "a", size=40 * GiB)

        started = asyncio.Event()

        async def _op():
            async with pool.reserve_inference("a", 4 * GiB):
                started.set()
                await asyncio.sleep(100)

        t = asyncio.create_task(_op())
        await asyncio.wait_for(started.wait(), timeout=2)
        assert pool._inflight_reservations == 4 * GiB
        t.cancel()
        with pytest.raises(asyncio.CancelledError):
            await t
        assert pool._inflight_reservations == 0

    @pytest.mark.asyncio
    async def test_timeout_proceeds_best_effort_then_releases(self, monkeypatch):
        """A wedged reclaim (event never fires) ⇒ wait_for times out ⇒ proceed
        best-effort (NOT a failure), reservation taken then released.
        """
        pool = _make_pool(max_wait_timeout=0.1)
        _patch_cap(monkeypatch, 56 * GiB)  # budget 54
        _add_entry(pool, "big", size=int(37.84 * GiB), is_pinned=True)
        _add_entry(pool, "embed", size=int(6.29 * GiB))
        _add_entry(pool, "tts", size=int(4.34 * GiB))

        # Hold a reservation so the next op is over budget, and make eviction
        # produce an event that NEVER fires (wedged reclaim).
        pool._inflight_reservations = 4 * GiB

        async def _wedged_unload(model_id, *, reason="unload"):
            entry = pool._entries[model_id]
            entry.engine = None
            entry.unload_complete = asyncio.Event()  # never set
            entry.state = EngineState.UNLOADING

        pool._unload_engine = AsyncMock(side_effect=_wedged_unload)

        async with pool.reserve_inference("tts", 4 * GiB):
            # Proceeded best-effort despite the wedge.
            assert pool._inflight_reservations == 8 * GiB
        assert pool._inflight_reservations == 4 * GiB  # only the pre-set one
        # The reserve_inference timeout uses asyncio.wait_for; the pool's own
        # _record_timeout is only for lock acquisition, so no lock timeout.
        assert pool._timeout_counter == 0


# ---------------------------------------------------------------------------
# §3c / §7 — pinned never evicted + self-exclusion
# ---------------------------------------------------------------------------


class TestVictimSelection:
    def test_pick_skips_self(self):
        pool = _make_pool()
        # Only one non-pinned model, and it IS the op's model → never a victim.
        _add_entry(pool, "tts", size=4 * GiB)
        assert pool._find_drain_or_evict_candidate() == "tts"
        assert pool._pick_inference_victim("tts") is None

    def test_pick_returns_other_nonpinned(self):
        pool = _make_pool()
        _add_entry(pool, "tts", size=4 * GiB, last_access=5.0)
        _add_entry(pool, "embed", size=6 * GiB, last_access=1.0)
        # 'embed' is older → LRU victim, and it's not self.
        assert pool._pick_inference_victim("tts") == "embed"

    def test_pick_never_returns_pinned(self):
        pool = _make_pool()
        _add_entry(pool, "tts", size=4 * GiB, last_access=5.0)
        _add_entry(pool, "big", size=37 * GiB, is_pinned=True, last_access=1.0)
        # Only non-pinned is 'tts' itself → no victim.
        assert pool._pick_inference_victim("tts") is None

    @pytest.mark.asyncio
    async def test_best_effort_when_only_pinned_and_self(self, monkeypatch):
        """All-pinned-plus-self config with nothing freeing ⇒ last-resort
        best-effort admit (logs loudly), never blocks forever.
        """
        pool = _make_pool()
        _patch_cap(monkeypatch, 40 * GiB)  # budget 38
        _add_entry(pool, "big", size=37 * GiB, is_pinned=True)
        _add_entry(pool, "tts", size=4 * GiB)  # self; 37+4 already > 38 budget
        pool._unload_engine = AsyncMock(
            side_effect=AssertionError("pinned/self must not be evicted")
        )
        async with pool.reserve_inference("tts", 4 * GiB):
            assert pool._inflight_reservations == 4 * GiB
        assert pool._inflight_reservations == 0


# ---------------------------------------------------------------------------
# §3d — engine estimate hooks
# ---------------------------------------------------------------------------


class TestEstimateHooks:
    def test_base_default_zero(self):
        eng = MagicMock(spec=BaseNonStreamingEngine)
        # The real default returns 0.
        assert BaseNonStreamingEngine.estimate_working_set_bytes(eng) == 0

    def test_tts_default_max_tokens(self):
        eng = TTSEngine("fake-tts")
        # No text/max_tokens → default 4096 codes × 1 MiB = 4 GiB.
        assert eng.estimate_working_set_bytes() == 4096 * MiB

    def test_tts_short_text_reserves_less(self):
        eng = TTSEngine("fake-tts")
        # Short text caps codes far below default: len*6 = 60 → max(75,60)=75
        # → clamped up to the 512 floor.
        est = eng.estimate_working_set_bytes(text="hello world", max_tokens=4096)
        assert est == 512 * MiB

    def test_tts_clamp_ceiling(self):
        eng = TTSEngine("fake-tts")
        # Huge max_tokens clamps at 8192.
        est = eng.estimate_working_set_bytes(text="x" * 100000, max_tokens=100000)
        assert est == 8192 * MiB

    def test_tts_streaming_under_reserves(self):
        eng = TTSEngine("fake-tts")
        stream = eng._estimate_streaming_working_set_bytes()
        full = eng.estimate_working_set_bytes()
        assert stream == 512 * MiB
        assert stream < full  # streaming reserves far less than all-at-once

    def test_forward_estimate_from_weights(self):
        """STT/STS/embedding/reranker default ≈ 0.08 × weights via the pool."""
        eng = TTSEngine("m")  # any BaseNonStreamingEngine subclass
        pool = _make_pool()
        _add_entry(pool, "m", size=10 * GiB)
        eng._pool = pool
        import math
        assert eng._estimate_forward_working_set_bytes() == math.ceil(
            10 * GiB * 0.08
        )

    def test_forward_estimate_no_pool_is_zero(self):
        eng = TTSEngine("m")
        assert eng._pool is None
        assert eng._estimate_forward_working_set_bytes() == 0


# ---------------------------------------------------------------------------
# §3e — scheduler ↔ pool cross-visibility
# ---------------------------------------------------------------------------


class TestSchedulerCrossVisibility:
    @staticmethod
    def _make_scheduler(mock_model, mock_tokenizer):
        """A real Scheduler over mocks, primed at the predictive-guard edge.

        current 10 + concurrent (2×0.8=1.6) = 11.6 GiB < soft 20 GiB, so with
        NO reservation the guard does NOT defer; a reservation tips it over.
        """
        from omlx.scheduler import Scheduler, SchedulerConfig

        sched = Scheduler(
            model=mock_model,
            tokenizer=mock_tokenizer,
            config=SchedulerConfig(model_name="m", max_num_seqs=8),
        )
        sched._prefill_memory_guard = True
        sched._memory_limit_bytes = 20 * GiB           # soft
        sched._model_size_bytes = 10 * GiB             # per-req est = 0.8 GiB
        sched._current_tick_memory_bytes = lambda: 10 * GiB
        sched.memory_monitor = None
        # One running request so the guard is armed (first req always passes).
        sched.running = {"r": MagicMock(cached_tokens=0)}
        # One waiting request with no cached tokens.
        wait_req = MagicMock(cached_tokens=0, sampling_params=MagicMock())
        sched.waiting.append(wait_req)
        # Stub batch-generator creation so the non-defer branch is harmless
        # (it pops, finds batch_generator None, puts the request back, breaks)
        # — no model work, no Metal. We detect the non-defer branch by whether
        # _ensure_batch_generator was called.
        sched.batch_generator = None
        sched._ensure_batch_generator = MagicMock()
        return sched, wait_req

    def test_guard_defers_only_with_reservation(
        self, mock_model, mock_tokenizer
    ):
        """REAL Scheduler._schedule_waiting(): the §3e reservation term flips
        the predictive guard from admit → defer. Same setup, only the pool's
        in-flight reservation changes.
        """
        # No reservation → guard does NOT defer (proceeds to pop+ensure).
        sched, wait_req = self._make_scheduler(mock_model, mock_tokenizer)
        sched._get_inflight_reservations = lambda: 0
        scheduled, _ = sched._schedule_waiting()
        assert sched._ensure_batch_generator.called  # admitted past the guard
        assert wait_req in sched.waiting  # put back (batch_generator is None)

        # Big reservation (a live TTS decode holding 12 GiB) → 10+1.6+12=23.6
        # > soft 20 → guard DEFERS: request untouched, ensure never called.
        sched, wait_req = self._make_scheduler(mock_model, mock_tokenizer)
        sched._get_inflight_reservations = lambda: 12 * GiB
        scheduled, _ = sched._schedule_waiting()
        assert scheduled == []
        assert not sched._ensure_batch_generator.called  # deferred at the guard
        assert sched.waiting[0] is wait_req

    def test_scheduler_default_reservation_callback_is_none(
        self, mock_model, mock_tokenizer
    ):
        """Out of the box (no enforcer) the guard adds 0 — unchanged behavior."""
        from omlx.scheduler import Scheduler

        sched = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        assert sched._get_inflight_reservations is None

    def test_enforcer_wires_callback_and_sums_inflight(
        self, mock_model, mock_tokenizer
    ):
        """REAL ProcessMemoryEnforcer: _propagate_memory_limit wires each
        scheduler's reservation reader to the pool live, and
        get_scheduler_inflight_bytes sums 0.08 × size × n_running (§3e).
        """
        from omlx.process_memory_enforcer import ProcessMemoryEnforcer
        from omlx.scheduler import Scheduler, SchedulerConfig

        pool = _make_pool()
        sched = Scheduler(
            model=mock_model,
            tokenizer=mock_tokenizer,
            config=SchedulerConfig(model_name="m"),
        )
        sched.running = {"a": MagicMock(), "b": MagicMock()}  # n_running = 2
        # Reach the scheduler via _resolve_scheduler: entry.engine.scheduler.
        entry = _add_entry(pool, "m", size=10 * GiB)
        entry.engine.scheduler = sched

        enforcer = ProcessMemoryEnforcer(engine_pool=pool)
        enforcer._get_hard_limit_bytes = lambda: 30 * GiB

        # Wire the callbacks (pool→scheduler reservation reader).
        enforcer._propagate_memory_limit()
        assert sched._get_inflight_reservations is not None
        assert sched._model_size_bytes == 10 * GiB  # also plumbed here
        # Reservation reader is LIVE against the pool.
        pool._inflight_reservations = 3 * GiB
        assert sched._get_inflight_reservations() == 3 * GiB
        pool._inflight_reservations = 0
        assert sched._get_inflight_reservations() == 0

        # get_scheduler_inflight_bytes = 0.08 × 10 GiB × 2 running.
        expected = int(10 * GiB * 0.08) * 2
        assert enforcer.get_scheduler_inflight_bytes() == expected

    def test_enforcer_sums_zero_when_idle(self, mock_model, mock_tokenizer):
        from omlx.process_memory_enforcer import ProcessMemoryEnforcer
        from omlx.scheduler import Scheduler

        pool = _make_pool()
        sched = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        sched.running = {}  # nothing running ⇒ 0 transient
        entry = _add_entry(pool, "m", size=10 * GiB)
        entry.engine.scheduler = sched
        enforcer = ProcessMemoryEnforcer(engine_pool=pool)
        enforcer._get_hard_limit_bytes = lambda: 30 * GiB
        enforcer._propagate_memory_limit()
        assert enforcer.get_scheduler_inflight_bytes() == 0

    @pytest.mark.asyncio
    async def test_reserve_inference_sees_scheduler_inflight(self, monkeypatch):
        """reserve_inference reads the enforcer's scheduler in-flight bytes via
        _reservable_free and waits/evicts instead of stacking over the wall.
        """
        pool = _make_pool()
        _patch_cap(monkeypatch, 56 * GiB)  # budget 54
        _add_entry(pool, "big", size=int(37.84 * GiB), is_pinned=True)
        _add_entry(pool, "tts", size=int(4.34 * GiB))

        # Fake enforcer exposing a live scheduler transient.
        enforcer = MagicMock()
        enforcer.get_scheduler_inflight_bytes = MagicMock(return_value=10 * GiB)
        pool._process_memory_enforcer = enforcer

        # budget 54 − committed 42.18 − scheduler 10 = ~1.82 GiB free.
        free = pool._reservable_free()
        assert free < 4 * GiB  # a 4 GiB TTS would NOT fit because of scheduler

        # Confirm the scheduler in-flight is what shrank the room: removing the
        # scheduler term restores exactly 10 GiB of free budget.
        assert pool._scheduler_inflight_bytes() == 10 * GiB
        no_sched_free = (
            pool._wall_budget() - pool._committed_plus_reservations()
        )
        assert no_sched_free - free == 10 * GiB


# ---------------------------------------------------------------------------
# Regression — exclusive-headroom-drain livelock (2026-06-02)
#
# An in-flight non-streaming op whose OWN engine is being drained/unloaded (an
# exclusive model's _clear_for_exclusive headroom drain) MUST NOT block in
# reserve_inference waiting on its own reclaim: it holds that engine's
# active_uses lease, so the drain cannot complete until the op finishes —
# waiting deadlocks the op against its own completion, broken only by the 300 s
# _max_wait_timeout. The op must reserve-and-run so the drain can proceed.
# ---------------------------------------------------------------------------
class TestDrainingSelfNoDeadlock:
    @pytest.mark.asyncio
    async def test_reserve_does_not_wait_on_own_draining_engine(self, monkeypatch):
        # Long timeout: a regression would hang ~30 s on the unset drain event.
        pool = _make_pool(max_wait_timeout=30.0)
        _patch_cap(monkeypatch, 50 * GiB)  # wall budget 48 GiB
        # A pinned model fills the budget so the op cannot "fit" → without the
        # guard it falls through to wait on a reclaim event.
        _add_entry(pool, "pinned-big", size=46 * GiB, is_pinned=True)
        # The op's OWN engine is mid-drain with an UNSET drain_complete (the
        # exclusive-headroom drain is waiting for active_uses → 0).
        own = _add_entry(pool, "m1", size=6 * GiB, state=EngineState.DRAINING)
        own.drain_complete = asyncio.Event()  # deliberately never set
        own.active_uses = 1  # our in-flight lease

        ran = {"v": False}

        async def _op():
            async with pool.reserve_inference("m1", 1 * GiB):
                ran["v"] = True

        # Must complete promptly (guard reserves-and-runs); a regression blocks
        # on own.drain_complete for the full 30 s max_wait_timeout.
        await asyncio.wait_for(_op(), timeout=2.0)
        assert ran["v"] is True
        assert pool._inflight_reservations == 0  # released on exit

    @pytest.mark.asyncio
    async def test_reserve_does_not_wait_on_own_unloading_engine(self, monkeypatch):
        pool = _make_pool(max_wait_timeout=30.0)
        _patch_cap(monkeypatch, 50 * GiB)
        _add_entry(pool, "pinned-big", size=46 * GiB, is_pinned=True)
        own = _add_entry(pool, "m1", size=6 * GiB, state=EngineState.ACTIVE)
        own.state = EngineState.UNLOADING
        own.unload_complete = asyncio.Event()  # never set
        own.active_uses = 1

        ran = {"v": False}

        async def _op():
            async with pool.reserve_inference("m1", 1 * GiB):
                ran["v"] = True

        await asyncio.wait_for(_op(), timeout=2.0)
        assert ran["v"] is True
        assert pool._inflight_reservations == 0

    def test_pending_reclaim_event_excludes_self(self):
        pool = _make_pool()
        own = _add_entry(pool, "m1", size=6 * GiB, state=EngineState.DRAINING)
        own.drain_complete = asyncio.Event()
        # Without exclusion the op's own drain is returned (the bug); with the
        # model_id excluded it is skipped so the caller proceeds best-effort.
        assert pool._pending_reclaim_event() is own.drain_complete
        assert pool._pending_reclaim_event("m1") is None

    def test_resolve_reservation_key_maps_model_path_to_key(self):
        # Engines reserve with model_name (full PATH); _entries is keyed by the
        # short model_id. The resolver must map path → key (else the guard /
        # self-skip / exclusion silently no-op — the real 2026-06-02 bug).
        pool = _make_pool()
        _add_entry(pool, "m1", size=GiB)  # _add_entry sets model_path="/fake/m1"
        assert pool._resolve_reservation_key("m1") == "m1"  # already a key
        assert pool._resolve_reservation_key("/fake/m1") == "m1"  # path → key
        assert pool._resolve_reservation_key("/unknown") == "/unknown"  # fallback

    @pytest.mark.asyncio
    async def test_reserve_with_full_path_resolves_own_draining_guard(self, monkeypatch):
        # The realistic failure: reserve with the FULL PATH while the entry is
        # keyed by the short id and is DRAINING. Resolution + guard must fire so
        # the op runs promptly (a regression hangs ~30s on its own drain).
        pool = _make_pool(max_wait_timeout=30.0)
        _patch_cap(monkeypatch, 50 * GiB)
        _add_entry(pool, "pinned-big", size=46 * GiB, is_pinned=True)
        own = _add_entry(pool, "m1", size=6 * GiB, state=EngineState.DRAINING)
        own.drain_complete = asyncio.Event()  # never set
        own.active_uses = 1

        ran = {"v": False}

        async def _op():
            async with pool.reserve_inference("/fake/m1", 1 * GiB):  # full PATH
                ran["v"] = True

        await asyncio.wait_for(_op(), timeout=2.0)
        assert ran["v"] is True
        assert pool._inflight_reservations == 0

    @pytest.mark.asyncio
    async def test_own_draining_op_evicts_victim_not_bypass(self, monkeypatch):
        # An op whose OWN engine is draining and is over budget must still EVICT
        # a non-pinned victim to make room before running — it must NOT bypass
        # the budget/eviction. The removed proceed-immediately guard DID bypass,
        # which let two TTS decode on top of a still-resident embedding and
        # SIGABRT'd (the 2026-06-02 Metal-OOM regression). It also must not wait
        # on its OWN drain (exclusion).
        pool = _make_pool(max_wait_timeout=5.0)
        _patch_cap(monkeypatch, 50 * GiB)  # budget = 50 - 4 (margin) = 46 GiB
        _add_entry(pool, "pinned-big", size=38 * GiB, is_pinned=True)
        _add_entry(pool, "embed", size=6 * GiB, state=EngineState.ACTIVE, last_access=0.5)
        own = _add_entry(pool, "tts", size=4 * GiB, state=EngineState.DRAINING)
        own.active_uses = 1  # our in-flight lease on the draining engine
        _stub_phase1_unload(pool)  # evict → UNLOADING + (deferred) unload_complete
        # committed 38+6+4 = 48 > budget 46 → must evict 'embed', not bypass.

        async def _op():
            async with pool.reserve_inference("/fake/tts", 1 * GiB):  # full PATH
                return "ran"

        task = asyncio.create_task(_op())
        await asyncio.sleep(0.05)
        # The op evicted the victim to make room — it did NOT bypass and decode
        # on top of it. (With the old guard, 'embed' would still be ACTIVE.)
        assert pool._entries["embed"].state == EngineState.UNLOADING
        _complete_unload(pool, "embed")  # finish the deferred cleanup
        result = await asyncio.wait_for(task, timeout=3.0)
        assert result == "ran"
        assert pool._inflight_reservations == 0

    def test_pending_reclaim_event_waits_for_loading_nonpinned(self):
        # A non-pinned model still LOADING is a pending reclaim (wait, then
        # evict). Fix for the single-shot OOM where a concurrent Embedding load
        # left an in-flight TTS with nothing to reclaim → best-effort → SIGABRT.
        pool = _make_pool()
        _add_entry(pool, "pinned-big", size=38 * GiB, is_pinned=True)
        loading = _add_entry(pool, "embed", size=8 * GiB, state=EngineState.LOADING)
        loading.ready_event = asyncio.Event()
        assert pool._pending_reclaim_event() is loading.ready_event
        # A LOADING *pinned* model is NOT returned (can't evict it).
        loading.is_pinned = True
        assert pool._pending_reclaim_event() is None

    @pytest.mark.asyncio
    async def test_reserve_waits_for_loading_model_then_evicts(self, monkeypatch):
        # The single-shot OOM: a concurrent model LOAD commits memory the
        # in-flight TTS op cannot reclaim yet (LOADING isn't evictable). The op
        # must WAIT for the load, then evict it — NOT best-effort into a Metal OOM.
        pool = _make_pool(max_wait_timeout=5.0)
        _patch_cap(monkeypatch, 50 * GiB)  # budget = 46 GiB
        _add_entry(pool, "pinned-big", size=38 * GiB, is_pinned=True)
        loading = _add_entry(pool, "embed", size=8 * GiB, state=EngineState.LOADING)
        loading.ready_event = asyncio.Event()
        own = _add_entry(pool, "tts", size=4 * GiB, state=EngineState.ACTIVE)
        own.active_uses = 1
        _stub_phase1_unload(pool)
        # committed 38+8+4 = 50 > 46 → over budget, but 'embed' is still LOADING.

        async def _op():
            async with pool.reserve_inference("/fake/tts", 1 * GiB):
                return "ran"

        task = asyncio.create_task(_op())
        await asyncio.sleep(0.1)
        assert not task.done(), "op must WAIT for the loading model, not best-effort"
        # Load completes → 'embed' becomes evictable.
        loading.state = EngineState.ACTIVE
        loading.engine = MagicMock()
        loading.engine.has_active_requests = MagicMock(return_value=False)
        loading.ready_event.set()
        await asyncio.sleep(0.1)
        # The op now evicts the (loaded) 'embed' to make room — not best-effort.
        assert pool._entries["embed"].state == EngineState.UNLOADING
        _complete_unload(pool, "embed")
        result = await asyncio.wait_for(task, timeout=3.0)
        assert result == "ran"
        assert pool._inflight_reservations == 0

    def test_pick_victim_excludes_self_even_when_self_is_lru(self):
        # The structural root of every best-effort OOM: the finder returns only
        # the single LRU non-pinned model; when that IS the op's own engine, a
        # drop-after-pick returns None and misses other evictable models → the op
        # best-efforts → Metal OOM. The picker must EXCLUDE self during selection.
        pool = _make_pool()
        _add_entry(pool, "pinned-big", size=38 * GiB, is_pinned=True)
        # 'tts' (self) is the LRU (older last_access); 'embed' is newer.
        _add_entry(pool, "tts", size=4 * GiB, state=EngineState.ACTIVE, last_access=1.0)
        _add_entry(pool, "embed", size=8 * GiB, state=EngineState.ACTIVE, last_access=99.0)
        assert pool._pick_inference_victim("tts") == "embed"
        # The underlying finder honors the exclusion directly...
        assert pool._find_drain_or_evict_candidate(exclude_model_id="tts") == "embed"
        # ...and without exclusion it would (wrongly, for inference) pick self.
        assert pool._find_drain_or_evict_candidate() == "tts"
