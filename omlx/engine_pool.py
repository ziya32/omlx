# SPDX-License-Identifier: Apache-2.0
"""
Engine pool for oMLX multi-model serving.

This module manages multiple model engines with drain-based eviction
and implicit request queuing. It supports:

- Drain-based model switching (in-flight requests finish before unload)
- Implicit request queuing via asyncio.Event (no explicit queue structure)
- Pre-load memory checking to ensure models fit before loading
- LRU eviction of least recently used models
- Model pinning to keep specific models always loaded
- BatchedEngine for all LLM models (continuous batching)

State machine per model:
    UNLOADED ──load──► LOADING ──success──► ACTIVE ──evict──► DRAINING ──empty──► UNLOADING ──cleanup──► UNLOADED
                          │                                      │                    ▲
                          ▼                                      ▼ (timeout)          │
                       UNLOADED (fail)                        UNLOADING ──────────────┘
"""

from __future__ import annotations

import asyncio
import gc
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from .model_settings import ModelSettingsManager

import mlx.core as mx

from .engine import BaseEngine, BatchedEngine
from .engine.embedding import EmbeddingEngine
from .engine.reranker import RerankerEngine
from .engine.stt import STTEngine
from .engine.sts import STSEngine
from .engine.tts import TTSEngine
from .engine.vlm import VLMBatchedEngine
from .exceptions import (
    EngineEvictedError,
    EnginePoolError,
    InsufficientMemoryError,
    ModelLoadingError,
    ModelNotFoundError,
    ModelTooLargeError,
)
from .model_discovery import discover_models, format_size
from .engine_core import get_mlx_executor
from .mx_buffer_lock import locked_free_and_clear, locked_sync_and_clear_cache
from .utils.proc_memory import get_phys_footprint
from .scheduler import SchedulerConfig

logger = logging.getLogger(__name__)

# Cooldown after a load failure before retrying (seconds)
LOAD_COOLDOWN = 30

# A load aborted mid-flight by the memory enforcer is transient (pressure),
# not a real load failure: retry in-loop instead of surfacing a 503. Bounded
# so a genuinely un-loadable model still errors out. Each retry re-enters the
# memory-wait path (_prepare_memory_for); the small backoff prevents a hot
# loop on the rare path where the pre-load check would 'proceed cautiously'
# straight back into pressure.
MAX_ENFORCER_ABORT_RETRIES = 8
ENFORCER_ABORT_RETRY_BACKOFF = 2.0

# Grace window for eviction-candidate selection. Engines accessed within
# this many seconds count as "active" even when their current
# active_uses is 0, so the FastAPI dispatch + acquire_engine entry
# window between back-to-back requests doesn't expose the model to an
# enforcer tick that races against the brief active_uses=0 dip.
# Tuned to be comfortably larger than the inter-request gap in
# concurrent stress workloads (~ms) but shorter than typical
# inter-request gaps in real traffic (~seconds-to-minutes) so an
# actually-idle model still gets evicted promptly under sustained
# pressure.
_ACTIVE_GRACE_SEC: float = 3.0

# Short grace specifically for the exclusive-defer (NOT the 3s enforcer grace above).
# Covers just the get_engine→acquire_engine dip where a routing request to the exclusive
# model reads as active_uses=0; long enough for that hand-off under load, short enough
# that a non-pinned load isn't held off for seconds after the exclusive model went idle
# (the 3s grace did that — flooding the log and tripping get_engine's excessive-loop guard).
_EXCLUSIVE_DEFER_GRACE_SEC: float = 0.5
# Rate-limit per loading model for the "Deferring load …" INFO lines: a non-pinned load
# can re-check the exclusive gate many times while a chat runs; one line per check floods.
_DEFER_LOG_INTERVAL_SEC: float = 5.0

# Safety margin subtracted from the Metal wall when budgeting in-flight
# non-streaming inference transients (reserve_inference). Covers allocator
# slop/fragmentation, the reservation-vs-max(active,phys) drift, and small
# un-reserved temporaries. Overridable for calibration. 2 GiB on a 64 GB box
# leaves the budget ~2 GiB below get_effective_metal_cap_bytes().
# Headroom below the hard Metal wall reserved for inference working sets, so a
# non-streaming op's reservation FORCES eviction of a coexisting non-pinned model
# rather than decoding on top of it (the 2-TTS OOM, 2026-06-02). RAM-PROPORTIONAL
# (a fraction of the wall, clamped): ~4 GiB on a 64 GB Mac's ~52 GiB wall (the
# validated value), but it does not starve a 16-32 GB box (where a fixed 4 GiB
# would be 17-33% of the wall) nor bloat a 128-256 GB one. It covers the
# estimate-vs-actual (phys_footprint / fragmentation) drift + allocator slop; the
# decode transient itself is reserved separately, so the margin need not cover it.
_INFERENCE_MARGIN_FRACTION: float = 0.08        # ~4 GiB at a 64 GB Mac's ~52 GiB wall
_INFERENCE_MARGIN_MIN_BYTES: int = 2 * 1024**3  # floor: enough for drift on small boxes
_INFERENCE_MARGIN_MAX_BYTES: int = 8 * 1024**3  # cap: a transient does not grow with RAM


class EngineState(Enum):
    """Lifecycle state for a model engine."""
    UNLOADED = "unloaded"     # No engine instance
    LOADING = "loading"       # Model being loaded
    ACTIVE = "active"         # Normal operation, accepts requests
    DRAINING = "draining"     # Finishing in-flight, new requests wait
    UNLOADING = "unloading"   # Engine stopped, Metal cleanup in progress


@dataclass
class EngineEntry:
    """Per-model state in the engine pool."""

    model_id: str  # Directory name (e.g., "llama-3b")
    model_path: str  # Full path to model directory
    model_type: Literal["llm", "vlm", "embedding", "reranker", "audio_stt", "audio_tts", "audio_sts"]  # Model type
    engine_type: Literal["batched", "embedding", "reranker", "vlm", "audio_stt", "audio_tts", "audio_sts"]  # Engine type to use
    estimated_size: int  # Pre-calculated from safetensors (bytes)
    config_model_type: str = ""  # Raw model_type from config.json (e.g., "deepseekocr_2")
    thinking_default: bool | None = None  # True if model thinks by default, False if not, None if unknown
    preserve_thinking_default: bool | None = None  # True when template supports preserve_thinking (Qwen 3.6+)
    engine: BaseEngine | EmbeddingEngine | RerankerEngine | STTEngine | STSEngine | TTSEngine | None = None  # Loaded engine instance
    last_access: float = 0.0  # Timestamp for LRU (0 if never loaded)
    last_release: float = 0.0  # When a lease was last released (request END;
    #                            last_access is bumped at dispatch = request
    #                            START, so the exclusive-defer dip grace must
    #                            anchor on max(last_access, last_release))
    is_pinned: bool = False  # Never evict if True
    abort_loading: bool = False  # Enforcer→loader signal: abort this in-flight
    #                              load (memory pressure). Reset at each load
    #                              start so a prior abort can't poison reloads.
    _vision_limits_cache: dict | None = None  # Cached compute_vision_limits result
    # Diagnostic fields: actual_size is observed phys_footprint delta after load
    # settles (None until first load completes); loading_started_at lets the
    # admin UI compute an ETA against the load_seconds_per_gb EMA.
    actual_size: int | None = None
    loading_started_at: float | None = None

    # State machine fields
    state: EngineState = EngineState.UNLOADED
    ready_event: asyncio.Event | None = None     # Set when LOADING → ACTIVE/UNLOADED
    drain_complete: asyncio.Event | None = None   # Set when DRAINING → UNLOADED
    unload_complete: asyncio.Event | None = None  # Set when UNLOADING → UNLOADED
    drain_started: float = 0.0                    # For timeout tracking
    load_error: Exception | None = None           # Propagated to waiters
    load_started: float = 0.0                     # For diagnostics
    load_failed_at: float = 0.0                   # Cooldown after load failure
    active_uses: int = 0                          # Number of request handlers currently using this engine
    exclusive: bool = False                        # Evict all non-pinned models on request (requires is_pinned)
    exclusive_max_hold: int = 0                    # Max seconds of continuous exclusive hold (0 = unlimited)
    exclusive_idle: asyncio.Event | None = None    # Signaled when exclusive model active_uses → 0
    _exclusive_hold_start: float = 0.0             # When active_uses went 0 → 1 (for max_hold tracking)
    _last_defer_log_time: float = 0.0              # Rate-limit "Deferring load …" logs (1 / _DEFER_LOG_INTERVAL_SEC)

    @property
    def is_loading(self) -> bool:
        """Backward-compatible property: True when state is LOADING."""
        return self.state == EngineState.LOADING


class EnginePool:
    """
    Manages multiple model engines with drain-based memory management.

    Features:
    - Pre-load memory checking (evict before load, not after)
    - Drain-based eviction: in-flight requests finish before unload
    - Implicit request queuing via asyncio.Event
    - Model pinning to prevent eviction
    - Automatic engine type selection based on model type

    Diagnostics:
    - get_debug_status(): lock-free snapshot for /debug/pool endpoint
    - Load timing: elapsed time logged for every model load

    Deferred:
    - Per-request trace_id propagation through get_engine/drain/load
    """

    def __init__(
        self,
        scheduler_config: SchedulerConfig | None = None,
        drain_timeout: float = 120.0,
        max_wait_timeout: float = 300.0,
    ):
        """
        Initialize the engine pool.

        Args:
            scheduler_config: Configuration for BatchedEngine schedulers
            drain_timeout: Seconds before force-aborting a draining model
            max_wait_timeout: Seconds before timing out a get_engine() wait

        Note:
            Pre-load admission consults the process memory enforcer's
            dynamic tier ceiling (``get_final_ceiling()``) via
            ``_current_ceiling()``. The enforcer is set by
            ``server.init_server()``; until then the pool admits
            unconditionally (ceiling == 0).
        """
        self._entries: dict[str, EngineEntry] = {}
        self._lock = asyncio.Lock()
        self._scheduler_config = scheduler_config or SchedulerConfig()
        self._process_memory_enforcer: object | None = None  # Set by server
        self._get_final_ceiling: object | None = None  # Set by server (callback)
        self._settings_manager: object | None = None  # Set by server
        self._suppress_ttl: bool = False  # Suppress TTL during benchmarks
        # Exponentially-weighted moving average of observed load speed in
        # seconds-per-GB. Used by the admin UI to project ETA for in-progress
        # loads based on EngineEntry.loading_started_at + estimated_size.
        self._load_seconds_per_gb_ema: float | None = None
        self._load_time_observations: int = 0
        self._drain_timeout = drain_timeout
        self._max_wait_timeout = max_wait_timeout
        # Incremented on ANY timeout firing. Tests assert this stays at 0.
        self._timeout_counter: int = 0
        # Track deferred engine cleanup tasks so shutdown can wait for them
        self._cleanup_tasks: list[asyncio.Task] = []
        # In-flight non-streaming inference working-set reservation (bytes).
        # Joined into the SAME committed total load admission reads
        # (_committed_plus_reservations), so model loads and non-streaming
        # inference size themselves against each other symmetrically under
        # this one lock. Incremented at admission by reserve_inference(),
        # subtracted in its finally; read by the scheduler's predictive
        # generation guard via a synchronous callback (§3e). Mutated only on
        # the event loop (every mutation site is a coroutine, so plain int
        # arithmetic is atomic between awaits); the scheduler read is a
        # lock-free atomic int load (GIL-safe). The finally-decrement is
        # deliberately LOCK-FREE: awaiting the pool lock inside a finally can
        # be interrupted by a second CancelledError (or the lock-acquire
        # timeout), permanently leaking the reservation.
        self._inflight_reservations: int = 0
        # Edge-triggered "a reservation was released" pulse (§P2). Over-budget
        # waiters whose overshoot is ATTRIBUTABLE to in-flight transients
        # (weights alone fit; only the reservations push past budget) grab the
        # CURRENT event under the lock and await it outside; every release
        # swaps in a fresh event and sets the old one, so each pulse wakes
        # exactly the waiters that observed it (no busy-loop, no lost wakeup:
        # a release between the check and the await set the very object the
        # waiter holds). Before this, those waiters either 507'd a load or
        # decoded best-effort over the Metal wall for pressure that clears in
        # seconds. Loop-only state, like the counter itself.
        self._reservation_released: asyncio.Event = asyncio.Event()
        # §P6: contended (over-budget) reservations currently RUNNING, and the
        # corrector task that keeps reclaiming IDLE victims while any exist.
        # Admission is otherwise one-shot: round-5 of the 2026-06-09 stress
        # campaign crashed with a 6.14GB embedding engine loaded-and-IDLE
        # through the whole fatal window — it was momentarily busy at the
        # TTS op's admit instant (17ms after its last embed), so the one-shot
        # victim check found "nothing reclaimable", and nothing ever looked
        # again. The corrector is that missing reflex.
        self._contended_admits: int = 0
        self._budget_corrector_task: asyncio.Task | None = None
        # Requests currently PARKED at get_engine's exclusive-contention gate
        # with their lease released (see the park in get_engine). They hold
        # no active_uses while parked, so without this counter they would be
        # invisible to total_active_uses / the /v1/idle busy signal.
        # Incremented with the park-release, decremented with the wake
        # re-acquire (both paired via the wait's try/finally); event-loop
        # only, so plain int arithmetic suffices.
        self._contention_parked: int = 0

    # -------------------------------------------------------------------------
    # State transition helpers
    # -------------------------------------------------------------------------

    def _set_state(
        self, entry: EngineEntry, new_state: EngineState, reason: str
    ) -> None:
        """Transition entry to new_state with structured logging."""
        old_state = entry.state
        entry.state = new_state
        logger.info(
            f"model={entry.model_id} "
            f"state={old_state.value}\u2192{new_state.value} "
            f"reason={reason}"
        )

    def _record_timeout(
        self, kind: str, model_id: str, elapsed: float
    ) -> None:
        """Record a timeout event. Called from any timeout handler."""
        self._timeout_counter += 1
        logger.error(
            f"LIVELOCK_SUSPECT: {kind} timeout for {model_id} "
            f"after {elapsed:.1f}s "
            f"(total_timeouts={self._timeout_counter})"
        )

    @asynccontextmanager
    async def _tracked_lock(self, caller: str):
        """Acquire self._lock with wait/hold timing and 60s acquire timeout."""
        t0 = time.monotonic()
        try:
            await asyncio.wait_for(self._lock.acquire(), timeout=60)
        except asyncio.TimeoutError:
            self._record_timeout("lock_acquire", caller, 60.0)
            raise EnginePoolError(
                f"Lock acquire timed out after 60s (caller={caller})"
            )
        acquired = time.monotonic()
        wait_ms = (acquired - t0) * 1000
        # logger.info(f"engine_pool._tracked_lock: lock acquired at {acquired:.0f}s, wait={wait_ms:.0f}ms caller={caller}")

        if wait_ms > 1000:
            logger.warning(f"lock wait={wait_ms:.0f}ms caller={caller}")

        try:
            yield
        finally:
            held_ms = (time.monotonic() - acquired) * 1000
            self._lock.release()
            # logger.info(f"engine_pool._tracked_lock: lock released at {time.monotonic():.0f}s, held={held_ms:.0f}ms caller={caller}")
            if held_ms > 500:
                logger.warning(f"lock held={held_ms:.0f}ms caller={caller}")

    # -------------------------------------------------------------------------
    # Memory accounting
    # -------------------------------------------------------------------------

    def _committed_memory(self) -> int:
        """Memory committed by ACTIVE + DRAINING + LOADING + UNLOADING models."""
        return sum(
            e.estimated_size
            for e in self._entries.values()
            if e.state in (
                EngineState.ACTIVE, EngineState.DRAINING,
                EngineState.LOADING, EngineState.UNLOADING,
            )
        )

    # -- In-flight inference reservation accounting (§3a) ------------------
    # One number, one lock: resident model weights (_committed_memory) plus
    # the working-set transient of in-flight non-streaming inference
    # (_inflight_reservations). Load admission and inference admission both
    # size against this total, so each sees the other's commitment.

    def _committed_plus_reservations(self) -> int:
        """Resident weights (existing) + in-flight inference transients."""
        return self._committed_memory() + self._inflight_reservations

    def _inference_margin_bytes(self, cap: int) -> int:
        """RAM-proportional headroom below the Metal wall (see module constants).

        ``clamp(cap * fraction, min, max)`` — scales with the wall so it is the
        validated ~4 GiB on a 64 GB Mac, ~2 GiB (floor) on a 16-32 GB box, and
        ~8 GiB (cap) on a 128-256 GB one.
        """
        return max(
            _INFERENCE_MARGIN_MIN_BYTES,
            min(_INFERENCE_MARGIN_MAX_BYTES, int(cap * _INFERENCE_MARGIN_FRACTION)),
        )

    def _wall_budget(self) -> int:
        """Inference budget = Metal wall − margin. 0 ⇒ no real cap ⇒ no-op.

        References the per-process Metal wall
        (``get_effective_metal_cap_bytes()``) — the value above which Metal
        rejects/panics — NOT the adaptive ceiling (which adapts up toward
        the wall and would re-introduce coexistence thrash). The margin is
        RAM-proportional (``_inference_margin_bytes``). Returns 0 when the cap
        accessor returns 0 (non-Apple / older macOS), making
        ``reserve_inference`` a strict no-op.
        """
        from .process_memory_enforcer import get_effective_metal_cap_bytes
        cap = get_effective_metal_cap_bytes()
        if cap <= 0:
            return 0
        return max(0, cap - self._inference_margin_bytes(cap))

    def _note_contended_admit(self) -> None:
        """Record an over-budget admit and arm the §P6 budget corrector."""
        self._contended_admits += 1
        task = self._budget_corrector_task
        if task is None or task.done():
            self._budget_corrector_task = asyncio.create_task(
                self._budget_corrector()
            )

    async def _budget_corrector(self) -> None:
        """§P6 — keep reclaiming IDLE victims while contended admits run.

        Admission's victim check is one-shot: an over-budget op that found
        "nothing reclaimable" at its admit instant (a victim busy for one
        more request, a §P3-disallowed drain) runs its whole decode while a
        victim that went idle seconds later stays resident — round 5 of the
        2026-06-09 stress campaign Metal-panicked with an idle 6.14GB
        embedding engine loaded through the entire fatal window. While any
        contended reservation is outstanding and the pool is over budget,
        this loop unloads idle non-pinned victims (1s cadence; idle-only —
        busy victims stay a drain decision for admission, so no §P3
        entanglement) and exits the moment balance returns.
        """
        try:
            while True:
                await asyncio.sleep(1.0)
                async with self._tracked_lock("budget_corrector"):
                    if self._contended_admits <= 0:
                        return
                    if self._reservable_free() >= 0:
                        return
                    victim = self._find_drain_or_evict_candidate()
                    if victim is None or self._victim_busy(victim):
                        continue
                    logger.info(
                        "budget corrector: evicting idle '%s' while %d "
                        "contended admit(s) run over budget", victim,
                        self._contended_admits,
                    )
                    await self._unload_engine(
                        victim, reason="evict_contended_corrector"
                    )
        except asyncio.CancelledError:
            pass  # pool shutdown

    def _small_transient_threshold(self) -> int:
        """Transients at or below this run UNACCOUNTED (margin-absorbed).

        A quarter of the inference margin (≈1 GiB on a 64 GB Mac): the margin
        exists to absorb estimate drift + small unaccounted transients, so the
        single-forward encoders (embedding ≈0.6 GiB, ASR/STS ≈0.3 GiB) bypass
        the reservation lane entirely — exactly the economics of the system
        that passed the stress suite — while TTS-class transients (≥2.4 GiB)
        still reserve, wait, and evict through the full §P2/§P3 lane. 0 when
        there is no Metal cap (everything already no-ops).
        """
        from .process_memory_enforcer import get_effective_metal_cap_bytes
        cap = get_effective_metal_cap_bytes()
        if cap <= 0:
            return 0
        return self._inference_margin_bytes(cap) // 4

    def _overshoot_allowance(self) -> int:
        """Sanctioned TOTAL of outstanding transients under contention (§P3+).

        ``_wall_budget`` already excludes the inference margin, so a CONTENDED
        admit (nothing reclaimable, nothing freeing) may dip INTO the margin —
        never past the raw cap. Bounding the SUM of outstanding reservations
        by the margin keeps small forwards (≈0.3–0.6 GiB each) interleaving
        under over-commit — the workload this box ran un-reserved for months —
        while a SECOND TTS-sized transient (≈4 GiB) parks on the release pulse
        (two of those stacked is the 2-TTS Metal panic, re-confirmed by the
        2026-06-09 re-validation crash).
        """
        from .process_memory_enforcer import get_effective_metal_cap_bytes
        cap = get_effective_metal_cap_bytes()
        if cap <= 0:
            return 0
        return self._inference_margin_bytes(cap)

    def _scheduler_inflight_bytes(self) -> int:
        """Live LLM/VLM scheduler generation transient (§3e); 0 if unwired.

        Read through the existing ``_process_memory_enforcer`` reference,
        which already resolves every scheduler. A TTS decode admitting while
        a batched prefill is live sees that transient and evicts/waits
        instead of stacking over the wall.
        """
        enf = self._process_memory_enforcer
        if enf is None:
            return 0
        getter = getattr(enf, "get_scheduler_inflight_bytes", None)
        if getter is None:
            return 0
        try:
            return int(getter())
        except Exception:  # noqa: BLE001
            return 0

    def _reservable_free(self) -> int:
        """Room for a NEW in-flight transient under the wall budget."""
        return (
            self._wall_budget()
            - self._committed_plus_reservations()
            - self._scheduler_inflight_bytes()
        )

    def _current_ceiling(self) -> int:
        """Resolve the current memory ceiling (dynamic tier ceiling).

        Prefers the ``_get_final_ceiling`` callback wired up by
        ``server.init_server()`` (``enforcer.get_final_ceiling`` =
        min(static, dynamic, metal_cap)); falls back to the enforcer
        reference directly when only that is set (e.g. in tests). Returns
        0 when neither is wired up or the memory guard is disabled —
        callers treat 0 as "no limit".
        """
        cb = self._get_final_ceiling
        if cb is not None:
            try:
                return int(cb())
            except Exception:  # noqa: BLE001
                return 0
        enforcer = self._process_memory_enforcer
        if enforcer is not None:
            try:
                return int(enforcer.get_final_ceiling())
            except Exception:  # noqa: BLE001
                return 0
        return 0

    @property
    def max_model_memory(self) -> int | None:
        """Memory ceiling for admission in bytes, or None if disabled.

        Derived from the enforcer's dynamic tier ceiling (``_current_ceiling``)
        now that the static ``max_model_memory`` setting is gone.
        """
        return self._current_ceiling() or None

    @property
    def current_model_memory(self) -> int:
        """Current memory used by loaded models in bytes.

        Backward-compatible property — delegates to _committed_memory().
        """
        return self._committed_memory()

    @property
    def model_count(self) -> int:
        """Total number of discovered models."""
        return len(self._entries)

    @property
    def loaded_model_count(self) -> int:
        """Number of currently loaded models."""
        return sum(1 for e in self._entries.values() if e.engine is not None)

    @property
    def total_active_uses(self) -> int:
        """Total in-flight engine leases across all models.

        The sum of every entry's ``active_uses`` — the number of request
        handlers that have acquired, **or are blocked waiting to acquire**, an
        engine. ``acquire_engine`` is the single chokepoint every request path
        (``use_engine``, audio ``_use_engine``, the embeddings endpoint) calls
        BEFORE its ``get_engine`` await, so a request wedged waiting for a
        loading / draining model still counts here even though it is not yet
        computing. ONE exception: a request parked at the exclusive-contention
        gate has its lease RELEASED while it waits (so the idle engine stays
        cleanly evictable) — those are tracked by ``contention_parked``
        instead; busy-ness consumers must add both. Pinning is a separate flag
        (``is_pinned``), so an idle pinned model contributes 0 — no baseline to
        subtract. Used by the server-wide idle tracker (``/v1/idle``) so a
        contended-but-not-computing server never reads as idle.
        """
        return sum(e.active_uses for e in self._entries.values())

    @property
    def contention_parked(self) -> int:
        """Requests parked (lease released) at the exclusive-contention gate.

        Complements ``total_active_uses``: a parked waiter dropped its lease so
        the eviction drain stays unblocked, which also removes it from the
        lease count — this counter keeps it visible to ``/v1/idle`` busy-ness.
        """
        return self._contention_parked

    @property
    def loading_model_count(self) -> int:
        """Number of entries with a model load currently in flight.

        A load may be driven by a lease-less caller (TTS pre-validation
        historically, the public/admin load endpoints, pinned preload), so the
        lease count alone can read 0 while tens of GB are mid-load — exactly
        when admitting gateway idle work (a second model load) is most
        dangerous. ``/v1/idle`` counts these as busy.
        """
        return sum(
            1 for e in self._entries.values()
            if e.state == EngineState.LOADING
        )

    def llm_request_counts(self) -> tuple[int, int]:
        """``(active, waiting)`` LLM scheduler requests across loaded engines.

        Walks each batched engine's private chain
        (``entry.engine._engine.engine``) to its ``_output_collectors`` /
        ``scheduler.waiting``. The one fragile private-attr walk lives HERE —
        next to ``_entries`` — so a core rename breaks exactly one place;
        both ``/v1/idle``'s busy signal and ``/api/status`` read it.
        Non-batched engines (embedding/audio) have no ``_engine`` chain and
        contribute 0.
        """
        active = 0
        waiting = 0
        for entry in self._entries.values():
            engine = entry.engine
            if engine is None:
                continue
            async_core = getattr(engine, "_engine", None)
            if async_core is None:
                continue
            core = getattr(async_core, "engine", None)
            if core is None:
                continue
            active += len(getattr(core, "_output_collectors", {}))
            sched = getattr(core, "scheduler", None)
            if sched is not None:
                waiting += len(getattr(sched, "waiting", []))
        return active, waiting

    # -------------------------------------------------------------------------
    # Vision limits (memory-aware, pre-load)
    # -------------------------------------------------------------------------

    # Peak GPU activation bytes per input pixel for Qwen3-VL/3.5 (32-layer ViT).
    # Empirically: 50M pixels OOM'd with 21.5 GB headroom (>430 B/px), while
    # ~5M pixels succeeded with 7 GB headroom (<1400 B/px).  700 B/px provides
    # a safe margin across different memory configurations.
    _VISION_BYTES_PER_PIXEL = 700
    _VISION_SAFETY_FACTOR = 0.7  # leave 30% headroom for KV growth / allocator
    _VISION_MAX_CONTEXT_FRACTION = 0.8  # max share of context window for vision tokens

    def compute_vision_limits(self, vlm_entry: EngineEntry) -> dict[str, Any]:
        """Compute vision processing limits for a VLM model.

        Uses the model's ``estimated_size`` from safetensors metadata and
        ``max_position_embeddings`` from config.json — does NOT require the
        model to be loaded.  Only pinned models count toward committed
        memory because unpinned models are evicted when the VLM loads.
        """
        enforcer = self._process_memory_enforcer
        max_bytes = enforcer.get_final_ceiling() if enforcer else 0
        if not max_bytes:
            return {}

        # Project memory when VLM is loaded
        pinned_other = sum(
            e.estimated_size for e in self._entries.values()
            if e.is_pinned and e is not vlm_entry
        )
        vlm_loaded_size = int(vlm_entry.estimated_size * 1.25)  # 25% KV headroom
        headroom = max(0, max_bytes - pinned_other - vlm_loaded_size)

        chunk_budget = int(
            headroom * self._VISION_SAFETY_FACTOR / self._VISION_BYTES_PER_PIXEL
        )

        ctx = self._read_context_window(vlm_entry)
        max_vision_tokens = int(ctx * self._VISION_MAX_CONTEXT_FRACTION) if ctx else 0

        return {
            "chunk_budget_pixels": chunk_budget,
            "max_context_fraction": self._VISION_MAX_CONTEXT_FRACTION,
            "max_vision_tokens": max_vision_tokens,
            "memory_headroom_bytes": headroom,
        }

    @staticmethod
    def _read_context_window(entry: EngineEntry) -> int:
        """Read max_position_embeddings from config.json without loading."""
        import json as _json
        from pathlib import Path

        cfg_path = Path(entry.model_path) / "config.json"
        try:
            with open(cfg_path) as f:
                cfg = _json.load(f)
            # VLMs: check text_config first, then root
            tc = cfg.get("text_config", {})
            for key in ("max_position_embeddings", "max_seq_len"):
                val = tc.get(key) if isinstance(tc, dict) else None
                if isinstance(val, int) and val > 0:
                    return val
            for key in ("max_position_embeddings", "max_seq_len"):
                val = cfg.get(key)
                if isinstance(val, int) and val > 0:
                    return val
        except Exception:
            pass
        return 0

    # -------------------------------------------------------------------------
    # Model discovery and settings
    # -------------------------------------------------------------------------

    def discover_models(
        self, model_dirs: str | list[str], pinned_models: list[str] | None = None
    ) -> None:
        """
        Discover models in the specified directory or directories.

        Args:
            model_dirs: Path or list of paths to directories containing model subdirectories
            pinned_models: List of model IDs to pin (never evict)
        """
        from pathlib import Path

        from .model_discovery import discover_models_from_dirs

        if isinstance(model_dirs, str):
            dirs = [Path(model_dirs)]
        else:
            dirs = [Path(d) for d in model_dirs]

        if len(dirs) == 1:
            discovered = discover_models(dirs[0])
        else:
            discovered = discover_models_from_dirs(dirs)

        pinned_set = set(pinned_models or [])

        for model_id, info in discovered.items():
            existing = self._entries.get(model_id)
            if existing is not None and (existing.engine is not None or existing.state != EngineState.UNLOADED):
                # Loaded or in-transition model: preserve runtime state, only update pinned flag
                existing.is_pinned = model_id in pinned_set
            else:
                # New or unloaded model: create fresh entry
                self._entries[model_id] = EngineEntry(
                    model_id=model_id,
                    model_path=info.model_path,
                    model_type=info.model_type,
                    engine_type=info.engine_type,
                    estimated_size=info.estimated_size,
                    config_model_type=getattr(info, "config_model_type", ""),
                    thinking_default=getattr(info, "thinking_default", None),
                    preserve_thinking_default=getattr(info, "preserve_thinking_default", None),
                    is_pinned=model_id in pinned_set,
                    state=EngineState.UNLOADED,
                )

            if model_id in pinned_set:
                logger.info(f"Pinned model: {model_id}")

        # Remove entries no longer discovered and not loaded
        discovered_ids = set(discovered.keys())
        stale = [
            mid
            for mid in self._entries
            if mid not in discovered_ids
            and self._entries[mid].engine is None
            and self._entries[mid].state == EngineState.UNLOADED
        ]
        for mid in stale:
            del self._entries[mid]

        # Warn about pinned models not found
        found_models = set(self._entries.keys())
        for model_id in pinned_set:
            if model_id not in found_models:
                logger.warning(f"Pinned model not found: {model_id}")

        _ceil = self._current_ceiling()
        mem_display = "disabled" if _ceil <= 0 else format_size(_ceil)
        logger.info(
            f"Discovered {len(self._entries)} models, "
            f"max memory: {mem_display}"
        )

    _MODEL_TYPE_TO_ENGINE: dict[str, str] = {
        "llm": "batched",
        "vlm": "vlm",
        "embedding": "embedding",
        "reranker": "reranker",
        "audio_stt": "audio_stt",
        "audio_tts": "audio_tts",
        "audio_sts": "audio_sts",
    }

    def apply_settings_overrides(
        self, settings_manager: "ModelSettingsManager"
    ) -> None:
        """Apply model_type_override and exclusive from persisted settings."""
        for model_id, entry in self._entries.items():
            settings = settings_manager.get_settings(model_id)
            if settings.model_type_override:
                entry.model_type = settings.model_type_override
                entry.engine_type = self._MODEL_TYPE_TO_ENGINE.get(
                    settings.model_type_override, "batched"
                )
                logger.info(
                    f"Applied model_type override for {model_id}: "
                    f"type={entry.model_type}, engine={entry.engine_type}"
                )
            entry.exclusive = settings.exclusive
            entry.exclusive_max_hold = settings.exclusive_max_hold
            if settings.exclusive:
                logger.info(
                    f"Applied exclusive setting for {model_id}: "
                    f"exclusive=True, max_hold={entry.exclusive_max_hold}s"
                )

    def get_model_ids(self) -> list[str]:
        """Get list of all discovered model IDs."""
        return list(self._entries.keys())

    def get_loaded_model_ids(self) -> list[str]:
        """Get list of currently loaded model IDs."""
        return [mid for mid, e in self._entries.items() if e.engine is not None]

    def get_entry(self, model_id: str) -> EngineEntry | None:
        """Get entry for a specific model, or None if not found."""
        return self._entries.get(model_id)

    def set_pinned(self, model_id: str, pinned: bool) -> bool:
        """
        Set the pinned status for a model.

        Args:
            model_id: The model ID to update
            pinned: Whether to pin (True) or unpin (False) the model

        Returns:
            True if successful, False if model not found.
        """
        entry = self._entries.get(model_id)
        if entry is None:
            return False
        entry.is_pinned = pinned
        return True

    def set_exclusive(
        self, model_id: str, exclusive: bool, max_hold: int = 0
    ) -> bool:
        """Set the exclusive status for a model.

        Args:
            model_id: The model ID to update
            exclusive: Whether to enable (True) or disable (False) exclusive mode
            max_hold: Max seconds of continuous exclusive hold (0 = unlimited)

        Returns:
            True if successful, False if model not found.
        """
        entry = self._entries.get(model_id)
        if entry is None:
            return False
        entry.exclusive = exclusive
        entry.exclusive_max_hold = max_hold
        if not exclusive:
            entry.exclusive_idle = None
            entry._exclusive_hold_start = 0.0
        return True

    def _case_insensitive_entry_match(self, name: str) -> str | None:
        """Find a model entry matching *name* case-insensitively.

        Returns the actual model_id if found, None otherwise.
        """
        lower = name.lower()
        for mid in self._entries:
            if mid.lower() == lower:
                return mid
        return None

    def resolve_model_id(self, model_id_or_alias: str, settings_manager) -> str:
        """Resolve a model alias to its actual model_id (directory name).

        Tries exact match in _entries first, then case-insensitive match,
        then scans model settings for alias match. If those fail and input
        contains a provider prefix (e.g. "omlx/my-model"), strips the prefix
        and retries. Returns the original string if no match found.
        """
        if model_id_or_alias in self._entries:
            return model_id_or_alias

        # Case-insensitive fallback
        ci_match = self._case_insensitive_entry_match(model_id_or_alias)
        if ci_match is not None:
            return ci_match

        all_settings = None
        if settings_manager is not None:
            all_settings = settings_manager.get_all_settings()
            for mid, ms in all_settings.items():
                if ms.aliases and model_id_or_alias in ms.aliases:
                    return mid

        # Strip provider prefix (e.g. "omlx/qwen3.5-35b" -> "qwen3.5-35b")
        if "/" in model_id_or_alias:
            stripped = model_id_or_alias.split("/", 1)[1]
            if stripped in self._entries:
                return stripped
            ci_match = self._case_insensitive_entry_match(stripped)
            if ci_match is not None:
                return ci_match
            if all_settings is not None:
                for mid, ms in all_settings.items():
                    if ms.aliases and stripped in ms.aliases:
                        return mid

        return model_id_or_alias

    # -------------------------------------------------------------------------
    # get_engine() — the core entry point (drain + implicit queue)
    # -------------------------------------------------------------------------

    async def get_engine(
        self, model_id: str, force_lm: bool = False,
    ) -> BaseEngine | EmbeddingEngine | RerankerEngine | STTEngine | STSEngine | TTSEngine:
        """
        Get or load engine for the specified model.

        Waits if the model is loading or draining. Never returns 507 for
        temporary memory contention — callers wait for drain to complete.

        Args:
            model_id: The model ID to get engine for
            force_lm: Force loading as LM (BatchedEngine) even for VLM models.
                Useful for text-only tasks like accuracy benchmarks.

        Returns:
            The loaded engine (BaseEngine for LLM, EmbeddingEngine for embeddings)

        Raises:
            ModelNotFoundError: If model is not discovered
            ModelTooLargeError: If model exceeds memory limit (permanent)
            ModelLoadingError: If wait times out or model in load cooldown
        """
        start_time = time.monotonic()
        iterations = 0
        enforcer_abort_retries = 0
        # When we park a non-pinned request at the exclusive-contention gate we
        # RELEASE its engine lease (so the now-idle engine can be evicted cleanly
        # for the exclusive model's headroom instead of dead-locking the drain)
        # and re-acquire it on EVERY wait exit — success, timeout, or
        # cancellation — via the wait's try/finally, so the caller's paired
        # release_engine() always stays balanced. While parked, the request is
        # counted in _contention_parked so /v1/idle still sees it as busy.
        released_for_contention = False

        while True:
            iterations += 1
            if iterations > 10:
                logger.warning(
                    f"get_engine({model_id}) excessive looping: "
                    f"iteration={iterations} "
                    f"elapsed={time.monotonic() - start_time:.1f}s"
                )

            should_load = False
            event = None
            wait_target = ""

            async with self._tracked_lock("get_engine"):
                entry = self._entries.get(model_id)
                if not entry:
                    raise ModelNotFoundError(model_id, list(self._entries.keys()))

                if entry.state == EngineState.ACTIVE and entry.engine is not None:
                    # Fast path: model loaded and ready
                    entry.last_access = time.time()

                    # Exclusive pinned models: clear non-pinned models before
                    # returning the engine so inference has maximum headroom.
                    if entry.is_pinned and entry.exclusive:
                        logger.debug(
                            f"get_engine({model_id}) exclusive fast path: "
                            f"active_uses={entry.active_uses}, "
                            f"exclusive_idle={'set' if entry.exclusive_idle is not None and entry.exclusive_idle.is_set() else 'unset' if entry.exclusive_idle is not None else 'None'}"
                        )
                        # Step 10: check if exclusive hold has expired
                        if (
                            entry.exclusive_max_hold > 0
                            and entry._exclusive_hold_start > 0
                        ):
                            hold = time.time() - entry._exclusive_hold_start
                            if hold > entry.exclusive_max_hold:
                                logger.info(
                                    f"Exclusive hold expired for '{model_id}' "
                                    f"({hold:.0f}s > {entry.exclusive_max_hold}s)"
                                )
                                self._refresh_vision_limits(entry)
                                return entry.engine

                        wait_event = await self._clear_for_exclusive(entry)
                        if wait_event is not None:
                            # Must wait for drain/unload — fall through to the
                            # wait-outside-lock section, then re-enter loop.
                            event = wait_event
                            wait_target = "exclusive_headroom"
                            # Don't set should_load — model is already loaded.
                            # Break out of the lock block to enter the wait path.
                        else:
                            # All clear — recalculate vision limits if VLM
                            logger.debug(
                                f"get_engine({model_id}) exclusive all clear, "
                                f"returning engine (active_uses={entry.active_uses})"
                            )
                            self._refresh_vision_limits(entry)
                            return entry.engine
                    else:
                        # Non-pinned fast path (Option A / Step 9b extended).
                        # Before returning an already-loaded non-pinned engine,
                        # check whether any exclusive pinned model is currently
                        # holding its lease (active_uses > 0).  If so, defer
                        # this request — same rationale as Step 9b's load-time
                        # gate in _prepare_memory_for: the MLX executor is
                        # single-threaded, so any non-pinned inference work
                        # interleaves with the exclusive model's scheduler
                        # steps on the same thread.  The load-time gate alone
                        # isn't enough because already-loaded models skip
                        # _prepare_memory_for entirely and would run freely
                        # on the shared executor during VLM inference.
                        #
                        # Safety: the waiter holds no lease on this entry, so
                        # _clear_for_exclusive on the other side is free to
                        # evict it while we wait.  When exclusive_idle fires
                        # and the loop re-enters, if our entry has been
                        # evicted the state will now be UNLOADED and we fall
                        # through to the normal load path — the request
                        # reloads from SSD and completes.
                        contention_event: asyncio.Event | None = None
                        for _mid, _e in self._entries.items():
                            if (
                                _e.exclusive
                                and _e.is_pinned
                                and _e.active_uses > 0
                                and _e.exclusive_idle is not None
                            ):
                                contention_event = _e.exclusive_idle
                                break
                        if contention_event is not None:
                            # Release the caller's lease while we PARK here. We are
                            # not using this engine — we're waiting on an UNRELATED
                            # exclusive model — so holding active_uses only blocks a
                            # memory-eviction drain of THIS idle engine: the deadlock
                            # where the drain waits for us while we wait for the
                            # exclusive model (it livelocks the drain for the full
                            # 300s get_engine timeout). Releasing restores this
                            # branch's own documented invariant (above): the entry
                            # becomes idle-evictable, so _clear_for_exclusive can
                            # unload it CLEANLY (no drain wait), and we reload from
                            # SSD on wake — gated again by _prepare_memory_for's
                            # exclusive check, so a reload only begins once the
                            # exclusive model is idle (no abort-mid-load churn).
                            # Re-acquired on wake before the next dispatch returns
                            # the engine, so the use-window stays eviction-protected.
                            # The waiter is non-pinned, so the release has no
                            # exclusive-idle side effects.
                            self.release_engine(model_id)
                            released_for_contention = True
                            self._contention_parked += 1
                            logger.debug(
                                f"get_engine({model_id}) non-pinned deferred "
                                f"(lease released): exclusive model active_uses>0"
                            )
                            event = contention_event
                            wait_target = "exclusive_contention"
                            # Don't set should_load — fall through to wait path.
                        else:
                            return entry.engine

                elif entry.state == EngineState.LOADING:
                    # Someone else is loading this model — wait for them
                    event = entry.ready_event

                elif entry.state == EngineState.DRAINING:
                    # The drain was triggered by a *different* model's
                    # _prepare_memory_for choosing this engine as its
                    # eviction victim — but we ARE that engine's user,
                    # so re-acquiring the live ref lets us batch with
                    # the existing leases instead of waiting through
                    # the full unload + reload cycle (~30s drain +
                    # ~25s reload on a 4B reranker/embedding under
                    # stress).
                    #
                    # Race seen in test_17_max_stress: VLM
                    # exclusive_idle fires; 6 non-VLM requests wake up
                    # and race for the pool lock; if a request for a
                    # DIFFERENT model (e.g. asr) wins the race, it
                    # triggers drain on the just-loaded model
                    # (e.g. Reranker) before a sibling request
                    # (rerank-2) can fast-path acquire. Without this
                    # branch rerank-2 waits the full drain + reload
                    # cycle, pushing tail latency past the 300s client
                    # timeout under heavy load.
                    #
                    # Safety: every server endpoint calls
                    # acquire_engine BEFORE awaiting get_engine (see
                    # use_engine ctxmgr + the streaming endpoints in
                    # server.py), so the caller's lease is already
                    # bumped when we return here. The drain monitor's
                    # has_work check (1/sec) is
                    # ``engine.has_active_requests() OR
                    # active_uses > 0`` — both must be False to call
                    # _unload_engine — so the new lease keeps the
                    # engine alive until the caller releases it.
                    # Once active_uses drops to 0 and no in-flight
                    # request remains, drain_monitor proceeds with
                    # unload at its next 1s tick. drain_timeout still
                    # caps pathological extensions: if elapsed >
                    # drain_timeout AND active_uses == 0, the monitor
                    # force-unloads and aborts in-flight requests.
                    if entry.engine is not None:
                        entry.last_access = time.time()
                        logger.info(
                            f"get_engine({model_id}) DRAINING fast-path: "
                            f"returning live engine "
                            f"(active_uses={entry.active_uses}, "
                            f"drain_elapsed="
                            f"{time.time() - entry.drain_started:.1f}s)"
                        )
                        return entry.engine
                    event = entry.drain_complete

                elif entry.state == EngineState.UNLOADING:
                    # Metal cleanup in progress — wait for it to finish,
                    # then the model will need to be reloaded.
                    event = entry.unload_complete

                elif entry.state == EngineState.UNLOADED:
                    # Check if model is too large for the memory ceiling
                    _ceil = self._current_ceiling()
                    if _ceil > 0 and entry.estimated_size > _ceil:
                        raise ModelTooLargeError(
                            model_id, entry.estimated_size, _ceil
                        )

                    # Check load failure cooldown
                    if entry.load_failed_at > 0:
                        elapsed_since_fail = time.time() - entry.load_failed_at
                        if elapsed_since_fail < LOAD_COOLDOWN:
                            raise ModelLoadingError(
                                f"Model {model_id} failed to load "
                                f"{elapsed_since_fail:.0f}s ago, "
                                f"retrying in {LOAD_COOLDOWN - elapsed_since_fail:.0f}s",
                                model_id=model_id,
                            )

                    # Need to load. Check memory, start drains if needed.
                    wait_event = await self._prepare_memory_for(entry)

                    if wait_event is not None:
                        # Not enough memory yet — wait for a drain to free
                        # space, then re-check on next loop iteration
                        event = wait_event
                    else:
                        # Memory is available. Mark as loading so other
                        # callers coalesce on our ready_event.
                        self._set_state(entry, EngineState.LOADING, "get_engine")
                        entry.ready_event = asyncio.Event()
                        entry.load_error = None
                        # Clear any stale abort flag left by a previous
                        # enforcer-aborted load. Without this the leftover
                        # True makes _load_engine abort this fresh load
                        # immediately (see the abort_loading check below), so
                        # a model could never reload after its first
                        # memory-pressure abort — poisoning it until restart.
                        entry.abort_loading = False
                        entry.load_started = time.time()
                        should_load = True
                        logger.info(f"get_engine({model_id}) loading model {entry.model_path}")
                else:
                    logger.info(f"unrecognized state {entry.state.value} for model {model_id}")
                    raise EnginePoolError(f"unrecognized state {entry.state.value} for model {model_id}")

            # --- Outside lock (INV-1) ---

            if should_load:
                # Wait for any pending cleanup tasks (engine.stop +
                # mx.clear_cache) so that Metal memory is actually freed
                # before we start loading.  Without this, the new model
                # load occupies the single MLX executor thread, starving
                # mx.clear_cache() and causing OOM.
                if self._cleanup_tasks:
                    await asyncio.gather(
                        *self._cleanup_tasks, return_exceptions=True
                    )
                    # Second GC + clear_cache pass: the cleanup tasks ran
                    # gc.collect() + mx.clear_cache(), but Metal's page
                    # deallocation is asynchronous.  A second round gives
                    # Metal a chance to reclaim pages before we allocate
                    # new ones for the model load.
                    gc.collect()
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(get_mlx_executor(), locked_sync_and_clear_cache)

                # Load the model (long operation, outside lock)
                load_error = None
                try:
                    await self._load_engine(model_id, force_lm=force_lm)
                except BaseException as e:
                    load_error = e

                # An enforcer-initiated mid-load abort is transient (memory
                # pressure), not a real load failure — detected by the
                # ModelLoadingError "aborted" marker. Computed here so the
                # post-lock retry decision (below) can see it too.
                _enforcer_abort = (
                    isinstance(load_error, ModelLoadingError)
                    and "aborted" in str(load_error)
                )

                # ATOMIC state transition + signal — all under ONE lock hold.
                # This ensures no TOCTOU gap between state change and event
                # signal, and no gap between setting load_error and signaling.
                async with self._tracked_lock("get_engine_post_load"):
                    if load_error is None:
                        self._set_state(
                            entry, EngineState.ACTIVE, "load_complete"
                        )
                        entry.last_access = time.time()
                        result = entry.engine  # Capture under lock
                    else:
                        # Default: record the load failure so waiters
                        # see the typed exception (load_error) and so
                        # quick-retry get_engine() calls hit the
                        # LOAD_COOLDOWN gate (load_failed_at) — this
                        # is what genuine load failures
                        # (corrupt weights, OSError, missing
                        # dependencies, etc.) want.
                        #
                        # Skip recording for known-transient failures:
                        # - ``CancelledError``: caller dropped, not a
                        #   real load failure
                        # - enforcer-initiated abort (ModelLoadingError
                        #   with "aborted" in the message): pressure-
                        #   driven mid-load termination. The model
                        #   itself didn't fail to load; the next
                        #   request can retry immediately. Recording
                        #   here would cascade-503 stress workloads
                        #   that exercise repeated eviction-then-reload
                        #   on small non-pinned models.
                        _skip_record = (
                            isinstance(load_error, asyncio.CancelledError)
                            or _enforcer_abort
                        )
                        if not _skip_record:
                            entry.load_error = load_error
                            entry.load_failed_at = time.time()
                        self._set_state(
                            entry, EngineState.UNLOADED, "load_failed"
                        )
                        result = None
                    entry.ready_event.set()  # Signal under lock — ALWAYS

                if load_error is not None:
                    # Transient enforcer abort: the model is fine, memory was
                    # just tight mid-load. Re-enter the loop to wait for memory
                    # to free (via _prepare_memory_for's drain / exclusive-idle
                    # waits) and retry, rather than surfacing a 503 the caller
                    # can't act on. Bounded by the retry cap and the overall
                    # max_wait_timeout so a genuinely un-loadable model still
                    # raises a clear error.
                    if (
                        _enforcer_abort
                        and enforcer_abort_retries < MAX_ENFORCER_ABORT_RETRIES
                        and (time.monotonic() - start_time)
                        < self._max_wait_timeout
                    ):
                        enforcer_abort_retries += 1
                        logger.info(
                            f"get_engine({model_id}) load aborted by memory "
                            f"enforcer (retry {enforcer_abort_retries}/"
                            f"{MAX_ENFORCER_ABORT_RETRIES}); waiting for "
                            f"memory to free, then retrying"
                        )
                        await asyncio.sleep(ENFORCER_ABORT_RETRY_BACKOFF)
                        continue
                    raise load_error
                return result

            elif event is not None:
                # Wait for state change, then re-check.
                # CancelledError propagates here if client disconnects.
                # Bounded by max_wait_timeout to prevent infinite waits.
                if not isinstance(wait_target, str) or wait_target == "":
                    # Classify the event for logging/timeout handling
                    if event is entry.ready_event:
                        wait_target = "ready_event"
                    elif event is getattr(entry, 'exclusive_idle', None):
                        wait_target = "exclusive_idle"
                    else:
                        # drain_complete or unload_complete from
                        # _clear_for_exclusive or _prepare_memory_for
                        wait_target = "drain_or_unload"
                logger.debug(
                    f"get_engine({model_id}) waiting on {wait_target}, "
                    f"iteration={iterations}, "
                    f"elapsed={time.monotonic() - start_time:.1f}s"
                )

                try:
                    try:
                        await asyncio.wait_for(
                            event.wait(), timeout=self._max_wait_timeout
                        )
                        logger.debug(
                            f"get_engine({model_id}) woke from {wait_target}, "
                            f"re-entering loop (iteration={iterations})"
                        )
                    finally:
                        # Re-acquire the lease released at the exclusive-
                        # contention park on EVERY wait exit — success, the
                        # timeout raise below, or CancelledError (client
                        # disconnect). get_engine must never return OR raise
                        # with the caller's lease consumed: the caller's
                        # paired release_engine() in its finally would
                        # otherwise double-release and steal a SIBLING
                        # request's lease on the same model (active_uses → 0
                        # mid-inference → the drain monitor / enforcer evicts
                        # the engine under it). This also keeps lease-less
                        # get_engine callers (admin/public load endpoints)
                        # net-zero if they ever park here. acquire_engine is
                        # synchronous, so this is cancellation-safe; it runs
                        # before the next dispatch can hand the engine out,
                        # so the use-window stays eviction-protected.
                        if released_for_contention:
                            self.acquire_engine(model_id)
                            released_for_contention = False
                            self._contention_parked -= 1
                except asyncio.TimeoutError:
                    elapsed = time.monotonic() - start_time

                    # If we're waiting for a drain and the draining model
                    # still has active inference, extend the wait — same
                    # logic as the drain monitor's active_uses extension.
                    if wait_target in ("drain_complete", "drain_or_unload", "exclusive_headroom"):
                        draining_entry = self._find_draining_entry(event)
                        if draining_entry is not None and draining_entry.active_uses > 0:
                            logger.warning(
                                f"get_engine({model_id}) wait timeout but "
                                f"drain target {draining_entry.model_id} has "
                                f"active_uses={draining_entry.active_uses}, "
                                f"extending wait "
                                f"(elapsed={elapsed:.1f}s)"
                            )
                            continue

                    self._record_timeout(
                        "get_engine_wait", model_id, elapsed
                    )
                    logger.error(
                        f"get_engine({model_id}) model named {entry.model_path} timed out waiting for {wait_target}, "
                        f"state={entry.state.value}"
                    )
                    raise ModelLoadingError(
                        f"Timed out waiting for model {model_id} "
                        f"({self._max_wait_timeout}s)",
                        model_id=model_id,
                    )

                if __debug__:
                    logger.debug(
                        f"get_engine({model_id}) model named {entry.model_path} woke from {wait_target}, "
                        f"state={entry.state.value}"
                    )

                # If we were waiting on a ready_event that fired due to load
                # failure, propagate the original error immediately instead of
                # looping back and hitting the cooldown check.  The loader set
                # load_error under the lock before setting the event, so the
                # happens-before guarantee from the event makes this safe.
                if entry.load_error is not None and entry.state == EngineState.UNLOADED:
                    raise ModelLoadingError(
                        f"Model {model_id} failed to load: {entry.load_error}",
                        model_id=model_id,
                    )

    def _find_draining_entry(self, event: asyncio.Event) -> EngineEntry | None:
        """Find the EngineEntry whose drain_complete matches *event*."""
        for e in self._entries.values():
            if e.drain_complete is event:
                return e
        return None

    # -------------------------------------------------------------------------
    # Engine use-counting (prevents cooperative eviction while request
    # handlers use an engine)
    #
    # Every server endpoint that holds a reference to an engine MUST bracket
    # its use with acquire_engine / release_engine (or the server-side
    # ``server.use_engine`` context manager, which wraps them). This
    # increments active_uses on the EngineEntry, which is checked by:
    #   - _prepare_memory_for   → drains instead of killing busy engines
    #   - _drain_monitor        → waits for active_uses==0 before unloading,
    #                             and extends the drain if timeout is hit
    #                             while active_uses > 0
    #
    # The process memory enforcer deliberately does NOT respect
    # active_uses on its hard-limit eviction path; it aborts in-flight
    # requests via the engine's abort_all_requests() contract and then
    # unloads. See docs/enforcer-eviction-review.md #2.
    #
    # For non-streaming endpoints, use try/finally:
    #     pool.acquire_engine(resolved_id)
    #     try:
    #         output = await engine.chat(...)
    #     finally:
    #         pool.release_engine(resolved_id)
    #
    # For streaming endpoints, use _with_engine_guard() which releases
    # in its finally block (handles normal completion, errors, and
    # client disconnect / GeneratorExit):
    #     pool.acquire_engine(resolved_id)
    #     return StreamingResponse(
    #         _with_engine_guard(stream_gen, pool, resolved_id)
    #     )
    # -------------------------------------------------------------------------

    def acquire_engine(self, model_id: str) -> None:
        """Increment active_uses for model_id.

        Called by server request handlers immediately after get_engine()
        returns, to protect the engine from eviction by the drain monitor
        and process_memory_enforcer while the handler is using it.

        Must be paired with release_engine().
        """
        entry = self._entries.get(model_id)
        if entry is not None:
            if entry.exclusive and entry.active_uses == 0:
                # Entering exclusive hold — create fresh event for waiters
                entry.exclusive_idle = asyncio.Event()
                entry._exclusive_hold_start = time.time()
                logger.debug(
                    f"acquire_engine({model_id}) exclusive 0→1, "
                    f"created exclusive_idle event"
                )
            entry.active_uses += 1
            if entry.exclusive:
                logger.debug(
                    f"acquire_engine({model_id}) active_uses={entry.active_uses}"
                )

    def ensure_engine_alive(self, model_id: str, engine_ref: Any) -> None:
        """Verify engine_ref is still the live engine for model_id.

        The process memory enforcer may evict a model (setting
        entry.engine = None and scheduling cleanup) at any event-loop
        yield point after a handler captured an engine reference via
        get_engine(). Handlers call this after such yields to detect
        the race and fail fast with EngineEvictedError instead of
        invoking methods on a stale engine.

        Raises:
            EngineEvictedError: if the entry was removed, or its
                engine was replaced, or the engine was unloaded.
        """
        entry = self._entries.get(model_id)
        if entry is None or entry.engine is not engine_ref:
            raise EngineEvictedError(model_id)

    def release_engine(self, model_id: str) -> None:
        """Decrement active_uses for model_id.

        Called when a request handler finishes using the engine.
        """
        entry = self._entries.get(model_id)
        if entry is not None and entry.active_uses > 0:
            entry.active_uses -= 1
            # Request END timestamp. last_access is bumped at get_engine
            # DISPATCH (request start), so for any request longer than the
            # defer grace the post-release active_uses=0 dip would look
            # ancient to _prepare_memory_for's dip check — the grace must
            # anchor on max(last_access, last_release).
            entry.last_release = time.time()
            if entry.exclusive:
                logger.debug(
                    f"release_engine({model_id}) active_uses={entry.active_uses}"
                )
            if entry.exclusive and entry.active_uses == 0:
                entry._exclusive_hold_start = 0.0
                # CRITICAL: Capture the event reference NOW, not later.
                # acquire_engine() may replace entry.exclusive_idle with
                # a fresh Event before _signal_exclusive_idle() runs.
                event_to_signal = entry.exclusive_idle
                # Fire async task: clear Metal cache, then signal waiters
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(
                        self._signal_exclusive_idle(event_to_signal)
                    )
                except RuntimeError:
                    # No running loop (shutdown) — signal directly
                    if event_to_signal is not None:
                        event_to_signal.set()

    # -------------------------------------------------------------------------
    # Exclusive pinned model support
    # -------------------------------------------------------------------------

    async def _clear_for_exclusive(
        self, pinned_entry: EngineEntry
    ) -> asyncio.Event | None:
        """Evict all non-pinned models to maximize headroom for a pinned model.

        Called under self._lock when a request arrives for a pinned+exclusive model.

        Returns:
            None — all non-pinned models cleared, caller may proceed.
            asyncio.Event — a drain or unload is in progress. Caller must
                await this event outside the lock, then re-enter get_engine().
        """
        for mid, e in self._entries.items():
            if e.is_pinned or e is pinned_entry:
                continue

            # Already transitioning — collect an event to wait on
            if e.state == EngineState.DRAINING:
                return e.drain_complete
            if e.state == EngineState.UNLOADING:
                return e.unload_complete

            # Not loaded — skip
            if e.engine is None or e.state != EngineState.ACTIVE:
                continue

            # Loaded non-pinned model — must go
            has_work = (
                e.engine.has_active_requests() or e.active_uses > 0
            )
            if has_work:
                # Active requests → drain gracefully
                self._start_drain(mid)
                logger.info(
                    f"Exclusive headroom: draining '{mid}' "
                    f"(active_uses={e.active_uses})"
                )
                return e.drain_complete
            else:
                # Idle → unload immediately
                logger.info(f"Exclusive headroom: evicting idle '{mid}'")
                await self._unload_engine(mid, reason="exclusive_headroom")
                # Continue loop — may need to evict more models.
                # But _unload_engine sets state to UNLOADING and defers
                # cleanup, so the next iteration will see UNLOADING and
                # return unload_complete if Metal cleanup isn't done yet.

        # All non-pinned models cleared (or never existed)
        return None

    def _grace_recheck_event(self, delay: float = 0.05) -> asyncio.Event:
        """A one-shot event set after *delay* seconds.

        Returned by the exclusive-defer when the exclusive model is at
        ``active_uses == 0`` but was accessed within ``_ACTIVE_GRACE_SEC`` — i.e. a
        request is routed to it and is mid ``get_engine``→``acquire_engine`` (the
        brief active_uses=0 dip). The non-pinned load awaits this, then re-enters
        ``get_engine`` and re-checks: by then the routed request has acquired (→ the
        ``active_uses>0`` defer waits on the real ``exclusive_idle``) or the exclusive
        model is truly idle (grace expired → the load proceeds). This converges in
        ~one or two hops and never orphans an ``exclusive_idle`` waiter, unlike
        clearing the per-acquire event would.
        """
        ev = asyncio.Event()
        # Scheduled on the loop's timer wheel (which holds a strong ref to the
        # TimerHandle) instead of a bare fire-and-forget Task: asyncio keeps
        # only weak references to tasks, so the Task form is one refactor away
        # from being garbage-collected mid-sleep — leaving the waiter stuck
        # for the full max_wait_timeout on an event nobody sets.
        asyncio.get_running_loop().call_later(delay, ev.set)
        return ev

    def _refresh_vision_limits(self, entry: EngineEntry) -> None:
        """Recalculate vision limits from committed memory state.

        Called under self._lock after _clear_for_exclusive() returns None
        (all non-pinned models cleared). Uses _committed_memory() rather
        than mx.get_active_memory() to avoid Metal calls under the lock.
        """
        engine = entry.engine
        if engine is None or not hasattr(engine, 'vision_chunk_budget_pixels'):
            return

        enforcer = self._process_memory_enforcer
        max_bytes = enforcer.get_final_ceiling() if enforcer else 0
        if not max_bytes:
            return

        committed = self._committed_memory()
        headroom = max(0, max_bytes - committed)

        engine.vision_chunk_budget_pixels = int(
            headroom * self._VISION_SAFETY_FACTOR
            / self._VISION_BYTES_PER_PIXEL
        ) if headroom > 0 else 0

        entry._vision_limits_cache = None  # Invalidate cached limits

        logger.debug(
            "Vision limits refreshed: chunk_budget_pixels=%d "
            "(headroom=%.1fGB, committed=%.1fGB, limit=%.1fGB)",
            engine.vision_chunk_budget_pixels,
            headroom / 1e9,
            committed / 1e9,
            max_bytes / 1e9,
        )

    async def _signal_exclusive_idle(
        self, event: asyncio.Event | None
    ) -> None:
        """Clear Metal buffer cache and signal a specific event.

        Runs as a fire-and-forget task spawned by release_engine().
        Takes the EVENT directly (not the entry) to avoid the capture
        race where acquire_engine() replaces entry.exclusive_idle
        before this task runs.

        Must run mx.synchronize + mx.clear_cache on the MLX executor
        so that mx.get_active_memory() reflects the freed KV/vision
        buffers before waiters re-check.
        """
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                get_mlx_executor(),
                locked_sync_and_clear_cache,
            )
        except Exception as e:
            logger.warning(
                f"Metal cache clear failed in exclusive idle signal: {e}"
            )
        finally:
            if event is not None:
                event.set()

    # -------------------------------------------------------------------------
    # Memory preparation (drain instead of reject)
    # -------------------------------------------------------------------------

    async def _prepare_memory_for(
        self, entry: EngineEntry
    ) -> asyncio.Event | None:
        """Ensure enough memory for entry. Starts drains if needed.

        Called under self._lock.

        Returns:
            None if memory is available — caller should proceed with load.
            asyncio.Event if caller must wait for a drain to complete first.
                Caller awaits this event (outside the lock), then re-enters
                get_engine() to re-check.

        Raises:
            ModelTooLargeError: If model can't fit even with all non-pinned
                models evicted (permanent failure, not retryable).
        """
        # Don't load non-pinned models while an exclusive model is actively
        # inferring.  Model loading monopolizes the single MLX executor
        # thread, starving the exclusive model's scheduler steps.  Wait for
        # the exclusive model to finish instead.
        if not entry.is_pinned:
            for mid, e in self._entries.items():
                if e.exclusive and e.is_pinned:
                    logger.debug(
                        f"_prepare_memory_for('{entry.model_id}') "
                        f"checking exclusive '{mid}': "
                        f"active_uses={e.active_uses}, "
                        f"exclusive_idle={'set' if e.exclusive_idle is not None and e.exclusive_idle.is_set() else 'unset' if e.exclusive_idle is not None else 'None'}"
                    )
                    now = time.time()
                    # Rate-limit the "Deferring load …" INFO to 1/_DEFER_LOG_INTERVAL_SEC
                    # per loading model — a non-pinned load re-checks this gate repeatedly
                    # while a chat runs, and one line per check floods the log.
                    _rate_ok = (now - entry._last_defer_log_time) >= _DEFER_LOG_INTERVAL_SEC
                    if e.active_uses > 0 and e.exclusive_idle is not None:
                        if _rate_ok:
                            entry._last_defer_log_time = now
                            logger.info(
                                f"Deferring load of '{entry.model_id}': exclusive "
                                f"model '{mid}' has {e.active_uses} active request(s)"
                            )
                        return e.exclusive_idle
                    # active_uses==0 but ACTIVE within the SHORT defer-grace: a request
                    # is mid get_engine→acquire_engine, or the previous request just
                    # released and the next pipelined one is still routing (the
                    # active_uses=0 dip). Loading a non-pinned model into that window
                    # coexists it with the (e.g. 44GB) exclusive model → OOM (the
                    # 2026-06-05 stress kill: Embedding-4B loaded on the 35B while two
                    # chat completions were still routing). Anchor on BOTH ends of a
                    # request — last_access (dispatch) AND last_release (lease drop) —
                    # because for any request longer than the grace, dispatch time
                    # alone has already aged out by the time the dip begins, leaving
                    # the realistic between-two-chats dip unprotected. Wait it out
                    # with a SINGLE sleep of the remaining grace (not a 50ms poll
                    # loop), then re-check: by then the request has acquired (→ the
                    # defer above) or the model is idle (load proceeds). The short
                    # grace (vs the 3s enforcer grace) avoids holding the load off for
                    # seconds after the chat finished — which flooded the log and
                    # tripped get_engine's excessive-loop guard.
                    last_activity = max(e.last_access, e.last_release)
                    remaining = last_activity + _EXCLUSIVE_DEFER_GRACE_SEC - now
                    if last_activity > 0 and remaining > 0:
                        if _rate_ok:
                            entry._last_defer_log_time = now
                            logger.info(
                                f"Deferring load of '{entry.model_id}': exclusive "
                                f"'{mid}' active_uses=0 dip (re-check in {remaining:.2f}s)"
                            )
                        return self._grace_recheck_event(
                            min(remaining, _EXCLUSIVE_DEFER_GRACE_SEC)
                        )

        # Don't load while another model is still cleaning up Metal resources.
        # UNLOADING means engine.stop() + mx.synchronize() + mx.clear_cache()
        # is in progress — Metal buffers are still allocated.
        for mid, e in self._entries.items():
            if (
                e.state == EngineState.UNLOADING
                and e.unload_complete is not None
            ):
                if __debug__:
                    logger.debug(
                        f"Waiting for {mid} to finish unloading "
                        f"before loading {entry.model_id}"
                    )
                return e.unload_complete

        # Serialize model loading: only one model may be in LOADING state at
        # a time. Concurrent loads can exhaust Metal memory because weight
        # files are read into GPU memory on the MLX executor thread, and if
        # two models load simultaneously their combined peak memory exceeds
        # the process limit even though each fits individually.
        for mid, e in self._entries.items():
            if (
                mid != entry.model_id
                and e.state == EngineState.LOADING
                and e.ready_event is not None
            ):
                if __debug__:
                    logger.debug(
                        f"Serializing load: waiting for {mid} to finish "
                        f"loading before starting {entry.model_id}"
                    )
                return e.ready_event

        ceiling = self._current_ceiling()
        if ceiling <= 0:
            # No memory ceiling — also check process memory
            return await self._check_process_memory(entry)

        required = entry.estimated_size
        if entry.model_type not in ("audio_stt", "audio_tts", "audio_sts", "embedding", "reranker"):
            required += int(entry.estimated_size * 0.25)  # KV headroom

        # Reservation-aware (§3b): size against committed weights PLUS
        # in-flight inference transients so a load that would collide with an
        # outstanding non-streaming decode evicts/waits instead of committing
        # weights over the wall (closes the load↔inference race).
        # §P5: ALSO count the live LLM/VLM generation transient
        # (_scheduler_inflight_bytes — the same term _reservable_free already
        # subtracts on the inference side). Without it, load admission was
        # blind to a RUNNING chat generation: the 2026-06-09 round-3 stress
        # crash admitted an Embedding-4B load to "total: 49.46GB" on a 51.8GB
        # cap while the 35B was mid-decode — its KV/decode transient then had
        # ~2.3GB of room, and the next scheduler step Metal-panicked. With
        # this term the load waits (exclusive_idle / drain) and loads after.
        while (
            self._committed_plus_reservations()
            + self._scheduler_inflight_bytes()
            + required
        ) > ceiling:
            victim_id = self._find_drain_or_evict_candidate()
            if victim_id is None:
                # Can't evict anything. Find a draining model to wait on.
                for mid, e in self._entries.items():
                    if e.state == EngineState.DRAINING:
                        return e.drain_complete  # Caller waits, then retries

                # Nothing draining — check if something is LOADING
                for mid, e in self._entries.items():
                    if e.state == EngineState.LOADING and e.ready_event is not None:
                        return e.ready_event  # Wait for loading model to finish

                # Nothing draining or loading — check if something is UNLOADING
                for mid, e in self._entries.items():
                    if e.state == EngineState.UNLOADING and e.unload_complete is not None:
                        return e.unload_complete  # Wait for Metal cleanup

                # Nothing draining, loading, or unloading.
                # Try without KV headroom as a last resort (§P5: still counts
                # the live generation transient — the headroom concession is
                # about OUR KV estimate, never about the running chat's room).
                required_no_headroom = entry.estimated_size
                if (
                    self._committed_plus_reservations()
                    + self._scheduler_inflight_bytes()
                    + required_no_headroom
                    <= ceiling
                ):
                    logger.info(
                        f"Loading {entry.model_id} without KV headroom "
                        f"(need {format_size(required)}, "
                        f"available "
                        f"{format_size(ceiling - self._committed_plus_reservations())})"
                    )
                    break  # Proceed without headroom

                # §P5: the live LLM/VLM generation transient is (part of) what
                # blocks this load. It clears when the exclusive model goes
                # idle — wait on that, then retry with the room it frees.
                if self._scheduler_inflight_bytes() > 0:
                    for _mid, _e in self._entries.items():
                        if (
                            _e.exclusive and _e.is_pinned
                            and _e.active_uses > 0
                            and _e.exclusive_idle is not None
                        ):
                            logger.info(
                                f"Load of {entry.model_id} waiting for the "
                                f"exclusive generation on '{_mid}' to finish "
                                f"(live transient "
                                f"{format_size(self._scheduler_inflight_bytes())})"
                            )
                            return _e.exclusive_idle

                # §P2: over the ceiling ONLY because of in-flight inference
                # transients (committed weights alone leave room) — those
                # release in seconds. Wait for the next release pulse instead
                # of failing the load with a permanent-style 507 for a
                # condition that clears on its own. get_engine's wait timeout
                # is the backstop for a wedged release.
                if (
                    self._inflight_reservations > 0
                    and self._committed_memory() + required_no_headroom
                    <= ceiling
                ):
                    logger.info(
                        f"Load of {entry.model_id} waiting for "
                        f"{format_size(self._inflight_reservations)} of "
                        f"in-flight inference transients to release "
                        f"(weights fit; transient pressure only)"
                    )
                    return self._reservation_released

                # Truly stuck (all pinned)
                raise ModelTooLargeError(
                    entry.model_id, required, ceiling
                )

            victim = self._entries[victim_id]
            victim_busy = (
                victim.engine is not None
                and (victim.engine.has_active_requests()
                     or victim.active_uses > 0)
            )
            if victim_busy:
                # Victim has in-flight requests or active uses — drain, don't kill
                self._start_drain(victim_id)
                # Return the drain event — caller waits for it, then retries
                return victim.drain_complete
            else:
                # Victim is idle — unload immediately (existing behavior)
                await self._unload_engine(victim_id, reason="evict_idle")
                # Loop back to re-check if we freed enough

        # Check process memory limit too
        return await self._check_process_memory(entry)

    async def _check_process_memory(
        self, entry: EngineEntry
    ) -> asyncio.Event | None:
        """Check process memory limit before loading.

        Called under self._lock. Returns None if OK, or a drain event to wait on.

        Note: mx.get_active_memory() reflects Metal allocator state which may
        lag behind our evictions (gc.collect + mx.clear_cache don't guarantee
        immediate Metal deallocation). When all evictable models are gone but
        Metal memory is still high, we fall back to checking _committed_memory()
        which tracks our known model weights. If that fits, we proceed — Metal
        will reclaim the memory during or shortly after the load.
        """
        if self._process_memory_enforcer is None:
            return None

        enforcer = self._process_memory_enforcer
        if enforcer.get_final_ceiling() <= 0:
            return None

        while True:
            # max(active, phys_footprint) matches what jetsam sees and
            # what ProcessMemoryEnforcer uses, so load decisions are
            # consistent. psutil/active-memory underreports IOAccelerator-
            # backed Metal on Apple Silicon UMA (95 GB gap on 31B+32k).
            current_active = max(mx.get_active_memory(), get_phys_footprint())
            # Reservation-aware (§3b): add outstanding inference transients so
            # a load racing a non-streaming decode evicts/waits. current_active
            # = max(active, phys) may already include a *materialized*
            # transient, so this can double-count — deliberately conservative
            # (errs toward evicting a racing load, never toward OOM).
            # §P5: also add the live LLM/VLM generation transient — the
            # NOT-yet-materialized KV/decode growth of a running chat.
            # current_active sees only what has materialized so far; the
            # round-3 stress crash admitted a load with the 35B mid-decode
            # and its next steps Metal-panicked into the room the load took.
            # Double-counting the materialized part is the same deliberate
            # conservatism as the reservation term above.
            projected = (
                current_active + entry.estimated_size
                + self._inflight_reservations
                + self._scheduler_inflight_bytes()
            )
            if projected <= enforcer.get_final_ceiling():
                return None

            # Try to evict/drain an LRU model to free memory
            victim_id = self._find_drain_or_evict_candidate()
            if victim_id is not None:
                victim = self._entries[victim_id]
                victim_busy = (
                    victim.engine is not None
                    and (victim.engine.has_active_requests()
                         or victim.active_uses > 0)
                )
                if victim_busy:
                    self._start_drain(victim_id)
                    return victim.drain_complete
                else:
                    logger.info(
                        f"Evicting '{victim_id}' to fit '{entry.model_id}' "
                        f"within process memory limit "
                        f"({format_size(projected)} > "
                        f"{format_size(enforcer.get_final_ceiling())})"
                    )
                    await self._unload_engine(
                        victim_id, reason="evict_process_memory"
                    )
                    continue

            # Check if anything is draining
            for mid, e in self._entries.items():
                if e.state == EngineState.DRAINING:
                    return e.drain_complete

            # No more victims to evict. If there are pending cleanup tasks,
            # wait for them to finish (mx.clear_cache) before re-checking.
            if self._cleanup_tasks:
                # Release the lock temporarily so cleanup can proceed,
                # then return a synthetic event to re-enter get_engine().
                wait_event = asyncio.Event()

                async def _wait_for_cleanup():
                    await asyncio.gather(
                        *self._cleanup_tasks, return_exceptions=True
                    )
                    wait_event.set()

                asyncio.create_task(_wait_for_cleanup())
                return wait_event

            # All cleanup done. Re-check actual Metal memory — Metal's
            # allocator may not have reclaimed pages even after clear_cache.
            current_active = max(mx.get_active_memory(), get_phys_footprint())
            # Reservation-aware (§3b) + §P5 scheduler term: same conservative
            # sum as the initial projection above so both reads agree under a
            # live decode/generation.
            projected = (
                current_active + entry.estimated_size
                + self._inflight_reservations
                + self._scheduler_inflight_bytes()
            )
            if projected <= enforcer.get_final_ceiling():
                logger.info(
                    f"Process memory after cleanup: "
                    f"{format_size(current_active)} + "
                    f"{entry.model_id} ({format_size(entry.estimated_size)}) "
                    f"= {format_size(projected)} <= "
                    f"{format_size(enforcer.get_final_ceiling())}. Proceeding."
                )
                return None

            # If an exclusive model has active requests, its runtime memory
            # (KV cache, vision buffers) will be freed when it finishes.
            # Wait for it rather than loading into contested memory or
            # rejecting with an immediate 507.
            for mid, e in self._entries.items():
                if e.exclusive and e.is_pinned:
                    logger.debug(
                        f"_check_process_memory('{entry.model_id}') "
                        f"checking exclusive '{mid}': "
                        f"active_uses={e.active_uses}, "
                        f"exclusive_idle={'set' if e.exclusive_idle is not None and e.exclusive_idle.is_set() else 'unset' if e.exclusive_idle is not None else 'None'}"
                    )
                    if e.active_uses > 0 and e.exclusive_idle is not None:
                        logger.info(
                            f"Waiting for exclusive model '{mid}' to finish "
                            f"({e.active_uses} active request(s)) before loading "
                            f"'{entry.model_id}'"
                        )
                        return e.exclusive_idle

            # Projected memory still too high. Check if committed model
            # weights alone fit and the overshoot is modest (within 25%
            # headroom). Metal may reuse freed pages during the load.
            # Reservation-aware (§3b): include outstanding inference
            # transients so a load can't slip in under a live decode.
            committed = self._committed_plus_reservations()
            headroom = int(enforcer.get_final_ceiling() * 0.25)
            if (
                committed + entry.estimated_size <= enforcer.get_final_ceiling()
                and projected <= enforcer.get_final_ceiling() + headroom
            ):
                logger.warning(
                    f"Process memory after cleanup still high "
                    f"({format_size(current_active)}), but committed "
                    f"({format_size(committed)}) + "
                    f"{entry.model_id} ({format_size(entry.estimated_size)}) "
                    f"fits. Metal residual "
                    f"{format_size(current_active - committed)} may be "
                    f"reclaimed during load. Proceeding cautiously."
                )
                return None

            # §P2: the projection includes in-flight inference transients
            # (and may double-count a materialized one). If removing them
            # brings the load within the cautious-proceed bound, they release
            # in seconds — wait for the next release pulse instead of a
            # permanent-style 507.
            if (
                self._inflight_reservations > 0
                and projected - self._inflight_reservations
                <= enforcer.get_final_ceiling() + headroom
            ):
                logger.info(
                    f"Load of {entry.model_id} waiting for "
                    f"{format_size(self._inflight_reservations)} of in-flight "
                    f"inference transients to release (projected "
                    f"{format_size(projected)} fits without them)"
                )
                return self._reservation_released

            # Truly cannot fit — even with cleanup done, Metal retains too
            # much memory for the new model to load safely.
            raise InsufficientMemoryError(
                required=entry.estimated_size,
                current=current_active,
                message=(
                    f"Cannot load {entry.model_id}: projected memory "
                    f"{format_size(projected)} would exceed process "
                    f"limit {format_size(enforcer.get_final_ceiling())} "
                    f"(current Metal: {format_size(current_active)}, "
                    f"committed: {format_size(committed)}, "
                    f"model: {format_size(entry.estimated_size)})"
                ),
            )

    # -------------------------------------------------------------------------
    # Drain mechanism
    # -------------------------------------------------------------------------

    def _start_drain(self, model_id: str) -> None:
        """Mark a model as draining. Called under self._lock."""
        entry = self._entries[model_id]
        self._set_state(entry, EngineState.DRAINING, "evict_for_memory")
        entry.drain_started = time.time()
        entry.drain_complete = asyncio.Event()

        # Launch monitor (runs as independent task)
        asyncio.create_task(self._drain_monitor(model_id))

    async def _drain_monitor(self, model_id: str) -> None:
        """Unload model when drained or timed out.

        All checks and state transitions happen under self._lock to prevent
        TOCTOU races (e.g., has_active_work returns False, but a stale
        reference is used after unload).

        Checks both has_active_requests() (in-flight GPU requests in the
        scheduler) AND active_uses > 0 (request handlers that have acquired
        the engine via acquire_engine but may not have started GPU work yet).
        If the drain timeout is reached but active_uses > 0, the drain is
        extended rather than force-unloading — this prevents stopping an
        engine while a request handler is mid-validation or mid-stream.
        """
        entry = self._entries[model_id]

        try:
            while True:
                await asyncio.sleep(1)

                # ATOMIC: check state + active work + unload under ONE lock hold
                async with self._tracked_lock("drain_monitor"):
                    if entry.state != EngineState.DRAINING:
                        return  # State changed externally (e.g., shutdown)

                    elapsed = time.time() - entry.drain_started

                    has_work = (
                        entry.engine is not None
                        and (entry.engine.has_active_requests()
                             or entry.active_uses > 0)
                    )
                    if not has_work:
                        # All requests finished — unload
                        logger.info(
                            f"Drain complete for {model_id} "
                            f"({elapsed:.1f}s), unloading"
                        )
                        await self._unload_engine(
                            model_id,
                            reason=f"drain_complete({elapsed:.1f}s)",
                        )
                        entry.drain_complete.set()
                        return

                    if elapsed > self._drain_timeout:
                        if entry.active_uses > 0:
                            if not hasattr(entry, '_last_drain_ext_log') or (
                                time.time() - entry._last_drain_ext_log >= 5
                            ):
                                logger.warning(
                                    f"Drain timeout for {model_id} but "
                                    f"active_uses={entry.active_uses}, "
                                    f"extending drain"
                                )
                                entry._last_drain_ext_log = time.time()
                            continue

                        self._record_timeout("drain", model_id, elapsed)

                        # Abort sends error to collectors so clients see a
                        # proper error, not a silent disconnect
                        engine_core = getattr(entry.engine, '_engine', None)
                        if engine_core is not None:
                            if hasattr(engine_core, 'abort_all_requests'):
                                try:
                                    await engine_core.abort_all_requests()
                                except Exception as abort_err:
                                    logger.warning(
                                        f"Error aborting requests during drain "
                                        f"timeout for {model_id}: {abort_err}"
                                    )

                        await self._unload_engine(
                            model_id,
                            reason=f"drain_timeout({elapsed:.1f}s)",
                        )
                        entry.drain_complete.set()
                        return

        except Exception as e:
            logger.error(f"drain_monitor crashed for {model_id}: {e}")
            # Emergency: unload to unblock waiters
            try:
                async with self._tracked_lock("drain_monitor_crash"):
                    if entry.engine is not None:
                        await self._unload_engine(
                            model_id, reason="drain_crash"
                        )
                    elif entry.state != EngineState.UNLOADED:
                        self._set_state(
                            entry, EngineState.UNLOADED, "drain_crash"
                        )
            except Exception:
                entry.state = EngineState.UNLOADED  # Best-effort
        finally:
            # ALWAYS signal, even on crash — prevents infinite waiter blocking
            # (INV-3)
            if entry.drain_complete and not entry.drain_complete.is_set():
                entry.drain_complete.set()

    # -------------------------------------------------------------------------
    # Active work detection
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # Victim selection
    # -------------------------------------------------------------------------

    def _find_drain_or_evict_candidate(
        self, exclude_model_id: str | None = None
    ) -> str | None:
        """
        Find the least recently used non-pinned loaded model suitable for
        eviction or draining. Skips models already in DRAINING or UNLOADING state.

        ``exclude_model_id`` is skipped during candidate selection (used by
        inference admission to exclude the op's OWN engine). It must be excluded
        HERE, not dropped from the single returned result, or a caller whose own
        engine is the LRU gets ``None`` and misses other evictable models (the
        2026-06-02 best-effort-OOM bug).

        Prefers idle models over models with active requests or active
        use-count (request handlers that have acquired the engine via
        acquire_engine).

        Idle is computed as ``has_active_requests() == False AND
        active_uses == 0 AND last_access > _ACTIVE_GRACE_SEC ago``.
        The last clause closes the race where a tick samples between
        ``release_engine`` of request N and ``acquire_engine`` of
        request N+1 (a few ms window). Without it, a model being
        actively served by back-to-back requests can look idle for the
        instant the enforcer samples and get picked for eviction.

        Note: the grace window is consulted HERE for ranking purposes
        only (recently-accessed engines sort after fully-idle ones).
        The enforcer's busy-victim filter uses
        ``is_engine_actively_held`` (no grace) — sustained traffic
        would otherwise prevent any eviction at all.

        Returns:
            Model ID of the candidate, or None if no evictable models exist.
        """
        now = time.time()
        candidates = []
        for mid, e in self._entries.items():
            if mid == exclude_model_id:
                continue  # never evict the caller's own engine
            if e.engine is None or e.is_pinned:
                continue
            if e.state in (
                EngineState.DRAINING,
                EngineState.LOADING,
                EngineState.UNLOADING,
            ):
                continue  # Lifecycle state in progress
            actively_held = self.is_engine_actively_held(e)
            recently_active = (
                e.last_access > 0
                and (now - e.last_access) < _ACTIVE_GRACE_SEC
            )
            # has_active=True ranks LAST in the sort tuple
            candidates.append(
                (actively_held or recently_active, e.last_access, mid)
            )
        if not candidates:
            return None
        candidates.sort()  # (False, old_time) sorts before (True, old_time)
        return candidates[0][2]

    # -- In-flight inference admission (§3a / §3c) -------------------------

    def _pick_inference_victim(self, model_id: str) -> str | None:
        """LRU non-pinned victim to evict for an inference reservation.

        Delegates to ``_find_drain_or_evict_candidate`` with the op's OWN model
        EXCLUDED from selection (it already skips ``is_pinned`` + lifecycle).
        Excluding here (not dropping the single returned result) is essential:
        the op holds its ``acquire_engine`` lease, so its engine is often the
        LRU; a drop-after-pick would then return None and miss other evictable
        models, sending the op to best-effort → Metal OOM (2026-06-02). Called
        under self._lock.
        """
        return self._find_drain_or_evict_candidate(exclude_model_id=model_id)

    def _victim_busy(self, model_id: str) -> bool:
        """Whether an eviction victim has in-flight work (drain, don't kill).

        Delegates to ``is_engine_actively_held`` — the pool's single busy
        predicate (already used by ``_find_drain_or_evict_candidate`` and the
        process_memory_enforcer) — so "actively held" cannot drift between
        victim selection and the drain-vs-kill choice, and a torn-down engine
        mid-eviction reads not-busy instead of raising. Called under
        self._lock.
        """
        victim = self._entries.get(model_id)
        return victim is not None and self.is_engine_actively_held(victim)

    def _pending_reclaim_event(
        self, exclude_model_id: str | None = None
    ) -> asyncio.Event | None:
        """First in-flight reclaim event to await: UNLOADING, DRAINING, then a
        LOADING non-pinned model (wait for the load, then a re-loop evicts it).

        Used by ``reserve_inference`` when nothing non-pinned can be picked
        as a fresh victim but a reclaim (or a load that will become evictable)
        is already underway — wait for it
        (the freed weights drop out of ``_committed_memory`` once cleanup
        completes) rather than admitting best-effort over the wall. Mirrors
        the UNLOADING/DRAINING wait fall-throughs in ``_prepare_memory_for``.
        Called under self._lock.

        ``exclude_model_id`` is skipped: a non-streaming op must NEVER wait on
        the reclaim of its OWN engine. It holds that engine's ``active_uses``
        lease, so the drain/unload cannot complete until the op finishes —
        waiting on it here would deadlock the op against its own completion
        (the 2026-06-02 exclusive-headroom-drain livelock).
        """
        for mid, e in self._entries.items():
            if mid == exclude_model_id:
                continue
            if e.state == EngineState.UNLOADING and e.unload_complete is not None:
                return e.unload_complete
        for mid, e in self._entries.items():
            if mid == exclude_model_id:
                continue
            if e.state == EngineState.DRAINING and e.drain_complete is not None:
                return e.drain_complete
        # A non-pinned model still LOADING is not evictable YET, but it will be
        # once loaded — wait for it, then a re-loop evicts it. Without this, a
        # concurrent load (e.g. an Embedding request) that commits memory an
        # in-flight TTS decode cannot reclaim sends the op to best-effort →
        # Metal OOM (the 2026-06-02 single-shot crash). Skip pinned (never an
        # eviction victim) and self.
        for mid, e in self._entries.items():
            if mid == exclude_model_id or e.is_pinned:
                continue
            if e.state == EngineState.LOADING and e.ready_event is not None:
                return e.ready_event
        return None

    def _resolve_reservation_key(self, model_id: str) -> str:
        """Map a non-streaming engine's identifier to the ``_entries`` key.

        Engines self-reserve with ``model_name`` — the full model PATH — but
        ``_entries`` is keyed by the short ``model_id``. Return the matching key
        (by ``model_path``) so ``reserve_inference``'s own-engine guard,
        self-skip, and reclaim-exclusion resolve the right entry; without this
        they silently no-op on a path/key mismatch and the op waits on its OWN
        drain (the 2026-06-02 livelock). Falls back to the input unchanged
        (already a key, or unknown). O(n) over a handful of loaded models.
        """
        if model_id in self._entries:
            return model_id
        for mid, e in self._entries.items():
            if e.model_path == model_id:
                return mid
        return model_id

    @asynccontextmanager
    async def reserve_inference(self, model_id: str, est_bytes: int):
        """Reserve working-set headroom for an in-flight non-streaming op.

        Folds the op's estimated transient into ``_inflight_reservations``
        (the same number load admission reads, §3b) under ``self._lock``,
        budgeted against the Metal wall minus margin (``_wall_budget``). An
        op that fits proceeds immediately; an op that would breach the wall
        waits for an in-flight release or evicts a non-pinned, non-self
        victim — reusing the pinned-safe ``_find_drain_or_evict_candidate``
        / ``_unload_engine`` / ``_start_drain`` machinery.

        Strict no-op (no lock, no state) when there is no Metal cap
        (``_wall_budget() == 0``), the op declares no transient
        (``est_bytes <= 0``) — §5 — or the transient is SMALL
        (``est_bytes <= _small_transient_threshold()``): absorbing small
        unaccounted transients is the inference MARGIN's designed job, and
        reserving them is actively harmful. With a large pinned model
        resident, every op is "over budget" by definition, so reserving a
        0.3–0.6 GiB forward turned each one into an admission event — victim
        evictions (``evict_for_inference``), reloads whose mlx load
        temporaries spike ~2× the model size (#429), and synchronized
        wake-ups at release boundaries. The 2026-06-09 stress campaign
        crashed four ways on that churn while the un-reserved system passed
        16/16 at the same wall: the §3d goal is to stop TWO BIG transients
        stacking, not to meter noise the margin already covers.

        Args:
            model_id: The model whose engine is performing the op. Never
                evicted (self-exclusion) so a long decode can't evict itself.
            est_bytes: Estimated peak working-set transient for the op
                (``estimate_working_set_bytes``). 0 ⇒ no-op.

        Release is guaranteed on every exit path (normal return, exception,
        cancellation) via the ``finally`` — a crashed/aborted op cannot leak
        a reservation and wedge admission.
        """
        budget = self._wall_budget()
        if est_bytes <= self._small_transient_threshold() or budget <= 0:
            # Non-Apple / no cap / no estimate: strict no-op (§5).
            yield
            return
        # Engines self-reserve with their model_name (a full PATH); _entries is
        # keyed by the short model_id. Resolve to the real key so the guard /
        # self-skip / reclaim-exclusion below match the right entry — otherwise
        # they silently no-op and the op can wait on its OWN drain (2026-06-02).
        model_id = self._resolve_reservation_key(model_id)
        reserved = False
        contended = False  # §P6: this admit ran over budget
        try:
            while True:
                async with self._tracked_lock("reserve_inference"):
                    if est_bytes <= self._reservable_free():
                        # FITS: take the reservation and proceed immediately.
                        self._inflight_reservations += est_bytes
                        reserved = True
                        break
                    victim = self._pick_inference_victim(model_id)  # pinned-/self-safe
                    if victim is not None and self._victim_busy(victim):
                        # §P3: deterministic drain DIRECTION for busy victims.
                        # Two concurrent over-budget ops on different busy
                        # models used to pick EACH OTHER, start mutual drains,
                        # and park on each other's drain_complete — neither
                        # drain can finish while both ops hold their leases (a
                        # 300s standoff that ends with BOTH decoding over the
                        # wall). Only the op whose pool key sorts smaller may
                        # drain a busy victim; the other falls through to the
                        # wait/yield paths below.
                        if model_id < victim:
                            self._start_drain(victim)
                            ev = self._entries[victim].drain_complete
                        else:
                            victim = None  # disallowed direction — no drain
                    elif victim is not None:
                        # Phase-1 unload: flips engine=None + schedules the
                        # DEFERRED cleanup; weights stay counted until the
                        # unload_complete event fires (so we await it below,
                        # outside the lock, before re-checking — we do NOT
                        # admit best-effort while the victim is still
                        # committed). Mirrors _prepare_memory_for.
                        await self._unload_engine(
                            victim, reason="evict_for_inference"
                        )
                        ev = self._entries[victim].unload_complete
                    if victim is None:
                        # Nothing evictable in OUR power (pinned + self only,
                        # or a busy victim in the disallowed §P3 direction).
                        # Wait on an INDEPENDENT reclaim already underway if
                        # one exists (excludes self, so it can never wait on
                        # us — cycle-free even while our own model drains).
                        ev = self._pending_reclaim_event(model_id)
                        if ev is None:
                            # CONTENDED ADMIT, margin-bounded (§P2/§P3+). The
                            # wall budget already excludes the inference
                            # margin, so over-budget ops may dip INTO that
                            # margin — never past the raw cap: admit while the
                            # TOTAL outstanding transients stay within
                            # _overshoot_allowance() (small forwards keep
                            # interleaving under over-commit; a SECOND
                            # TTS-sized transient parks — two stacked is the
                            # 2-TTS Metal panic, re-confirmed 2026-06-09 when
                            # an evict_for_memory drain made BOTH queued TTS
                            # ops yield within 20ms). inflight==0 always
                            # admits (SOLO overshoot — with nothing running
                            # there is no pulse to wait for, and one bounded
                            # overshoot is the old-world behavior; also what
                            # unblocks a drain waiting on OUR lease). The
                            # waiters park on the release pulse: reservation
                            # holders are by construction RUNNING, never
                            # parked, so a pulse always comes — no deadlock,
                            # and the winner's drain completes as the chain's
                            # leases drop one by one.
                            if (
                                self._inflight_reservations == 0
                                or self._inflight_reservations + est_bytes
                                <= self._overshoot_allowance()
                            ):
                                self._inflight_reservations += est_bytes
                                reserved = True
                                contended = True
                                self._note_contended_admit()  # §P6 corrector
                                logger.warning(
                                    "reserve_inference: %s over budget, "
                                    "nothing reclaimable — contended admit "
                                    "within the margin (outstanding %s)",
                                    model_id,
                                    format_size(self._inflight_reservations),
                                )
                                break
                            ev = self._reservation_released
                # ---- outside the lock: await the reclaim, then re-loop ----
                # We only ever wait on ANOTHER model's reclaim — the victim we
                # just evicted, or a reclaim already underway. We NEVER wait on
                # our OWN engine's reclaim (``_pending_reclaim_event`` excludes
                # ``model_id``): doing so would deadlock the op against its own
                # active_uses lease (the 2026-06-02 livelock). Another model's
                # reclaim completes independently of this op, so this cannot
                # deadlock; ``_max_wait_timeout`` is the backstop for a wedged
                # one. We do NOT proceed early just because our own engine is
                # draining — that would skip making room and decode over the
                # wall (the 2-TTS OOM); we make room first, then run.
                try:
                    await asyncio.wait_for(
                        ev.wait(), timeout=self._max_wait_timeout
                    )
                except asyncio.TimeoutError:
                    # Reclaim wedged — don't fail a healthy request; best-effort.
                    async with self._tracked_lock("reserve_inference"):
                        self._inflight_reservations += est_bytes
                        reserved = True
                        contended = True
                        self._note_contended_admit()  # §P6 corrector
                    break
                # Re-acquire, recompute _reservable_free: the victim may now be
                # fully UNLOADED (weights dropped from _committed_memory).
            yield
        finally:
            if reserved:
                # LOCK-FREE on purpose. Awaiting the pool lock inside a
                # finally can itself be interrupted — a second CancelledError
                # (disconnect + shutdown double-cancel) lands at the lock
                # acquire, and _tracked_lock's 60s acquire timeout raises
                # EnginePoolError — either skips the decrement and leaks a
                # multi-GB reservation forever (no reconciliation exists).
                # All mutators run on the event loop, so plain int
                # arithmetic between awaits is atomic; the scheduler's
                # cross-thread reader was always a lock-free GIL-atomic load.
                self._inflight_reservations = max(
                    0, self._inflight_reservations - est_bytes
                )
                if contended:
                    # §P6: the corrector loop exits on its own once this hits
                    # 0 (or the budget fits again). Plain int — same lock-free
                    # rationale as the decrement above.
                    self._contended_admits = max(0, self._contended_admits - 1)
                # §P2 release pulse: wake waiters parked on "a transient will
                # clear in seconds". Swap-then-set (all sync — still safe in
                # this lock-free finally): waiters that grabbed the OLD event
                # wake now; later waiters grab the fresh one.
                released_pulse = self._reservation_released
                self._reservation_released = asyncio.Event()
                released_pulse.set()

    def is_engine_actively_held(self, entry: EngineEntry) -> bool:
        """Strict check: handler currently holds a lease OR scheduler is running.

        This is what the process_memory_enforcer's busy-victim filter
        wants — "if I abort this engine right now, will I break an
        in-flight request?". Active scheduler work or a positive
        ``active_uses`` count is the only ground truth for that.

        The grace window (recently_active) is NOT included here. It's a
        sort-ranking signal for ``_find_drain_or_evict_candidate`` so
        models in between back-to-back requests get ranked after fully
        idle ones, NOT a "can't ever evict" lock. If we treated grace as
        a hard block in the enforcer, sustained traffic would prevent
        any eviction at all because every model's ``last_access`` keeps
        refreshing inside the grace window — memory would grow past
        limit without recovery.
        """
        if entry.engine is None:
            return False
        try:
            return entry.engine.has_active_requests() or entry.active_uses > 0
        except (AttributeError, TypeError):
            return False

    # -------------------------------------------------------------------------
    # Engine unloading
    # -------------------------------------------------------------------------

    async def _unload_engine(self, model_id: str, *, reason: str = "unload") -> None:
        """
        Immediately stop and unload an engine.

        Sets state to UNLOADED. This aborts any in-progress requests.

        IMPORTANT: This method is often called under self._lock.  The state
        transition and engine reference clearing happen quickly (no awaits),
        then the heavy async cleanup (engine.stop + mx.clear_cache) runs
        AFTER the caller releases the lock via _deferred_engine_cleanup().
        This prevents the pool lock from being held while waiting for the
        MLX executor, which would block all get_engine() callers and cause
        cascading timeouts.

        Args:
            model_id: The model ID to unload
            reason: Reason string for state transition logging
        """
        entry = self._entries.get(model_id)
        if not entry or entry.engine is None:
            # Still set state for consistency if entry exists.
            # Don't overwrite UNLOADING — deferred cleanup will handle it.
            if entry is not None and entry.state not in (
                EngineState.UNLOADED, EngineState.UNLOADING,
            ):
                self._set_state(entry, EngineState.UNLOADED, reason)
            return

        logger.info(f"Unloading model: {model_id}")

        # Phase 1: Stop accepting new requests (immediate, safe under lock).
        # Set UNLOADING — not UNLOADED — so memory accounting keeps this
        # model's memory committed until Metal buffers are actually freed.
        engine_to_stop = entry.engine
        entry.engine = None  # prevent new requests
        entry.last_access = 0.0
        entry.last_release = 0.0  # clear the dip-grace anchor with it
        entry.actual_size = None  # observed size lost when engine unloads
        entry.unload_complete = asyncio.Event()
        self._set_state(entry, EngineState.UNLOADING, reason)

        # Phase 2: Schedule heavy async cleanup as an independent task so the
        # pool lock is NOT held during engine.stop() and mx.clear_cache().
        # _deferred_engine_cleanup will set state to UNLOADED once Metal is done.
        task = asyncio.create_task(
            self._deferred_engine_cleanup(model_id, engine_to_stop)
        )
        self._cleanup_tasks.append(task)
        task.add_done_callback(lambda t: self._cleanup_tasks.remove(t)
                               if t in self._cleanup_tasks else None)

    async def _deferred_engine_cleanup(
        self, model_id: str, engine: Any
    ) -> None:
        """Stop engine and clear Metal cache outside the pool lock.

        This runs as an independent asyncio task so that _unload_engine()
        (which is typically called under self._lock) returns immediately
        after the state transition.  The actual engine teardown and
        mx.clear_cache() happen here without holding the pool lock,
        preventing cascading timeouts on get_engine() callers.
        """
        try:
            await asyncio.wait_for(engine.stop(), timeout=30)
        except asyncio.TimeoutError:
            self._record_timeout("engine_stop", model_id, 30.0)
            # Force cleanup without waiting for graceful stop
            try:
                engine_core = getattr(engine, '_engine', None)
                if engine_core is not None:
                    inner = getattr(engine_core, 'engine', None)
                    if inner is not None:
                        inner._running = False
            except Exception:
                pass
        except Exception as e:
            logger.warning(f"Error stopping engine for {model_id}: {e}")

        # Drop the engine reference so model tensors can be collected.
        # Without this, gc.collect() can't free the MLX arrays because
        # they're still reachable via the local `engine` variable.
        del engine

        # Force garbage collection + clear_cache to release memory, ALL on the
        # global MLX executor under the buffer lock (locked_free_and_clear).
        # Doing the gc.collect() on the event-loop thread frees the victim's
        # MLX arrays concurrently with in-flight generation on the executor,
        # which corrupts that generation (e.g. garbled TTS audio). Running the
        # free on the executor serializes it with generation. See issue #85.
        # Synchronize before clearing to prevent releasing Metal buffers
        # still referenced by in-flight command buffers. See issue #300.
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                get_mlx_executor(), locked_free_and_clear
            )
        finally:
            # Phase 2 complete: transition from UNLOADING → UNLOADED.
            # MUST be in finally — if mx.synchronize/clear_cache fails,
            # we still need to unblock waiters and free memory accounting.
            entry = self._entries.get(model_id)
            if entry is not None and entry.state == EngineState.UNLOADING:
                self._set_state(entry, EngineState.UNLOADED, "cleanup_complete")
                if entry.unload_complete is not None:
                    entry.unload_complete.set()

        logger.info(
            f"Unloaded model: {model_id}, "
            f"memory usage: {format_size(self._committed_memory())}"
        )

    # -------------------------------------------------------------------------
    # Engine loading
    # -------------------------------------------------------------------------

    async def _load_engine(self, model_id: str, force_lm: bool = False) -> None:
        """
        Load an engine for the specified model.

        Called outside the lock. The entry is in LOADING state, so no other
        coroutine will try to load the same model concurrently (INV-7).

        Args:
            model_id: The model ID to load
            force_lm: Force loading as BatchedEngine even for VLM models.

        Raises:
            ModelLoadingError: If model is already being loaded
        """
        entry = self._entries[model_id]

        effective_type = entry.engine_type
        if force_lm and effective_type == "vlm":
            effective_type = "batched"
            logger.info(f"Loading model as LM (force_lm=True): {model_id}")
        else:
            logger.info(f"Loading model: {model_id}")
        t0 = time.monotonic()
        entry.loading_started_at = t0
        # Snapshot pre-load memory so we can attribute the delta to this load.
        pre_load_memory = max(mx.get_active_memory(), get_phys_footprint())

        # Retrieve per-model settings for post-load transforms
        model_settings = None
        if self._settings_manager is not None:
            model_settings = self._settings_manager.get_settings(model_id)

        # Check if DFlash is enabled — takes priority over engine type
        # since DFlash has its own load pipeline (target + draft bundles).
        # Only LLM/VLM effective types qualify; embedding/reranker/audio
        # routes don't go through dflash-mlx.  Falls back silently to the
        # default engine if dflash-mlx isn't installed or load fails, so a
        # misconfigured setting can't take a model offline.
        engine = None
        if (
            model_settings is not None
            and effective_type in ("batched", "vlm")
        ):
            dflash_enabled = getattr(model_settings, "dflash_enabled", False)
            dflash_draft = getattr(model_settings, "dflash_draft_model", None)
            if dflash_enabled and dflash_draft:
                enforcer = self._process_memory_enforcer
                try:
                    from .engine.dflash import DFlashEngine
                    engine = DFlashEngine(
                        model_name=entry.model_path,
                        draft_model_path=dflash_draft,
                        draft_quant_bits=getattr(
                            model_settings, "dflash_draft_quant_bits", None
                        ),
                        model_settings=model_settings,
                        fallback_engine_type=(
                            "vlm" if effective_type == "vlm" else "batched"
                        ),
                        scheduler_config=self._scheduler_config,
                        process_memory_max_bytes=(
                            enforcer.get_final_ceiling() if enforcer else 0
                        ),
                    )
                    logger.info(
                        f"DFlash enabled for {model_id}, draft={dflash_draft}"
                    )
                except ImportError:
                    logger.warning(
                        f"DFlash enabled for {model_id} but dflash-mlx is "
                        f"not installed; falling back to default engine."
                    )
                except Exception as e:
                    logger.warning(
                        f"DFlash init failed for {model_id}: {e}; "
                        f"falling back to default engine."
                    )

        # Per-model trust_remote_code (security opt-in, issue #926).
        # When unset, defaults to False — repos with custom modeling_*.py
        # will fail to load until the user explicitly toggles this on
        # in the admin UI's model settings modal.
        trc = bool(getattr(model_settings, "trust_remote_code", False)) if model_settings else False

        # Create engine based on engine type (if DFlash didn't take it)
        if engine is None:
            if effective_type == "embedding":
                engine = EmbeddingEngine(
                    model_name=entry.model_path,
                    trust_remote_code=trc,
                )
            elif effective_type == "reranker":
                engine = RerankerEngine(
                    model_name=entry.model_path,
                    trust_remote_code=trc,
                )
            elif effective_type == "vlm":
                enforcer = self._process_memory_enforcer
                engine = VLMBatchedEngine(
                    model_name=entry.model_path,
                    trust_remote_code=trc,
                    scheduler_config=self._scheduler_config,
                    model_settings=model_settings,
                    process_memory_max_bytes=(
                        enforcer.get_final_ceiling() if enforcer else 0
                    ),
                )
            elif entry.engine_type == "audio_stt":
                engine = STTEngine(model_name=entry.model_path)
            elif entry.engine_type == "audio_tts":
                engine = TTSEngine(model_name=entry.model_path)
            elif entry.engine_type == "audio_sts":
                engine = STSEngine(
                    model_name=entry.model_path,
                    config_model_type=entry.config_model_type,
                )
            else:
                # BatchedEngine with continuous batching (default)
                engine = BatchedEngine(
                    model_name=entry.model_path,
                    trust_remote_code=trc,
                    scheduler_config=self._scheduler_config,
                    model_settings=model_settings,
                )

        _is_dflash_engine = engine is not None and type(engine).__name__ == "DFlashEngine"

        try:
            await engine.start()
        except Exception as start_error:
            if _is_dflash_engine:
                # DFlash engine failed to start — fall back to the
                # model's natural engine type (VLM or Batched)
                logger.warning(
                    f"DFlash start failed for {model_id}: {start_error}. "
                    f"Falling back to {effective_type} engine."
                )
                try:
                    await engine.stop()
                except Exception:
                    pass
                gc.collect()
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    get_mlx_executor(),
                    locked_sync_and_clear_cache,
                )

                if effective_type == "vlm":
                    enforcer = self._process_memory_enforcer
                    engine = VLMBatchedEngine(
                        model_name=entry.model_path,
                        trust_remote_code=trc,
                        scheduler_config=self._scheduler_config,
                        model_settings=model_settings,
                        process_memory_max_bytes=(
                            enforcer.get_final_ceiling() if enforcer else 0
                        ),
                    )
                else:
                    engine = BatchedEngine(
                        model_name=entry.model_path,
                        trust_remote_code=trc,
                        scheduler_config=self._scheduler_config,
                        model_settings=model_settings,
                    )
                try:
                    await engine.start()
                except Exception as fallback_error:
                    raise RuntimeError(
                        f"DFlash load failed: {start_error}. "
                        f"{effective_type} fallback also failed: "
                        f"{fallback_error}"
                    ) from start_error
                logger.info(
                    f"Successfully loaded {model_id} as {effective_type} "
                    f"(fallback from DFlash)"
                )

            elif force_lm and entry.engine_type == "vlm":
                # force_lm created a BatchedEngine but mlx-lm can't
                # load this VLM model — fall back to VLMBatchedEngine.
                logger.warning(
                    f"LM loading failed for VLM model {model_id} "
                    f"(force_lm=True), falling back to VLM engine: "
                    f"{start_error}"
                )
                try:
                    await engine.stop()
                except Exception:
                    pass
                gc.collect()
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    get_mlx_executor(),
                    locked_sync_and_clear_cache,
                )

                enforcer = self._process_memory_enforcer
                engine = VLMBatchedEngine(
                    model_name=entry.model_path,
                    trust_remote_code=trc,
                    scheduler_config=self._scheduler_config,
                    model_settings=model_settings,
                    process_memory_max_bytes=(
                        enforcer.get_final_ceiling() if enforcer else 0
                    ),
                )
                try:
                    await engine.start()
                except Exception as fallback_error:
                    raise RuntimeError(
                        f"LM load failed (force_lm=True): {start_error}. "
                        f"VLM fallback also failed: {fallback_error}"
                    ) from start_error

                logger.info(
                    f"Successfully loaded {model_id} as VLM "
                    f"(fallback from force_lm)"
                )
            elif entry.engine_type == "vlm":
                # VLM loading failed — fall back to LLM (BatchedEngine)
                logger.warning(
                    f"VLM loading failed for {model_id}, "
                    f"falling back to LLM: {start_error}"
                )
                try:
                    await engine.stop()
                except Exception:
                    pass
                gc.collect()
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    get_mlx_executor(),
                    locked_sync_and_clear_cache,
                )

                engine = BatchedEngine(
                    model_name=entry.model_path,
                    trust_remote_code=trc,
                    scheduler_config=self._scheduler_config,
                    model_settings=model_settings,
                )
                try:
                    await engine.start()
                except Exception as fallback_error:
                    raise RuntimeError(
                        f"VLM load failed: {start_error}. "
                        f"LLM fallback also failed: {fallback_error}"
                    ) from start_error

                entry.model_type = "llm"
                entry.engine_type = "batched"
                logger.info(
                    f"Successfully loaded {model_id} as LLM "
                    f"(fallback from VLM)"
                )
            else:
                raise

        # Check if memory enforcer requested abort during loading
        if entry.abort_loading:
            logger.warning(
                f"Model load aborted by memory enforcer: {model_id}"
            )
            try:
                await engine.stop()
            except Exception as e:
                logger.warning(
                    f"Error stopping aborted engine for {model_id}: {e}"
                )
            gc.collect()
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                get_mlx_executor(),
                locked_sync_and_clear_cache,
            )
            raise ModelLoadingError(
                f"Model {model_id} load aborted: "
                f"process memory limit exceeded",
                model_id=model_id,
            )

        entry.engine = engine
        entry.last_access = time.time()
        # Back-reference so non-streaming engines can self-reserve working set
        # against this pool inside their heavy methods (§3d). Engines guard
        # every use (``if self._pool is not None``) so non-pooled / unit-test
        # use of an engine never touches the pool.
        engine._pool = self
        # Stamp the canonical ``_entries`` key alongside the back-reference.
        # Engines are constructed with ``model_name = entry.model_path`` (the
        # full PATH) while ``_entries`` is keyed by the short ``model_id``, so
        # an engine-side ``pool._entries.get(self.model_name)`` always misses
        # — which silently zeroed the §3d weight-fraction working-set
        # estimates for embedding/reranker/STT/STS (their reservations were
        # strict no-ops in pooled production). This is the single chokepoint
        # every engine type passes through, and the same mismatch class
        # _resolve_reservation_key patches on the reserve side.
        engine._pool_model_id = model_id

        # Propagate memory limit to new engine's scheduler
        if self._process_memory_enforcer is not None:
            self._process_memory_enforcer._propagate_memory_limit()

        # Release intermediate Metal buffers from model loading.
        # mlx_lm.load() creates large temporaries (weight transforms,
        # quantization intermediates) that stay in the Metal buffer pool
        # because mx.set_cache_limit(total_mem) prevents automatic release.
        # Without this, memory stays at ~2x model size until the first
        # inference request triggers a clear. (#429)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            get_mlx_executor(),
            locked_sync_and_clear_cache,
        )

        elapsed = time.monotonic() - t0
        # Capture observed phys_footprint delta to populate EngineEntry.actual_size
        # and feed the load-speed EMA the admin UI consumes for load-time ETAs.
        post_load_memory = max(mx.get_active_memory(), get_phys_footprint())
        observed_delta = max(0, post_load_memory - pre_load_memory)
        entry.actual_size = observed_delta or entry.estimated_size
        entry.loading_started_at = None
        size_gb = entry.estimated_size / (1024 ** 3)
        if size_gb > 0 and elapsed > 0:
            sample = elapsed / size_gb
            if self._load_seconds_per_gb_ema is None:
                self._load_seconds_per_gb_ema = sample
            else:
                # 0.1 weight on each new sample — same EMA shape main uses.
                self._load_seconds_per_gb_ema = (
                    self._load_seconds_per_gb_ema * 0.9 + sample * 0.1
                )
            self._load_time_observations += 1
        logger.info(
            f"Loaded model: {model_id} in {elapsed:.1f}s "
            f"(actual: {format_size(entry.actual_size)}, "
            f"estimated: {format_size(entry.estimated_size)}, "
            f"total: {format_size(self._committed_memory())})"
        )

    # -------------------------------------------------------------------------
    # Lifecycle methods
    # -------------------------------------------------------------------------

    async def preload_pinned_models(self) -> None:
        """
        Preload all pinned models at startup.

        This ensures pinned models are always available.
        """
        pinned_models = [
            model_id for model_id, e in self._entries.items() if e.is_pinned
        ]

        for model_id in pinned_models:
            try:
                logger.info(f"Preloading pinned model: {model_id}")
                await self.get_engine(model_id)
            except Exception as e:
                logger.error(f"Failed to preload pinned model {model_id}: {e}")

    async def shutdown(self) -> None:
        """Shutdown all engines gracefully.

        Holds the lock for the entire shutdown to prevent new loads from
        starting while we're tearing down.  This is acceptable because
        shutdown is a terminal operation — no new requests should arrive.
        """
        # §P6: stop the budget corrector first (it takes the same lock).
        task = self._budget_corrector_task
        if task is not None and not task.done():
            task.cancel()
        async with self._tracked_lock("shutdown"):
            for model_id in list(self._entries.keys()):
                entry = self._entries.get(model_id)
                if entry is None:
                    continue

                if entry.engine is not None:
                    try:
                        await self._unload_engine(
                            model_id, reason="shutdown"
                        )
                    except Exception as e:
                        logger.error(
                            f"Error unloading {model_id} during shutdown: {e}"
                        )

                # Force state to UNLOADED for any remaining states
                # (e.g., LOADING with no engine yet)
                if entry.state != EngineState.UNLOADED:
                    self._set_state(entry, EngineState.UNLOADED, "shutdown")

                # Signal all waiters so they unblock
                if entry.drain_complete and not entry.drain_complete.is_set():
                    entry.drain_complete.set()
                if entry.unload_complete and not entry.unload_complete.is_set():
                    entry.unload_complete.set()
                if entry.ready_event and not entry.ready_event.is_set():
                    entry.ready_event.set()

        # Wait for deferred cleanup tasks (engine.stop + mx.clear_cache)
        # outside the lock so they can actually run.
        if self._cleanup_tasks:
            logger.info(
                f"Waiting for {len(self._cleanup_tasks)} engine cleanup tasks..."
            )
            await asyncio.gather(*self._cleanup_tasks, return_exceptions=True)
            self._cleanup_tasks.clear()

        logger.info("Engine pool shutdown complete")

    def _model_status_entry(self, mid: str, e: EngineEntry) -> dict:
        """Build a single model entry for get_status()."""
        info: dict[str, Any] = {
            "id": mid,
            "model_path": e.model_path,
            "loaded": e.engine is not None,
            "state": e.state.value,
            "is_loading": e.state == EngineState.LOADING,
            "estimated_size": e.estimated_size,
            "pinned": e.is_pinned,
            "engine_type": e.engine_type,
            "model_type": e.model_type,
            "config_model_type": e.config_model_type,
            "thinking_default": e.thinking_default,
            "preserve_thinking_default": e.preserve_thinking_default,
            "last_access": e.last_access if e.last_access > 0 else None,
            "exclusive": e.exclusive,
            "exclusive_max_hold": e.exclusive_max_hold,
            "active_uses": e.active_uses,
            "exclusive_idle_pending": e.exclusive_idle is not None and not e.exclusive_idle.is_set() if e.exclusive_idle is not None else False,
        }
        if e.model_type == "vlm":
            if e._vision_limits_cache is None:
                e._vision_limits_cache = self.compute_vision_limits(e)
            if e._vision_limits_cache:
                info["vision_limits"] = e._vision_limits_cache
        return info

    def get_status(self) -> dict:
        """
        Get pool status for monitoring endpoints.

        Returns:
            Dictionary with pool status information
        """
        return {
            "max_model_memory": self._current_ceiling() or None,
            "current_model_memory": self._committed_memory(),
            "model_count": len(self._entries),
            "loaded_count": sum(
                1 for e in self._entries.values() if e.engine is not None
            ),
            "models": [
                self._model_status_entry(mid, e)
                for mid, e in sorted(self._entries.items())
            ],
        }

    def get_debug_status(self) -> dict:
        """Get detailed pool status for the /debug/pool diagnostic endpoint.

        Lock-free read-only snapshot — fast and safe to call at any time.
        Only includes non-UNLOADED models to keep the response small.
        """
        now = time.time()
        models = {}
        for mid, e in self._entries.items():
            if e.state == EngineState.UNLOADED:
                continue
            info: dict[str, Any] = {
                "state": e.state.value,
                "loaded": e.engine is not None,
            }
            if e.state == EngineState.LOADING:
                info["load_elapsed_s"] = round(now - e.load_started, 1)
            if e.state == EngineState.DRAINING:
                info["drain_elapsed_s"] = round(now - e.drain_started, 1)
            if e.state == EngineState.ACTIVE:
                info["active_requests"] = (
                    e.engine.has_active_requests()
                    if e.engine
                    else False
                )
            info["exclusive"] = e.exclusive
            info["exclusive_max_hold"] = e.exclusive_max_hold
            info["active_uses"] = e.active_uses
            info["exclusive_idle_pending"] = (
                e.exclusive_idle is not None and not e.exclusive_idle.is_set()
                if e.exclusive_idle is not None
                else False
            )
            models[mid] = info

        unloaded_count = sum(
            1 for e in self._entries.values()
            if e.state == EngineState.UNLOADED
        )

        return {
            "lock_held": self._lock.locked(),
            "committed_memory_gb": round(self._committed_memory() / 1e9, 2),
            "inflight_reservations_gb": round(
                self._inflight_reservations / 1e9, 2
            ),
            "max_model_memory_gb": (
                round(self._current_ceiling() / 1e9, 2)
                if self._current_ceiling()
                else None
            ),
            "timeout_counter": self._timeout_counter,
            "active_models": models,
            "unloaded_count": unloaded_count,
        }

    def get_crash_diagnostic_snapshot(self) -> dict[str, Any]:
        """Lock-free snapshot for native crash dumps (e.g. SIGABRT from MLX/Metal).

        Includes :meth:`get_debug_status` plus every discovered model (including
        UNLOADED) with ``active_uses`` and pinning for eviction analysis.
        """
        return {
            "debug_pool": self.get_debug_status(),
            "models": {
                mid: {
                    "state": e.state.value,
                    "engine_loaded": e.engine is not None,
                    "active_uses": e.active_uses,
                    "is_pinned": e.is_pinned,
                    "exclusive": e.exclusive,
                    "engine_type": e.engine_type,
                    "model_type": e.model_type,
                    "estimated_size_bytes": e.estimated_size,
                }
                for mid, e in sorted(self._entries.items())
            },
            "pending_cleanup_tasks": len(self._cleanup_tasks),
        }

    async def check_ttl_expirations(
        self,
        settings_manager: ModelSettingsManager,
        global_idle_timeout_seconds: int | None = None,
    ) -> list[str]:
        """Check and unload models that have exceeded their TTL.

        Pinned models are skipped (TTL is ignored for pinned models).
        Models with active requests are drained (not force-killed).
        Models in LOADING or DRAINING states are skipped.
        Suppressed during benchmark runs via _suppress_ttl flag.

        Args:
            settings_manager: The settings manager to read TTL values from.
            global_idle_timeout_seconds: Global idle timeout fallback (None = no global TTL).

        Returns:
            List of model IDs that were unloaded or started draining.
        """
        if self._suppress_ttl:
            return []

        now = time.time()
        expired: list[str] = []

        async with self._tracked_lock("check_ttl"):
            for model_id, entry in self._entries.items():
                if entry.engine is None or entry.is_pinned:
                    continue
                if entry.state not in (EngineState.ACTIVE,):
                    continue  # Skip LOADING, DRAINING, UNLOADED

                settings = settings_manager.get_settings(model_id)
                effective_ttl = settings.ttl_seconds
                if effective_ttl is None:
                    effective_ttl = global_idle_timeout_seconds
                if effective_ttl is None:
                    continue

                idle_time = now - entry.last_access
                if idle_time < effective_ttl:
                    continue

                # Check if model has active requests (works for all engine types)
                if entry.engine.has_active_requests():
                    # TTL expired + active → start drain instead of refresh
                    logger.info(
                        f"TTL expired for model '{model_id}' with active "
                        f"requests (idle {idle_time:.0f}s > "
                        f"ttl {settings.ttl_seconds}s), starting drain"
                    )
                    self._start_drain(model_id)
                    expired.append(model_id)
                    continue

                logger.info(
                    f"TTL expired for model '{model_id}' "
                    f"(idle {idle_time:.0f}s > ttl {effective_ttl}s)"
                )
                await self._unload_engine(model_id, reason="ttl_expired")
                expired.append(model_id)

        return expired
