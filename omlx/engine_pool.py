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
    EnginePoolError,
    InsufficientMemoryError,
    ModelLoadingError,
    ModelNotFoundError,
    ModelTooLargeError,
)
from .model_discovery import discover_models, format_size
from .engine_core import get_mlx_executor
from .scheduler import SchedulerConfig

logger = logging.getLogger(__name__)

# Cooldown after a load failure before retrying (seconds)
LOAD_COOLDOWN = 30


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
    engine: BaseEngine | EmbeddingEngine | RerankerEngine | STTEngine | STSEngine | TTSEngine | None = None  # Loaded engine instance
    last_access: float = 0.0  # Timestamp for LRU (0 if never loaded)
    is_pinned: bool = False  # Never evict if True
    abort_loading: bool = False  # Deprecated: kept for backward compat only
    _vision_limits_cache: dict | None = None  # Cached compute_vision_limits result

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
        max_model_memory: int | None,
        scheduler_config: SchedulerConfig | None = None,
        drain_timeout: float = 120.0,
        max_wait_timeout: float = 300.0,
    ):
        """
        Initialize the engine pool.

        Args:
            max_model_memory: Maximum memory for loaded models in bytes,
                or None for no limit (disabled)
            scheduler_config: Configuration for BatchedEngine schedulers
            drain_timeout: Seconds before force-aborting a draining model
            max_wait_timeout: Seconds before timing out a get_engine() wait
        """
        self._entries: dict[str, EngineEntry] = {}
        self._lock = asyncio.Lock()
        self._max_model_memory = max_model_memory
        self._scheduler_config = scheduler_config or SchedulerConfig()
        self._process_memory_enforcer: object | None = None  # Set by server
        self._settings_manager: object | None = None  # Set by server
        self._suppress_ttl: bool = False  # Suppress TTL during benchmarks
        self._drain_timeout = drain_timeout
        self._max_wait_timeout = max_wait_timeout
        # Incremented on ANY timeout firing. Tests assert this stays at 0.
        self._timeout_counter: int = 0
        # Track deferred engine cleanup tasks so shutdown can wait for them
        self._cleanup_tasks: list[asyncio.Task] = []

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

    @property
    def max_model_memory(self) -> int | None:
        """Maximum memory for loaded models in bytes, or None if disabled."""
        return self._max_model_memory

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
        max_bytes = getattr(enforcer, "max_bytes", 0) if enforcer else 0
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

        mem_display = "disabled" if self._max_model_memory is None else format_size(self._max_model_memory)
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
                        return entry.engine

                elif entry.state == EngineState.LOADING:
                    # Someone else is loading this model — wait for them
                    event = entry.ready_event

                elif entry.state == EngineState.DRAINING:
                    # Model is being unloaded — wait for drain, then it will
                    # need to be reloaded (or another model will free space)
                    event = entry.drain_complete

                elif entry.state == EngineState.UNLOADING:
                    # Metal cleanup in progress — wait for it to finish,
                    # then the model will need to be reloaded.
                    event = entry.unload_complete

                elif entry.state == EngineState.UNLOADED:
                    # Check if model is too large for memory limit
                    if (
                        self._max_model_memory is not None
                        and entry.estimated_size > self._max_model_memory
                    ):
                        raise ModelTooLargeError(
                            model_id, entry.estimated_size, self._max_model_memory
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
                    await loop.run_in_executor(get_mlx_executor(), mx.clear_cache)

                # Load the model (long operation, outside lock)
                load_error = None
                try:
                    await self._load_engine(model_id, force_lm=force_lm)
                except BaseException as e:
                    load_error = e

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
                        if not isinstance(load_error, asyncio.CancelledError):
                            entry.load_error = load_error  # Set under lock
                        entry.load_failed_at = time.time()
                        self._set_state(
                            entry, EngineState.UNLOADED, "load_failed"
                        )
                        result = None
                    entry.ready_event.set()  # Signal under lock — ALWAYS

                if load_error is not None:
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
                    await asyncio.wait_for(
                        event.wait(), timeout=self._max_wait_timeout
                    )
                    logger.debug(
                        f"get_engine({model_id}) woke from {wait_target}, "
                        f"re-entering loop (iteration={iterations})"
                    )
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
    # Engine use-counting (prevents eviction while request handlers use engine)
    #
    # Every server endpoint that holds a reference to an engine MUST bracket
    # its use with acquire_engine / release_engine (or use_engine context
    # manager).  This increments active_uses on the EngineEntry, which is
    # checked by:
    #   - _prepare_memory_for   → drains instead of killing busy engines
    #   - _drain_monitor        → waits for active_uses==0 before unloading,
    #                             and extends the drain if timeout is hit
    #                             while active_uses > 0
    #   - process_memory_enforcer → skips victims with active_uses > 0
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

    def release_engine(self, model_id: str) -> None:
        """Decrement active_uses for model_id.

        Called when a request handler finishes using the engine.
        """
        entry = self._entries.get(model_id)
        if entry is not None and entry.active_uses > 0:
            entry.active_uses -= 1
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

    @asynccontextmanager
    async def use_engine(self, model_id: str):
        """Context manager: get_engine + acquire/release use-counting.

        Usage:
            async with pool.use_engine(model_id) as engine:
                await engine.transcribe(...)

        The engine is protected from eviction for the duration of the
        context manager.
        """
        engine = await self.get_engine(model_id)
        self.acquire_engine(model_id)
        try:
            yield engine
        finally:
            self.release_engine(model_id)

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
        max_bytes = getattr(enforcer, 'max_bytes', 0) if enforcer else 0
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
                lambda: (mx.synchronize(), mx.clear_cache()),
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
                    if e.active_uses > 0 and e.exclusive_idle is not None:
                        logger.info(
                            f"Deferring load of '{entry.model_id}': "
                            f"exclusive model '{mid}' has "
                            f"{e.active_uses} active request(s)"
                        )
                        return e.exclusive_idle
                    # Log WHY we didn't defer
                    if e.active_uses == 0:
                        logger.debug(
                            f"_prepare_memory_for('{entry.model_id}') "
                            f"NOT deferring: exclusive '{mid}' has "
                            f"active_uses=0 (no active requests)"
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

        if self._max_model_memory is None:
            # No model memory limit — also check process memory
            return await self._check_process_memory(entry)

        required = entry.estimated_size
        if entry.model_type not in ("audio_stt", "audio_tts", "audio_sts", "embedding", "reranker"):
            required += int(entry.estimated_size * 0.25)  # KV headroom

        while self._committed_memory() + required > self._max_model_memory:
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
                # Try without KV headroom as a last resort
                required_no_headroom = entry.estimated_size
                if self._committed_memory() + required_no_headroom <= self._max_model_memory:
                    logger.info(
                        f"Loading {entry.model_id} without KV headroom "
                        f"(need {format_size(required)}, "
                        f"available {format_size(self._max_model_memory - self._committed_memory())})"
                    )
                    break  # Proceed without headroom

                # Truly stuck (all pinned)
                raise ModelTooLargeError(
                    entry.model_id, required, self._max_model_memory
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
        if enforcer.max_bytes <= 0:
            return None

        while True:
            current_active = mx.get_active_memory()
            projected = current_active + entry.estimated_size
            if projected <= enforcer.max_bytes:
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
                        f"{format_size(enforcer.max_bytes)})"
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
            current_active = mx.get_active_memory()
            projected = current_active + entry.estimated_size
            if projected <= enforcer.max_bytes:
                logger.info(
                    f"Process memory after cleanup: "
                    f"{format_size(current_active)} + "
                    f"{entry.model_id} ({format_size(entry.estimated_size)}) "
                    f"= {format_size(projected)} <= "
                    f"{format_size(enforcer.max_bytes)}. Proceeding."
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
            committed = self._committed_memory()
            headroom = int(enforcer.max_bytes * 0.25)
            if (
                committed + entry.estimated_size <= enforcer.max_bytes
                and projected <= enforcer.max_bytes + headroom
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

            # Truly cannot fit — even with cleanup done, Metal retains too
            # much memory for the new model to load safely.
            raise InsufficientMemoryError(
                required=entry.estimated_size,
                current=current_active,
                message=(
                    f"Cannot load {entry.model_id}: projected memory "
                    f"{format_size(projected)} would exceed process "
                    f"limit {format_size(enforcer.max_bytes)} "
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

    def _find_drain_or_evict_candidate(self) -> str | None:
        """
        Find the least recently used non-pinned loaded model suitable for
        eviction or draining. Skips models already in DRAINING or UNLOADING state.

        Prefers idle models over models with active requests or active
        use-count (request handlers that have acquired the engine via
        acquire_engine/use_engine).

        Returns:
            Model ID of the candidate, or None if no evictable models exist.
        """
        candidates = []
        for mid, e in self._entries.items():
            if e.engine is None or e.is_pinned:
                continue
            if e.state == EngineState.DRAINING:
                continue  # Already being drained
            if e.state == EngineState.LOADING:
                continue  # Don't evict something being loaded
            if e.state == EngineState.UNLOADING:
                continue  # Metal cleanup in progress
            has_active = (
                e.engine.has_active_requests() or e.active_uses > 0
            )
            candidates.append((has_active, e.last_access, mid))
        if not candidates:
            return None
        candidates.sort()  # (False, old_time) sorts before (True, old_time)
        return candidates[0][2]

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

        # Force garbage collection to release memory.
        # Run mx.clear_cache on the global MLX executor to avoid concurrent
        # Metal operations with running engines. See issue #85.
        # Synchronize before clearing to prevent releasing Metal buffers
        # still referenced by in-flight command buffers. See issue #300.
        try:
            gc.collect()
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                get_mlx_executor(), lambda: (mx.synchronize(), mx.clear_cache())
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

        # Retrieve per-model settings for post-load transforms
        model_settings = None
        if self._settings_manager is not None:
            model_settings = self._settings_manager.get_settings(model_id)

        # Create engine based on engine type
        if effective_type == "embedding":
            engine = EmbeddingEngine(model_name=entry.model_path)
        elif effective_type == "reranker":
            engine = RerankerEngine(model_name=entry.model_path)
        elif effective_type == "llm_reranker":
            engine = LLMRerankerEngine(
                model_name=entry.model_path,
                scheduler_config=self._scheduler_config,
            )
        elif effective_type == "vlm":
            enforcer = self._process_memory_enforcer
            engine = VLMBatchedEngine(
                model_name=entry.model_path,
                scheduler_config=self._scheduler_config,
                model_settings=model_settings,
                process_memory_max_bytes=(
                    getattr(enforcer, "max_bytes", 0) if enforcer else 0
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
                scheduler_config=self._scheduler_config,
                model_settings=model_settings,
            )

        try:
            await engine.start()
        except Exception as start_error:
            if force_lm and entry.engine_type == "vlm":
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
                    lambda: (mx.synchronize(), mx.clear_cache()),
                )

                engine = VLMBatchedEngine(
                    model_name=entry.model_path,
                    scheduler_config=self._scheduler_config,
                    model_settings=model_settings,
                )
                await engine.start()

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
                    lambda: (mx.synchronize(), mx.clear_cache()),
                )

                engine = BatchedEngine(
                    model_name=entry.model_path,
                    scheduler_config=self._scheduler_config,
                    model_settings=model_settings,
                )
                await engine.start()

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
                lambda: (mx.synchronize(), mx.clear_cache()),
            )
            raise ModelLoadingError(
                f"Model {model_id} load aborted: "
                f"process memory limit exceeded"
            )

        entry.engine = engine
        entry.last_access = time.time()

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
            lambda: (mx.synchronize(), mx.clear_cache()),
        )

        elapsed = time.monotonic() - t0
        logger.info(
            f"Loaded model: {model_id} in {elapsed:.1f}s "
            f"(estimated: {format_size(entry.estimated_size)}, "
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
            "loaded": e.engine is not None,
            "state": e.state.value,
            "is_loading": e.state == EngineState.LOADING,
            "estimated_size": e.estimated_size,
            "pinned": e.is_pinned,
            "engine_type": e.engine_type,
            "model_type": e.model_type,
            "config_model_type": e.config_model_type,
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
            "max_model_memory": self._max_model_memory,
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
            "max_model_memory_gb": (
                round(self._max_model_memory / 1e9, 2)
                if self._max_model_memory
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
        self, settings_manager: ModelSettingsManager
    ) -> list[str]:
        """Check and unload models that have exceeded their TTL.

        Pinned models are skipped (TTL is ignored for pinned models).
        Models with active requests are drained (not force-killed).
        Models in LOADING or DRAINING states are skipped.
        Suppressed during benchmark runs via _suppress_ttl flag.

        Args:
            settings_manager: The settings manager to read TTL values from.

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
                if settings.ttl_seconds is None:
                    continue

                idle_time = now - entry.last_access
                if idle_time < settings.ttl_seconds:
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
                    f"(idle {idle_time:.0f}s > ttl {settings.ttl_seconds}s)"
                )
                await self._unload_engine(model_id, reason="ttl_expired")
                expired.append(model_id)

        return expired
