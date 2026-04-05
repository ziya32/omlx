# SPDX-License-Identifier: Apache-2.0
"""
Process-level memory enforcer for oMLX.

Monitors total Metal memory usage via mx.get_active_memory() and enforces
the max_process_memory limit by unloading LRU models from EnginePool.

The enforcer runs as a background asyncio task that polls memory usage at
a configurable interval (default: 1 second). When usage exceeds the limit,
it immediately unloads the least-recently-used non-pinned model. If the
model is mid-inference, the inference is aborted as part of engine shutdown.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

import mlx.core as mx

from .engine_pool import EngineState

if TYPE_CHECKING:
    from .engine_pool import EnginePool
    from .model_settings import ModelSettingsManager

logger = logging.getLogger(__name__)


def _format_gb(b: int) -> str:
    """Format bytes as GB string."""
    return f"{b / 1024**3:.1f}GB"


class ProcessMemoryEnforcer:
    """
    Background task that enforces process-level memory limits.

    Polls mx.get_active_memory() every poll_interval seconds and unloads
    LRU models from EnginePool when the limit is exceeded.
    """

    def __init__(
        self,
        engine_pool: EnginePool,
        max_bytes: int,
        poll_interval: float = 1.0,
        settings_manager: ModelSettingsManager | None = None,
        prefill_memory_guard: bool = True,
    ):
        """
        Initialize the process memory enforcer.

        Args:
            engine_pool: The engine pool to evict models from.
            max_bytes: Maximum allowed Metal memory in bytes.
            poll_interval: Seconds between memory checks.
            settings_manager: Optional settings manager for TTL checks.
            prefill_memory_guard: Whether to enable pre-flight memory
                estimation to reject requests that would exceed limits.
        """
        self._engine_pool = engine_pool
        self._max_bytes = max_bytes
        self._poll_interval = poll_interval
        self._settings_manager = settings_manager
        self._prefill_memory_guard = prefill_memory_guard
        self._task: asyncio.Task | None = None
        self._running = False
        # Diagnostic counters (readable via get_status / debug endpoints)
        self._peak_memory_bytes: int = 0
        self._overage_count: int = 0
        self._last_overage_log: float = 0.0  # monotonic timestamp
        self._last_no_candidate_log: float = 0.0

    @property
    def max_bytes(self) -> int:
        """Maximum allowed Metal memory in bytes."""
        return self._max_bytes

    @max_bytes.setter
    def max_bytes(self, value: int) -> None:
        old = self._max_bytes
        self._max_bytes = value
        if self._running:
            self._propagate_memory_limit()
            self._set_metal_memory_limit()
        logger.info(
            f"Process memory limit changed: "
            f"{_format_gb(old)} -> {_format_gb(value)}"
        )

    @property
    def is_running(self) -> bool:
        """Whether the enforcement loop is active."""
        return self._running

    def start(self) -> None:
        """Start the background enforcement loop."""
        if self._running:
            return
        self._running = True
        self._propagate_memory_limit()
        self._set_metal_memory_limit()
        self._task = asyncio.create_task(self._enforcement_loop())
        logger.info(
            f"Process memory enforcer started "
            f"(limit: {_format_gb(self._max_bytes)}, "
            f"interval: {self._poll_interval}s)"
        )

    def _get_hard_limit_bytes(self) -> int:
        """Hard limit for inline prefill check: system_ram - 4GB.

        Returns 0 if enforcement is disabled (max_bytes <= 0).
        Always >= max_bytes so prefill gets headroom above the soft limit.
        """
        if self._max_bytes <= 0:
            return 0
        from .settings import get_system_memory

        return max(get_system_memory() - 4 * 1024**3, self._max_bytes)

    def _set_metal_memory_limit(self) -> None:
        """No-op. Metal-level limits removed to prevent model load swap.

        mx.set_memory_limit() causes MLX to aggressively reclaim cached
        buffers during model loading, creating alloc/free churn that
        pushes the system into swap. All memory enforcement is handled
        by mx.get_active_memory() polling instead. (#429)
        """
        pass

    def _clear_metal_memory_limit(self) -> None:
        """No-op. See _set_metal_memory_limit."""
        pass

    @property
    def prefill_memory_guard(self) -> bool:
        """Whether prefill memory guard is enabled."""
        return self._prefill_memory_guard

    @prefill_memory_guard.setter
    def prefill_memory_guard(self, value: bool) -> None:
        self._prefill_memory_guard = value
        if self._running:
            self._propagate_memory_limit()
            if value:
                self._set_metal_memory_limit()
            else:
                self._clear_metal_memory_limit()
        logger.info(f"Prefill memory guard: {'enabled' if value else 'disabled'}")

    def _propagate_memory_limit(self) -> None:
        """Propagate soft/hard memory limits to schedulers for inline prefill checking."""
        hard_limit = self._get_hard_limit_bytes()
        for entry in self._engine_pool._entries.values():
            if entry.engine is not None:
                scheduler = getattr(entry.engine, "scheduler", None)
                if scheduler is not None:
                    scheduler._memory_limit_bytes = self._max_bytes
                    scheduler._memory_hard_limit_bytes = hard_limit
                    scheduler._prefill_memory_guard = self._prefill_memory_guard
                    bg = getattr(scheduler, "batch_generator", None)
                    if bg is not None and hasattr(bg, "_memory_limit_bytes"):
                        bg._memory_limit_bytes = self._max_bytes
                        bg._memory_hard_limit_bytes = hard_limit

    async def stop(self) -> None:
        """Stop the background enforcement loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Process memory enforcer stopped")

    async def _enforcement_loop(self) -> None:
        """Main polling loop."""
        while self._running:
            try:
                await self._check_and_enforce()
                await self._check_ttl()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Process memory enforcer error: {e}")
            await asyncio.sleep(self._poll_interval)

    async def _check_ttl(self) -> None:
        """Check and unload models that exceeded their TTL."""
        if self._settings_manager is None:
            return
        await self._engine_pool.check_ttl_expirations(self._settings_manager)

    async def _check_and_enforce(self) -> None:
        """Check current memory and enforce limit if exceeded.

        When Metal memory exceeds the soft process limit, abort all
        active requests on the LRU non-pinned model and unload it.
        Pinned models are never evicted.

        Eviction is one-victim-per-tick: Metal memory reclamation is
        asynchronous (EnginePool._unload_engine only clears the engine
        reference and schedules _deferred_engine_cleanup; the heavy
        mx.synchronize() + mx.clear_cache() runs on the MLX executor
        after we release the lock), so re-checking mx.get_active_memory()
        inside a loop would cascade-evict every non-pinned model in a
        single tick. If memory is still over the limit after this
        unload, the next _enforcement_loop iteration will re-check and
        pick another victim. See docs/enforcer-eviction-review.md #1.

        Eviction is immediate and does not wait for drain — clients
        receive errors via abort_all_requests() rather than a silent
        disconnect. A victim's active_uses count does NOT protect it:
        when process memory is exhausted, aborting in-flight handlers
        is preferable to OOM.
        """
        if self._max_bytes <= 0:
            return

        current = mx.get_active_memory()
        if current > self._peak_memory_bytes:
            self._peak_memory_bytes = current
        if current <= self._max_bytes:
            return

        self._overage_count += 1
        overage = current - self._max_bytes
        now = time.monotonic()
        if now - self._last_overage_log >= 5.0:
            self._last_overage_log = now
            logger.warning(
                f"Process memory limit exceeded: "
                f"{_format_gb(current)} / {_format_gb(self._max_bytes)} "
                f"(over by {_format_gb(overage)})"
            )

        # Acquire EnginePool lock and evict one non-pinned LRU victim.
        # Active requests are aborted before unload so clients see a
        # proper error — EngineCore.stop() only cancels the engine loop
        # silently without notifying collectors. Note: prefill loops
        # self-check via _memory_limit_bytes (same thread, no GIL
        # issue), so they abort independently of this enforcer.
        async with self._engine_pool._tracked_lock("process_memory_enforcer"):
            victim = self._engine_pool._find_drain_or_evict_candidate()
            if victim is not None:
                entry = self._engine_pool._entries.get(victim)
                # Defensive: _find_drain_or_evict_candidate already
                # filters engine=None, and we hold the pool lock, so
                # this should not trigger. If it does, fall through to
                # the no-candidate diagnostic path below.
                if entry is not None and entry.engine is not None:
                    # Every engine (BatchedEngine/VLMBatchedEngine via
                    # EngineCore, BaseNonStreamingEngine subclasses via
                    # their cooperative abort flag) implements
                    # abort_all_requests, so we no longer special-case
                    # non-abortable engines.
                    aborted = await entry.engine.abort_all_requests()
                    if aborted > 0:
                        logger.warning(
                            f"Aborted {aborted} requests on "
                            f"'{victim}' before eviction"
                        )
                    logger.warning(
                        f"Evicting non-pinned model '{victim}' to enforce "
                        f"process memory limit (active_uses="
                        f"{entry.active_uses})"
                    )
                    await self._engine_pool._unload_engine(victim)
                    return

            # No eviction candidate — throttle this diagnostic to
            # once per 5 seconds (it can repeat every poll cycle).
            now = time.monotonic()
            if now - self._last_no_candidate_log < 5.0:
                return

            self._last_no_candidate_log = now

            loading = [
                e.model_id
                for e in self._engine_pool._entries.values()
                if e.is_loading
            ]
            if loading:
                logger.warning(
                    f"Memory limit exceeded while loading "
                    f"{loading} — waiting for load to complete "
                    f"and deferred cleanup to free Metal memory."
                )
                return

            draining = [
                e.model_id
                for e in self._engine_pool._entries.values()
                if e.state == EngineState.DRAINING
            ]
            pinned = [
                e.model_id
                for e in self._engine_pool._entries.values()
                if e.engine is not None and e.is_pinned
            ]
            if draining:
                logger.warning(
                    f"Memory limit exceeded while draining "
                    f"{draining} — waiting for active requests "
                    f"to finish."
                )
            elif pinned:
                logger.warning(
                    f"Memory limit exceeded but all loaded "
                    f"models are pinned ({pinned}) — cannot evict."
                )
            else:
                snapshot = self._engine_pool.get_crash_diagnostic_snapshot()
                logger.warning(
                    "🚨 Memory limit exceeded but no models are loaded to evict "
                    "(metal=%s, limit=%s, snapshot=%s)",
                    _format_gb(mx.get_active_memory()),
                    _format_gb(self._max_bytes),
                    snapshot,
                )

    def get_status(self) -> dict:
        """Get enforcer status for monitoring endpoints."""
        current = mx.get_active_memory() if self._running else 0
        return {
            "enabled": self._running,
            "max_bytes": self._max_bytes,
            "max_formatted": _format_gb(self._max_bytes),
            "current_bytes": current,
            "current_formatted": _format_gb(current),
            "utilization": (
                current / self._max_bytes if self._max_bytes > 0 else 0.0
            ),
            "peak_memory_bytes": self._peak_memory_bytes,
            "peak_memory_formatted": _format_gb(self._peak_memory_bytes),
            "overage_count": self._overage_count,
        }

    def reset_peak(self) -> None:
        """Reset peak memory and overage counter (for tests)."""
        self._peak_memory_bytes = 0
        self._overage_count = 0
