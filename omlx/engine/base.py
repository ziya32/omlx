# SPDX-License-Identifier: Apache-2.0
"""
Base engine interface for oMLX inference.
"""

import asyncio
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional

import mlx.core as mx

from omlx.engine_core import get_mlx_executor
from omlx.mx_buffer_lock import locked_sync_and_clear_cache


@dataclass
class GenerationOutput:
    """
    Output from generation.

    Compatible with both simple and batched engines.
    """
    text: str
    tokens: List[int] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: Optional[str] = "stop"
    # For streaming
    new_text: str = ""
    finished: bool = True
    # For tool calling (Harmony and other models)
    tool_calls: Optional[List[Dict[str, Any]]] = None
    # Prefix cache stats
    cached_tokens: int = 0


class BaseEngine(ABC):
    """
    Abstract base class for inference engines.

    Both SimpleEngine and BatchedEngine implement this interface,
    allowing the server to use either without code changes.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the model name."""
        pass

    @property
    @abstractmethod
    def tokenizer(self) -> Any:
        """Get the tokenizer."""
        pass

    @abstractmethod
    async def start(self) -> None:
        """Start the engine (load model if not loaded)."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the engine and cleanup resources."""
        pass

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        stop: Optional[List[str]] = None,
        request_id: Optional[str] = None,
        **kwargs,
    ) -> GenerationOutput:
        """
        Generate a complete response (non-streaming).

        Args:
            prompt: Input text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            top_k: Top-k sampling (0 = disabled)
            repetition_penalty: Repetition penalty (1.0 = disabled)
            stop: Stop sequences
            **kwargs: Additional model-specific parameters

        Returns:
            GenerationOutput with complete text
        """
        pass

    @abstractmethod
    async def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        stop: Optional[List[str]] = None,
        request_id: Optional[str] = None,
        **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        """
        Stream generation token by token.

        Args:
            prompt: Input text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            top_k: Top-k sampling (0 = disabled)
            repetition_penalty: Repetition penalty (1.0 = disabled)
            stop: Stop sequences
            **kwargs: Additional model-specific parameters

        Yields:
            GenerationOutput with incremental text
        """
        pass

    @abstractmethod
    async def chat(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        tools: Optional[List[dict]] = None,
        request_id: Optional[str] = None,
        **kwargs,
    ) -> GenerationOutput:
        """
        Chat completion (non-streaming).

        Args:
            messages: List of chat messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            top_k: Top-k sampling (0 = disabled)
            repetition_penalty: Repetition penalty (1.0 = disabled)
            tools: Optional tool definitions
            **kwargs: Additional model-specific parameters

        Returns:
            GenerationOutput with assistant response
        """
        pass

    @abstractmethod
    async def stream_chat(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        tools: Optional[List[dict]] = None,
        request_id: Optional[str] = None,
        **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        """
        Stream chat completion token by token.

        Args:
            messages: List of chat messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            top_k: Top-k sampling (0 = disabled)
            repetition_penalty: Repetition penalty (1.0 = disabled)
            tools: Optional tool definitions
            **kwargs: Additional model-specific parameters

        Yields:
            GenerationOutput with incremental text
        """
        pass

    @property
    def scheduler(self) -> Optional[Any]:
        """Get the engine's scheduler, if any.

        Used by ProcessMemoryEnforcer to propagate memory limits for
        inline prefill checking.  Returns None by default; overridden
        by BatchedEngine / VLMBatchedEngine which have a scheduler via
        their AsyncEngineCore.
        """
        return None

    @property
    def max_context_window(self) -> Optional[int]:
        """Get the model's max context window from its config.

        Returns max_position_embeddings (or equivalent) if available,
        otherwise None. Subclasses may override for model-specific logic.
        """
        return None

    @property
    @abstractmethod
    def model_type(self) -> Optional[str]:
        """Get the model type from config.json (e.g., 'gpt_oss', 'llama', 'qwen2').

        This can be used to apply model-specific processing.

        Returns:
            Model type string or None if not available.
        """
        pass

    @property
    def grammar_compiler(self):
        """Return the grammar compiler for this engine, or ``None``.

        Subclasses that support xgrammar should override this with a
        lazy-initializing property.
        """
        return None

    @property
    def prefix_cache_enabled(self) -> bool:
        """Whether automatic prefix caching is active on this engine.

        Subclasses that wire up a BlockAwarePrefixCache should override this.
        """
        return False

    def has_active_requests(self) -> bool:
        """Check if the engine has active in-flight requests.

        Used by EnginePool.check_ttl_expirations() to prevent unloading
        a model while requests are still being processed.

        Returns:
            True if there are active requests, False otherwise.
        """
        return False

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics.

        Returns:
            Dictionary containing engine statistics.
        """
        pass

    @abstractmethod
    def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        """Get cache statistics.

        Returns:
            Dictionary containing cache statistics, or None if not applicable.
        """
        pass


class BaseNonStreamingEngine(ABC):
    """Base class for non-streaming engines (embedding, reranker, STT, TTS, STS).

    These engines compute outputs by submitting work to the single-threaded
    MLX executor via ``run_in_executor``. They don't support streaming or
    chat completion interfaces.

    Abort protocol
    --------------
    ``abort_all_requests`` signals in-flight operations to fail at their
    next checkpoint with :class:`RequestAbortedError`. Python cannot
    preempt an MLX kernel that is already running on the executor
    thread, but the async wrapper that awaits the executor future can
    discard the result and raise ``RequestAbortedError`` to the handler.
    This matches the ``BatchedEngine`` / ``VLMBatchedEngine`` contract
    so :class:`ProcessMemoryEnforcer` can treat every engine uniformly.

    The abort is **terminal**: once fired, the engine refuses all new
    and in-flight operations until ``stop()`` runs. The enforcer always
    pairs ``abort_all_requests`` with ``_unload_engine``, so a fresh
    request arriving after abort sees ``entry.engine is None`` via
    ``EnginePool.ensure_engine_alive`` and receives HTTP 503 without
    ever reaching this engine.
    """

    def __init__(self):
        self._active_count = 0
        self._active_lock = threading.Lock()
        # Terminal abort flag set by abort_all_requests(). Checked by
        # concrete engines via _raise_if_aborted at each run_in_executor
        # boundary. asyncio.Event() is safe to construct without a
        # running loop in Python 3.10+.
        self._aborted = asyncio.Event()
        self._activities: Dict[str, Dict[str, Any]] = {}

    def has_active_requests(self) -> bool:
        """Check if the engine has active in-flight requests."""
        with self._active_lock:
            return self._active_count > 0

    _ACTIVITY_RESERVED_KEYS = {
        "request_id",
        "kind",
        "detail",
        "started_at",
        "last_activity_at",
        "total_items",
    }

    def _sanitize_activity_metadata(
        self, metadata: Dict[str, Any] | None
    ) -> Dict[str, Any]:
        """Drop reserved activity keys from caller-provided metadata.

        Timing keys are owned by the tracker: _begin_activity sets them and
        _update_activity always advances last_activity_at to "now".
        """
        if not metadata:
            return {}
        return {
            key: value
            for key, value in metadata.items()
            if key not in self._ACTIVITY_RESERVED_KEYS
        }

    def _begin_activity(
        self,
        kind: str,
        detail: str | None = None,
        total_items: int | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> str:
        """Track a non-streaming operation for admin visibility."""
        activity_id = str(uuid.uuid4())
        now = time.monotonic()
        with self._active_lock:
            self._active_count += 1
            activity = {
                "request_id": activity_id,
                "kind": kind,
                "detail": detail or kind,
                "started_at": now,
                "last_activity_at": now,
                "total_items": total_items,
            }
            activity.update(self._sanitize_activity_metadata(metadata))
            self._activities[activity_id] = activity
        return activity_id

    def _update_activity(self, activity_id: str, **updates: Any) -> None:
        """Update tracked non-streaming operation metadata."""
        with self._active_lock:
            activity = self._activities.get(activity_id)
            if activity is None:
                return
            activity.update(self._sanitize_activity_metadata(updates))
            activity["last_activity_at"] = time.monotonic()

    def _end_activity(self, activity_id: str) -> None:
        """End an activity."""
        with self._active_lock:
            removed = self._activities.pop(activity_id, None)
            if removed is None:
                raise RuntimeError(
                    f"Activity {activity_id} ended more than once or was never started"
                )
            self._active_count -= 1
            if self._active_count < 0:
                raise RuntimeError("Active request count became negative")

    async def _finish_activity(self, activity_id: str) -> None:
        """End an activity and clear the Metal buffer pool.

        Always clears per request. Gating the clear on `_active_count == 0`
        caused unbounded Metal pool growth under concurrent workloads (#684),
        because indexing clients keep the active count above zero indefinitely.
        `mx.synchronize()` is required before `mx.clear_cache()` to avoid
        Metal buffer races on M3/M4 (#300, #888, #1106). The clear runs on
        the MLX executor AND under the process-wide buffer-access lock
        (`locked_sync_and_clear_cache`) so it can't reclaim the buffer pool
        mid-read from the async phase-2 KV save or another engine path.
        """
        self._end_activity(activity_id)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            get_mlx_executor(),
            locked_sync_and_clear_cache,
        )

    async def abort_all_requests(self) -> int:
        """Signal all in-flight operations to abort at the next checkpoint.

        Non-streaming engines run MLX work on the single-threaded MLX
        executor via ``run_in_executor``. Python can't preempt an MLX
        kernel mid-call, but it can cause the async wrapper to discard
        the executor result and raise :class:`RequestAbortedError` to
        the handler. Memory reclamation happens naturally after the
        in-flight call finishes — the enforcer's subsequent
        ``_unload_engine`` + deferred cleanup handles ``mx.clear_cache``.

        Returns:
            The number of in-flight operations observed at abort time.
            Used only for logging by the enforcer.
        """
        count = self._active_count
        self._aborted.set()
        return count

    def _mark_stopped(self) -> None:
        """Mark the engine as terminal so post-stop calls raise typed errors.

        Concrete ``stop()`` implementations must call this at the top of
        their stop sequence, before clearing ``self._model``. It sets
        ``_aborted`` so that any handler racing with stop — one that
        captured an engine reference before the enforcer's abort +
        unload and hasn't tripped ``EnginePool.ensure_engine_alive`` —
        sees :class:`RequestAbortedError` at the next
        ``_raise_if_aborted`` checkpoint instead of a plain
        ``RuntimeError("Engine not started")`` from the
        ``self._model is None`` guard.
        """
        self._aborted.set()

    def _raise_if_aborted(self) -> None:
        """Raise :class:`RequestAbortedError` if this engine has been aborted.

        Concrete engines must call this:

        1. **At each public entry point**, *before* the ``self._model
           is None`` "engine started" guard, so a handler racing with
           ``stop()`` observes the typed abort rather than the plain
           RuntimeError.
        2. **Immediately after each ``run_in_executor`` await**, so an
           in-flight operation whose abort fired while its executor
           future was running discards the result and raises instead
           of returning stale output to the handler.
        """
        if self._aborted.is_set():
            from ..exceptions import RequestAbortedError
            raise RequestAbortedError(
                f"Engine for {self.model_name} has been aborted "
                f"due to memory pressure. Please retry the request."
            )

    def get_activity_snapshot(self) -> Dict[str, Any]:
        """Return active non-streaming operations for admin display."""
        now = time.monotonic()
        with self._active_lock:
            activities = []
            for activity in self._activities.values():
                item = dict(activity)
                started_at = item.pop("started_at", None)
                last_activity_at = item.pop("last_activity_at", None)
                item["elapsed_seconds"] = (
                    max(0.0, now - started_at) if started_at is not None else None
                )
                item["last_activity_age_seconds"] = (
                    max(0.0, now - last_activity_at)
                    if last_activity_at is not None
                    else None
                )
                activities.append(item)
            return {
                "active_requests": self._active_count,
                "activities": activities,
            }

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the model name."""
        pass

    @abstractmethod
    async def start(self) -> None:
        """Start the engine (load model if not loaded)."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the engine and cleanup resources."""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics.

        Returns:
            Dictionary containing engine statistics.
        """
        pass
