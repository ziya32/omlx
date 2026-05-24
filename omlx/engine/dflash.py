# SPDX-License-Identifier: Apache-2.0
"""
DFlash engine for block diffusion speculative decoding.

This engine wraps dflash-mlx to provide 3-4x faster decoding on Apple Silicon.
For short/medium contexts it uses speculative decoding; for long contexts
(>DFLASH_MAX_CTX) it evicts dflash models and switches to omlx's BatchedEngine
or VLMBatchedEngine which have paged cache, SSD cache, and continuous batching.
"""

import asyncio
import copy
import gc
import json
import logging
import os
import threading
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import mlx.core as mx

from ..api.tool_calling import convert_tools_for_template
from ..api.utils import clean_special_tokens, detect_and_strip_partial
from ..exceptions import RequestAbortedError
from ..utils.model_loading import maybe_apply_pre_load_patches
from .base import BaseEngine, GenerationOutput
from ..mx_buffer_lock import locked_sync_and_clear_cache, run_locked

logger = logging.getLogger(__name__)

DEFAULT_MAX_DFLASH_CTX = 4096


def is_dflash_compatible(model_path: str | Path) -> tuple[bool, str]:
    """Decide whether ``model_path`` can run on the current dflash backend.

    DFlash registers QwenGdnTargetOps and Gemma4TargetOps. The top-level
    ``model_type`` is the canonical discriminator: Gemma4 multimodal
    configs use ``gemma4`` at the top, while MTP-only variants (e.g. the
    Gemma4 ``-assistant`` checkpoint) declare ``gemma4_assistant`` even
    though their nested ``text_config.model_type`` is still
    ``gemma4_text``. Reading top-level only keeps the gate aligned with
    what dflash will actually load.

    Imported by ``admin/routes.py`` to flag per-model DFlash compatibility
    in the admin UI; that import is wrapped in try/except so a missing
    symbol would degrade silently to "not compatible" for every model.

    Returns:
        (is_compatible, reason). ``reason`` is empty when compatible.
    """
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        return False, f"config.json not found at {config_path}"
    try:
        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        return False, f"failed to read config.json: {e}"

    model_type = str(cfg.get("model_type") or "").lower()

    is_qwen = "qwen" in model_type
    is_gemma4 = model_type in ("gemma4", "gemma4_text")
    if not (is_qwen or is_gemma4):
        return False, (
            f"DFlash supports only Qwen and Gemma4 models "
            f"(model_type='{cfg.get('model_type', '')}')"
        )
    return True, ""


class DFlashEngine(BaseEngine):
    """
    DFlash speculative decoding engine with automatic fallback.

    For prompts within max_dflash_ctx tokens, uses block diffusion speculative
    decoding for 3-4x faster generation. For longer prompts, evicts dflash
    models from memory and delegates to a fallback engine (BatchedEngine or
    VLMBatchedEngine) that provides paged cache, SSD cache, and continuous
    batching.
    """

    def __init__(
        self,
        model_name: str,
        draft_model_path: str,
        draft_quant_bits: int | None = None,
        draft_quant_enabled: bool | None = None,
        draft_quant_weight_bits: int | None = None,
        draft_quant_activation_bits: int | None = None,
        draft_quant_group_size: int | None = None,
        model_settings: Any | None = None,
        fallback_engine_type: str = "batched",
        scheduler_config: Any | None = None,
        process_memory_max_bytes: int = 0,
        omlx_ssd_cache_dir: str | Path | None = None,
    ):
        self._model_name = model_name
        self._draft_model_path = draft_model_path
        self._draft_quant_bits = draft_quant_bits
        # New-style draft-quant config (dflash 0.1.5+). Lets callers pass an
        # explicit ``enabled`` toggle plus separate bit-widths so a profile
        # can flip quantization on without filling in every value.
        self._draft_quant_enabled = draft_quant_enabled
        self._draft_quant_weight_bits = draft_quant_weight_bits
        self._draft_quant_activation_bits = draft_quant_activation_bits
        self._draft_quant_group_size = draft_quant_group_size
        self._omlx_ssd_cache_dir = (
            Path(omlx_ssd_cache_dir) if omlx_ssd_cache_dir else None
        )
        self._model_settings = model_settings
        self._fallback_engine_type = fallback_engine_type
        self._scheduler_config = scheduler_config
        self._process_memory_max_bytes = process_memory_max_bytes

        self._target_model = None
        self._draft_model = None
        self._tokenizer_obj = None
        self._executor_tokenizer = None
        self._loaded = False
        self._model_type_str = None
        self._fallback_engine: BaseEngine | None = None
        self._in_fallback_mode = False
        # Per-engine dflash-mlx runtime context, built once in start() from
        # ModelSettings tuning fields (dflash_draft_window_size,
        # dflash_draft_sink_size, dflash_verify_mode, etc.). None means
        # stream_dflash_generate will use its own defaults.
        self._runtime_context: Any | None = None
        # dflash prefix cache reference (populated in start() / cleared
        # on eviction). Kept here so callers that touch the attribute
        # after a failed start() don't AttributeError.
        self._dflash_prefix_cache: Any | None = None
        # L2 SSD cache max bytes (PR #1326). None falls back to the
        # 20 GiB default; ``_resolve_dflash_l2_dir`` consumes this.
        self._ssd_cache_max_bytes = int(
            getattr(model_settings, "dflash_ssd_cache_max_bytes", 20 * 1024**3)
            if model_settings
            else 20 * 1024**3
        )
        # Cached settings reads for #1276 tuning. None → let dflash-mlx pick
        # its default (window=1024, sink=64, verify_mode='adaptive').
        self._draft_window_size = (
            getattr(model_settings, "dflash_draft_window_size", None)
            if model_settings else None
        )
        self._draft_sink_size = (
            getattr(model_settings, "dflash_draft_sink_size", None)
            if model_settings else None
        )
        self._verify_mode = (
            getattr(model_settings, "dflash_verify_mode", None)
            if model_settings else None
        )
        # Prefix-cache (L1/L2) settings used by _build_runtime_context +
        # _resolve_dflash_l2_dir.
        self._in_memory_cache_enabled = (
            bool(getattr(model_settings, "dflash_in_memory_cache", True))
            if model_settings else True
        )
        self._in_memory_cache_max_entries = int(
            getattr(model_settings, "dflash_in_memory_cache_max_entries", 4)
            if model_settings else 4
        )
        self._in_memory_cache_max_bytes = int(
            getattr(
                model_settings, "dflash_in_memory_cache_max_bytes",
                8 * 1024**3,
            )
            if model_settings else 8 * 1024**3
        )
        self._ssd_cache_requested = (
            bool(getattr(model_settings, "dflash_ssd_cache", False))
            if model_settings else False
        )
        # Protocol-specific output parser factory (gemma4 / harmony) —
        # detected at start() once the target model is loaded. None means
        # the streaming detokenizer is used as-is (qwen, llama, etc.).
        self._output_parser_factory: Any | None = None

        # Single-request tracking for has_active_requests / abort_request.
        # DFlash processes one request at a time on the MLX executor; the
        # active request id (or "<no_id>" when caller didn't supply one)
        # lives here for the duration of generate / stream_generate.
        # threading.Event is used (not asyncio.Event) so the executor-thread
        # for-loop in _run_generate_streaming can poll is_set() safely
        # without crossing the asyncio/thread boundary.
        self._active_request_id: str | None = None
        self._active_request_lock = threading.Lock()
        self._abort_event = threading.Event()
        # Terminal abort flag set by abort_all_requests — separate from the
        # per-call abort_event so a memory-pressure abort persists past the
        # current request and refuses new ones until stop().
        self._aborted_terminal = False

        # ``dflash_max_ctx`` precedence: per-model setting > env var >
        # DEFAULT_MAX_DFLASH_CTX. A ``None`` setting (the model_settings
        # default) means "unlimited" — every prompt size stays on dflash.
        # An env override of "" is treated as "fall back to default".
        if model_settings is not None and hasattr(model_settings, "dflash_max_ctx"):
            self._max_dflash_ctx = model_settings.dflash_max_ctx
        else:
            raw = os.environ.get("DFLASH_MAX_CTX", str(DEFAULT_MAX_DFLASH_CTX)).strip()
            try:
                self._max_dflash_ctx = max(1, int(raw)) if raw else DEFAULT_MAX_DFLASH_CTX
            except ValueError:
                self._max_dflash_ctx = DEFAULT_MAX_DFLASH_CTX

    @staticmethod
    def _build_quant_spec(
        weight_bits: int | None,
        activation_bits: int | None,
        group_size: int | None,
    ) -> str:
        """Convert draft quantization config into dflash 0.1.5's spec string.

        None values fall back to dflash defaults (w4a16:gs64), so the spec
        stays valid when a profile or external API sets ``enabled=True``
        without filling in every bit value.
        """
        wb = weight_bits if weight_bits is not None else 4
        ab = activation_bits if activation_bits is not None else 16
        gs = group_size if group_size is not None else 64
        return f"w{wb}a{ab}:gs{gs}"

    def _resolve_dflash_l2_dir(self) -> Path | None:
        """Compute the dflash L2 cache directory under the omlx SSD cache root."""
        if not self._ssd_cache_requested:
            return None
        if self._omlx_ssd_cache_dir is None:
            logger.warning(
                "DFlash SSD cache requested but omlx paged SSD cache directory "
                "is not configured; disabling L2."
            )
            return None
        if not self._in_memory_cache_enabled:
            logger.warning(
                "DFlash SSD cache requires in-memory cache; disabling L2."
            )
            return None
        return self._omlx_ssd_cache_dir / "dflash_l2"

    def _build_runtime_context(self) -> Any:
        """Build a dflash-mlx ``RuntimeContext`` from the engine's settings.

        Called at ``start()``; the result is cached on ``self._runtime_context``
        and passed to every ``stream_dflash_generate`` invocation. Tests
        also call this directly to verify per-setting plumbing.
        """
        from dflash_mlx.runtime.config import runtime_config_from_defaults
        from dflash_mlx.runtime.context import build_runtime_context

        l2_dir = self._resolve_dflash_l2_dir()
        l2_enabled = l2_dir is not None
        cfg = runtime_config_from_defaults(
            prefix_cache=self._in_memory_cache_enabled,
            prefix_cache_max_entries=self._in_memory_cache_max_entries,
            prefix_cache_max_bytes=self._in_memory_cache_max_bytes,
            prefix_cache_l2=l2_enabled,
            prefix_cache_l2_dir=str(l2_dir) if l2_dir else "",
            # Per-model L2 disk budget. dflash-mlx's _evict_to_budget drops
            # the oldest snapshots once dflash_l2/ exceeds this, so the
            # directory stays bounded instead of filling the disk
            # (PR #1326). Falls back to the 20 GiB default set in
            # __init__ via ModelSettings.
            prefix_cache_l2_max_bytes=self._ssd_cache_max_bytes if l2_enabled else 0,
            # None → dflash-mlx fills in DEFAULT_RUNTIME_CONFIG values
            # (window=1024, sink=64, verify_mode='adaptive').
            draft_window_size=self._draft_window_size,
            draft_sink_size=self._draft_sink_size,
            verify_mode=self._verify_mode,
        )
        return build_runtime_context(cfg)

    def _get_think_token_id(self, attr: str) -> int | None:
        """Safely read think_start_id / think_end_id from the tokenizer."""
        try:
            return getattr(self._tokenizer_obj, attr, None)
        except (ValueError, TypeError):
            return None

    def _detect_needs_think_prefix(self, prompt_tokens: list[int]) -> bool:
        """Detect if prompt ends with an open ``<think>`` tag.

        DFlash bypasses the scheduler, so the ``<think>\\n`` prefix that
        the scheduler normally prepends to the first chunk for reasoning
        models must be reproduced here. Mirrors the scheduler's detection
        logic. Returns False for disabled-thinking patterns like
        ``<think></think>`` where ``</think>`` immediately follows
        ``<think>`` in the prompt tail.
        """
        if not prompt_tokens:
            return False

        think_start_id = self._get_think_token_id('think_start_id')
        if think_start_id is None and self._tokenizer_obj is not None:
            try:
                tid = self._tokenizer_obj.convert_tokens_to_ids("<think>")
                if tid == getattr(self._tokenizer_obj, 'unk_token_id', None):
                    return False
                think_start_id = tid
            except (AttributeError, KeyError, TypeError):
                return False

        if not think_start_id:
            return False

        last_tokens = list(prompt_tokens[-3:])
        if think_start_id not in last_tokens:
            return False

        last_idx = len(last_tokens) - 1 - last_tokens[::-1].index(think_start_id)
        after_start = last_tokens[last_idx + 1:]

        if after_start:
            think_end_id = self._get_think_token_id('think_end_id')
            if think_end_id is not None and think_end_id in after_start:
                return False
            if self._tokenizer_obj is not None:
                try:
                    tid = self._tokenizer_obj.convert_tokens_to_ids("</think>")
                    unk = getattr(self._tokenizer_obj, 'unk_token_id', None)
                    if tid != unk and tid in after_start:
                        return False
                except (AttributeError, KeyError, TypeError):
                    pass

        return True

    def _think_prefix_text(self) -> str:
        """Return the opening think tag string to prepend (e.g. '<think>\\n')."""
        tag = getattr(self._tokenizer_obj, 'think_start', '<think>')
        return f"{tag}\n"

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer_obj

    @property
    def model_type(self) -> str | None:
        return self._model_type_str

    async def start(self) -> None:
        if self._loaded:
            return

        from ..engine_core import get_mlx_executor

        loop = asyncio.get_running_loop()

        def _load_models():
            # dflash-mlx 0.1.7 (1ba6713) moved load_target_bundle / load_draft_bundle
            # out of runtime/__init__ into runtime.bundle; importing from the
            # legacy path now raises ImportError. Use the explicit submodule.
            from dflash_mlx.runtime.bundle import (
                load_target_bundle, load_draft_bundle,
            )

            # Apply the same pre-load patches BatchedEngine uses before
            # mlx_lm.load() runs. MTP-bearing targets (e.g. Qwen3.6 *-mtp)
            # need the MTP-compat sanitize patch or stock mlx-lm double-shifts
            # the already-converted norm and emits garbage. dflash and mtp are
            # mutually exclusive per model_settings, so this never attaches an
            # MTP head; it only fixes sanitize. See issue #1318.
            maybe_apply_pre_load_patches(
                self._model_name, model_settings=self._model_settings
            )

            model, tokenizer, meta = load_target_bundle(self._model_name)

            draft, draft_meta = load_draft_bundle(
                self._draft_model_path,
                quantize_draft=bool(self._draft_quant_bits),
            )
            return model, tokenizer, meta, draft

        result = await loop.run_in_executor(get_mlx_executor(), lambda: run_locked(_load_models))
        self._target_model, self._tokenizer_obj, target_meta, self._draft_model = result

        # Deep-copy tokenizer for executor-thread usage (dflash generation).
        # The original self._tokenizer_obj stays for event-loop operations
        # (encode, apply_chat_template, count_chat_tokens).
        # See: https://github.com/huggingface/tokenizers/issues/537
        self._executor_tokenizer = copy.deepcopy(self._tokenizer_obj)

        # Extract model_type from config
        config = target_meta.get("config", {})
        if isinstance(config, dict):
            self._model_type_str = config.get("model_type")
        elif hasattr(config, "model_type"):
            self._model_type_str = config.model_type

        # Build the dflash-mlx runtime context from ModelSettings tuning
        # fields (#1276). Passed to every stream_dflash_generate call so
        # users see their configured draft_window_size / draft_sink_size /
        # verify_mode rather than dflash-mlx defaults.
        try:
            from dflash_mlx.runtime.config import runtime_config_from_defaults
            from dflash_mlx.runtime.context import build_runtime_context

            cfg_kwargs: dict[str, Any] = {}
            if self._draft_window_size is not None:
                cfg_kwargs["draft_window_size"] = self._draft_window_size
            if self._draft_sink_size is not None:
                cfg_kwargs["draft_sink_size"] = self._draft_sink_size
            if self._verify_mode is not None:
                cfg_kwargs["verify_mode"] = self._verify_mode
            runtime_config = runtime_config_from_defaults(**cfg_kwargs)
            self._runtime_context = build_runtime_context(runtime_config)
        except Exception as e:
            # Tuning is non-essential; if the dflash-mlx API moves again or
            # one of the settings doesn't apply, fall back to defaults.
            logger.warning(
                "DFlash runtime context build failed (%s); using "
                "dflash-mlx defaults", e,
            )
            self._runtime_context = None

        self._loaded = True
        self._in_fallback_mode = False
        logger.info(
            f"DFlashEngine loaded: target={self._model_name}, "
            f"draft={self._draft_model_path}, "
            f"max_ctx={self._max_dflash_ctx}, "
            f"fallback={self._fallback_engine_type}, "
            f"draft_window={self._draft_window_size}, "
            f"draft_sink={self._draft_sink_size}, "
            f"verify_mode={self._verify_mode}"
        )

    async def _evict_dflash_and_start_fallback(self) -> None:
        """Evict dflash models from memory, verify release, then start fallback engine."""
        from ..engine_core import get_mlx_executor

        loop = asyncio.get_running_loop()
        pre_active = mx.get_active_memory()

        # Release dflash model references
        self._target_model = None
        self._draft_model = None
        self._executor_tokenizer = None

        # Force memory reclaim with settle barrier
        gc.collect()
        await loop.run_in_executor(
            get_mlx_executor(),
            locked_sync_and_clear_cache,
        )

        # Poll for actual memory release (same pattern as engine_pool._unload_engine)
        for settle_round in range(10):
            active_now = mx.get_active_memory()
            freed = pre_active - active_now
            if freed > 0:
                logger.info(
                    f"DFlash models evicted: freed={freed / 1024**3:.2f}GB "
                    f"(round {settle_round + 1})"
                )
                break
            await asyncio.sleep(0.5)
            gc.collect()
            await loop.run_in_executor(
                get_mlx_executor(),
                locked_sync_and_clear_cache,
            )
        else:
            logger.warning("DFlash model eviction: memory settle timed out")

        # Start fallback engine.  For VLM, forward process_memory_max_bytes
        # so the per-prefill enforcer check in scheduler still works once
        # we cross the DFlash context cap; BatchedEngine doesn't take it.
        if self._fallback_engine_type == "vlm":
            from .vlm import VLMBatchedEngine
            self._fallback_engine = VLMBatchedEngine(
                model_name=self._model_name,
                scheduler_config=self._scheduler_config,
                model_settings=self._model_settings,
                process_memory_max_bytes=self._process_memory_max_bytes,
            )
        else:
            from .batched import BatchedEngine
            self._fallback_engine = BatchedEngine(
                model_name=self._model_name,
                scheduler_config=self._scheduler_config,
                model_settings=self._model_settings,
            )
        await self._fallback_engine.start()
        self._in_fallback_mode = True
        logger.info(
            f"DFlash fallback engine started: {self._fallback_engine_type}"
        )

    async def stop(self) -> None:
        # Signal any in-flight executor loop to exit at its next event,
        # then await the fallback shutdown so we don't tear down state
        # under it.
        self._abort_event.set()
        if self._fallback_engine is not None:
            await self._fallback_engine.stop()
            self._fallback_engine = None
        self._target_model = None
        self._draft_model = None
        self._tokenizer_obj = None
        self._executor_tokenizer = None
        self._in_fallback_mode = False
        self._loaded = False
        # Reset abort state so a subsequent start() is a clean slate.
        self._aborted_terminal = False
        self._abort_event.clear()
        with self._active_request_lock:
            self._active_request_id = None
        logger.info("DFlashEngine stopped")

    def _apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        is_partial: bool | None = None,
    ) -> str:
        """Apply chat template to messages.

        Args:
            messages: List of chat messages.
            tools: Optional tool definitions.
            chat_template_kwargs: Optional kwargs for the chat template
                (e.g. enable_thinking, reasoning_effort).
            is_partial: Explicit partial-mode signal from the API server.
                ``True``/``False`` — server has already decided; the
                ``partial`` key is cleaned from message dicts but no
                detection is performed. ``None`` (default) — auto-detect
                from messages for backward compatibility with direct
                engine callers. Mirrors BatchedEngine's contract so the
                server can ``count_chat_tokens(..., is_partial=...)``
                followed by ``_apply_chat_template(..., is_partial=...)``
                render with identical flags.
        """
        if hasattr(self._tokenizer_obj, "apply_chat_template"):
            if is_partial is None:
                is_partial = detect_and_strip_partial(messages)
            else:
                # Server already resolved partial; just clean residual keys
                # so the chat template never sees the non-standard field.
                for msg in messages:
                    msg.pop("partial", None)
            template_kwargs = {
                "tokenize": False,
                "add_generation_prompt": not is_partial,
            }
            if is_partial:
                template_kwargs["continue_final_message"] = True
            if tools:
                template_kwargs["tools"] = tools
            if chat_template_kwargs:
                template_kwargs.update(chat_template_kwargs)
            try:
                return self._tokenizer_obj.apply_chat_template(
                    messages, **template_kwargs
                )
            except TypeError:
                if chat_template_kwargs:
                    for key in chat_template_kwargs:
                        template_kwargs.pop(key, None)
                template_kwargs.pop("tools", None)
                template_kwargs.pop("enable_thinking", None)
                return self._tokenizer_obj.apply_chat_template(
                    messages, **template_kwargs
                )
        else:
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
            return prompt + "\nassistant:"

    def count_chat_tokens(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        is_partial: bool | None = None,
    ) -> int:
        """Count prompt tokens for chat messages after applying chat template.

        Args:
            messages: List of chat messages.
            tools: Optional tool definitions.
            chat_template_kwargs: Optional kwargs for chat template.
            is_partial: Explicit partial-mode signal (see _apply_chat_template).
        """
        template_tools = convert_tools_for_template(tools) if tools else None
        prompt = self._apply_chat_template(
            messages, template_tools,
            chat_template_kwargs=chat_template_kwargs,
            is_partial=is_partial,
        )
        return len(self._tokenizer_obj.encode(prompt))

    def _should_fallback(self, prompt_tokens: list[int]) -> bool:
        """``None`` ``_max_dflash_ctx`` means dflash handles every prompt size."""
        if self._max_dflash_ctx is None:
            return False
        return len(prompt_tokens) >= self._max_dflash_ctx

    def _enter_active(self, request_id: str | None) -> str:
        """Mark this engine as actively processing one request.

        Returns the request id we use internally — caller's id when
        provided, else a synthesized "<no_id>" sentinel so abort_request
        with a real id can never accidentally match an anonymous slot.
        Clears any previous per-call abort flag before returning.
        """
        rid = request_id if request_id else "<no_id>"
        with self._active_request_lock:
            self._active_request_id = rid
            self._abort_event.clear()
        return rid

    def _exit_active(self) -> None:
        """Clear active-request tracking after a generate / stream call."""
        with self._active_request_lock:
            self._active_request_id = None

    def _raise_if_terminally_aborted(self) -> None:
        """Raise RequestAbortedError if abort_all_requests has fired."""
        if self._aborted_terminal:
            raise RequestAbortedError(
                f"Engine for {self._model_name} has been aborted "
                f"due to memory pressure. Please retry the request."
            )

    def _run_generate_streaming(
        self,
        prompt_tokens: list[int],
        max_tokens: int,
        temperature: float,
        queue: asyncio.Queue,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        """Run dflash generation with streaming on MLX executor thread."""
        from dflash_mlx.generate import get_stop_token_ids
        from dflash_mlx.runtime import stream_dflash_generate

        try:
            stop_ids = get_stop_token_ids(self._executor_tokenizer)

            # Use streaming detokenizer for proper UTF-8 handling (CJK etc.)
            detokenizer = None
            try:
                from mlx_lm.tokenizer_utils import NaiveStreamingDetokenizer
                detokenizer = NaiveStreamingDetokenizer(self._executor_tokenizer)
            except ImportError:
                pass

            # dflash-mlx 0.1.7 doesn't expose sampling temperature through
            # stream_dflash_generate; the runtime decides via VerifyConfig.
            # Temperature is honored only on fallback (BatchedEngine) when
            # DFlash falls back; here it's dropped at the API boundary.
            generator = stream_dflash_generate(
                target_model=self._target_model,
                tokenizer=self._executor_tokenizer,
                draft_model=self._draft_model,
                prompt="",
                max_new_tokens=max_tokens,
                stop_token_ids=stop_ids,
                prompt_tokens_override=prompt_tokens,
                runtime_context=self._runtime_context,
            )
            for event in generator:
                # Abort check between events.  dflash-mlx's runtime is a
                # synchronous generator running on the MLX executor thread —
                # we can't preempt the in-flight verify pass, but we can
                # break out of the loop at the next event boundary so
                # client-disconnect / memory-pressure aborts surface within
                # one cycle (~50-200ms) instead of waiting for the full
                # generation to finish.
                if self._abort_event.is_set():
                    logger.info(
                        "DFlash generation aborted (abort_event set)"
                    )
                    asyncio.run_coroutine_threadsafe(
                        queue.put(("", [], True, {"aborted": True})), loop
                    )
                    return
                event_type = event.get("event")

                if event_type == "token":
                    token_id = event["token_id"]
                    # Skip EOS/stop tokens from output
                    if token_id in stop_ids:
                        continue
                    if detokenizer is not None:
                        detokenizer.add_token(token_id)
                        text = detokenizer.last_segment
                    else:
                        text = self._executor_tokenizer.decode([token_id])
                    asyncio.run_coroutine_threadsafe(
                        queue.put((text, [token_id], False, None)), loop
                    )

                elif event_type == "summary":
                    gen_tokens = event.get("generation_tokens", 0)
                    accept_ratio = event.get("acceptance_ratio", 0)
                    cycles = event.get("cycles_completed", 0)
                    elapsed_us = event.get("elapsed_us", 0)
                    elapsed_s = elapsed_us / 1e6 if elapsed_us else 0
                    gen_tps = gen_tokens / elapsed_s if elapsed_s > 0 else 0
                    fallback = event.get("fallback_ar", False)
                    logger.info(
                        f"DFlash generation complete: "
                        f"{gen_tokens} tokens, "
                        f"{gen_tps:.1f} tok/s, "
                        f"acceptance={accept_ratio:.1%}, "
                        f"cycles={cycles}"
                        f"{', fallback=AR' if fallback else ''}"
                    )
                    metrics = {
                        "prompt_tokens": event.get("prompt_token_count", 0),
                        "completion_tokens": gen_tokens,
                        "acceptance_ratio": accept_ratio,
                        "cycles_completed": cycles,
                    }
                    asyncio.run_coroutine_threadsafe(
                        queue.put(("", [], True, metrics)), loop
                    )

        except Exception as e:
            logger.error(f"DFlash streaming generation error: {e}")
            asyncio.run_coroutine_threadsafe(
                queue.put(("", [], True, {"error": str(e)})), loop
            )

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
        stop: list[str] | None = None,
        request_id: str | None = None,
        **kwargs,
    ) -> GenerationOutput:
        self._raise_if_terminally_aborted()
        if not self._loaded:
            await self.start()

        prompt_tokens = self._tokenizer_obj.encode(prompt)

        # Fallback: evict dflash models, start LLM/VLM engine
        if self._should_fallback(prompt_tokens):
            if not self._in_fallback_mode:
                logger.info(
                    f"DFlash context fallback: {len(prompt_tokens)} >= {self._max_dflash_ctx}, "
                    f"evicting dflash models and switching to {self._fallback_engine_type} engine"
                )
                await self._evict_dflash_and_start_fallback()
            return await self._fallback_engine.generate(
                prompt=prompt, max_tokens=max_tokens, temperature=temperature,
                top_p=top_p, top_k=top_k, min_p=min_p,
                repetition_penalty=repetition_penalty,
                presence_penalty=presence_penalty, stop=stop,
                request_id=request_id, **kwargs,
            )

        # Already in fallback mode but short context came in.
        # Stay in fallback mode (reloading dflash models is expensive).
        if self._in_fallback_mode:
            return await self._fallback_engine.generate(
                prompt=prompt, max_tokens=max_tokens, temperature=temperature,
                top_p=top_p, top_k=top_k, min_p=min_p,
                repetition_penalty=repetition_penalty,
                presence_penalty=presence_penalty, stop=stop,
                request_id=request_id, **kwargs,
            )

        from ..engine_core import get_mlx_executor
        from dflash_mlx.generate import get_stop_token_ids
        from dflash_mlx.runtime import stream_dflash_generate

        loop = asyncio.get_running_loop()
        stop_ids = get_stop_token_ids(self._tokenizer_obj)

        def _run():
            # dflash-mlx 0.1.7 removed generate_dflash_once; drain the
            # streaming generator into a single summary instead. The token-
            # event stream produces the same generated_token_ids the legacy
            # API returned, and the summary event carries the metrics.
            # See note in _run_generate_streaming re: temperature being
            # dropped at the dflash-mlx v0.1.7 API boundary.
            generator = stream_dflash_generate(
                target_model=self._target_model,
                tokenizer=self._executor_tokenizer,
                draft_model=self._draft_model,
                prompt="",
                max_new_tokens=max_tokens,
                stop_token_ids=stop_ids,
                prompt_tokens_override=prompt_tokens,
                runtime_context=self._runtime_context,
            )
            token_ids: list[int] = []
            summary_metrics: dict = {}
            for event in generator:
                ev = event.get("event")
                if ev == "token":
                    tid = event["token_id"]
                    if tid not in stop_ids:
                        token_ids.append(tid)
                elif ev == "summary":
                    summary_metrics = {
                        k: v for k, v in event.items() if k != "event"
                    }
            return {
                "generated_token_ids": token_ids,
                "prompt_token_count": summary_metrics.get(
                    "prompt_token_count", len(prompt_tokens)
                ),
                "generation_tokens": summary_metrics.get(
                    "generation_tokens", len(token_ids)
                ),
                **{k: v for k, v in summary_metrics.items()
                   if k not in ("prompt_token_count", "generation_tokens")},
            }

        self._enter_active(request_id)
        try:
            summary = await loop.run_in_executor(get_mlx_executor(), _run)
            # Discard the result if abort fired while the executor was
            # running — handler sees the typed abort instead of stale text.
            if self._abort_event.is_set():
                raise RequestAbortedError(
                    "Request aborted during DFlash generation"
                )
        finally:
            self._exit_active()

        generated = summary.get("generated_token_ids", [])
        text = self._tokenizer_obj.decode(generated, skip_special_tokens=True)
        text = clean_special_tokens(text)

        return GenerationOutput(
            text=text,
            tokens=generated,
            prompt_tokens=summary.get("prompt_token_count", len(prompt_tokens)),
            completion_tokens=summary.get("generation_tokens", len(generated)),
            finish_reason="stop",
        )

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
        stop: list[str] | None = None,
        request_id: str | None = None,
        **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        self._raise_if_terminally_aborted()
        if not self._loaded:
            await self.start()

        prompt_tokens = self._tokenizer_obj.encode(prompt)

        # Fallback: evict dflash models, start LLM/VLM engine
        if self._should_fallback(prompt_tokens):
            if not self._in_fallback_mode:
                logger.info(
                    f"DFlash context fallback: {len(prompt_tokens)} >= {self._max_dflash_ctx}, "
                    f"evicting dflash models and switching to {self._fallback_engine_type} engine"
                )
                await self._evict_dflash_and_start_fallback()
            async for output in self._fallback_engine.stream_generate(
                prompt=prompt, max_tokens=max_tokens, temperature=temperature,
                top_p=top_p, top_k=top_k, min_p=min_p,
                repetition_penalty=repetition_penalty,
                presence_penalty=presence_penalty, stop=stop,
                request_id=request_id, **kwargs,
            ):
                yield output
            return

        # Already in fallback mode — stay there
        if self._in_fallback_mode:
            async for output in self._fallback_engine.stream_generate(
                prompt=prompt, max_tokens=max_tokens, temperature=temperature,
                top_p=top_p, top_k=top_k, min_p=min_p,
                repetition_penalty=repetition_penalty,
                presence_penalty=presence_penalty, stop=stop,
                request_id=request_id, **kwargs,
            ):
                yield output
            return

        prompt_len = len(prompt_tokens)
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        from ..engine_core import get_mlx_executor
        self._enter_active(request_id)
        loop.run_in_executor(
            get_mlx_executor(),
            self._run_generate_streaming,
            prompt_tokens,
            max_tokens,
            temperature,
            queue,
            loop,
        )

        total_text = ""
        total_completion = 0

        try:
            while True:
                new_text, new_tokens, finished, metrics = await queue.get()

                total_text += new_text
                total_completion += len(new_tokens)

                finish_reason = None
                if finished:
                    finish_reason = "stop"
                    if metrics and metrics.get("error"):
                        finish_reason = "error"
                    elif metrics and metrics.get("aborted"):
                        # Surface aborts as a typed exception so the FastAPI
                        # streaming handlers convert into a proper SSE error
                        # event instead of a silent close.
                        raise RequestAbortedError(
                            "Request aborted during DFlash generation"
                        )

                yield GenerationOutput(
                    text=total_text,
                    new_text=new_text,
                    tokens=new_tokens,
                    prompt_tokens=prompt_len,
                    completion_tokens=total_completion,
                    finished=finished,
                    finish_reason=finish_reason,
                )

                if finished:
                    break
        finally:
            self._exit_active()

    async def chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        tools: list[dict] | None = None,
        request_id: str | None = None,
        **kwargs,
    ) -> GenerationOutput:
        if not self._loaded:
            await self.start()

        template_tools = convert_tools_for_template(tools) if tools else None
        ct_kwargs = kwargs.pop("chat_template_kwargs", None)
        prompt = self._apply_chat_template(
            messages, template_tools, chat_template_kwargs=ct_kwargs
        )

        return await self.generate(
            prompt=prompt, max_tokens=max_tokens, temperature=temperature,
            top_p=top_p, top_k=top_k, min_p=min_p,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            request_id=request_id, **kwargs,
        )

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        tools: list[dict] | None = None,
        request_id: str | None = None,
        **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        if not self._loaded:
            await self.start()

        template_tools = convert_tools_for_template(tools) if tools else None
        ct_kwargs = kwargs.pop("chat_template_kwargs", None)
        prompt = self._apply_chat_template(
            messages, template_tools, chat_template_kwargs=ct_kwargs
        )

        async for output in self.stream_generate(
            prompt=prompt, max_tokens=max_tokens, temperature=temperature,
            top_p=top_p, top_k=top_k, min_p=min_p,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            request_id=request_id, **kwargs,
        ):
            yield output

    @property
    def scheduler(self):
        """Expose the fallback scheduler when in fallback mode.

        ProcessMemoryEnforcer uses ``engine.scheduler`` to propagate
        memory limits for inline prefill checking.  DFlash itself
        manages cache inside dflash-mlx and has no scheduler, so we
        return ``None`` until the engine has fallen over to BatchedEngine
        / VLMBatchedEngine — at which point the inner scheduler becomes
        visible (and is the only one that needs the limit, since DFlash
        prefill is capped at DFLASH_MAX_CTX).
        """
        if self._in_fallback_mode and self._fallback_engine is not None:
            return self._fallback_engine.scheduler
        return None

    def has_active_requests(self) -> bool:
        """True if either the DFlash native path or the fallback engine
        has work in flight.  Used by EnginePool's drain monitor and the
        admin dashboard's busy gauge."""
        if (self._fallback_engine is not None
                and self._fallback_engine.has_active_requests()):
            return True
        with self._active_request_lock:
            return self._active_request_id is not None

    async def abort_request(self, request_id: str) -> bool:
        """Abort the in-flight request matching this id, if any.

        Forwarded to the fallback engine when in fallback mode (the
        request id may live there).  In native DFlash mode we set the
        per-call abort_event, which the executor-thread loop in
        _run_generate_streaming polls between events.  Returns True if
        a matching request was found and signalled.
        """
        if self._in_fallback_mode and self._fallback_engine is not None:
            forward = getattr(self._fallback_engine, "abort_request", None)
            if forward is not None:
                return await forward(request_id)
            return False

        with self._active_request_lock:
            if self._active_request_id == request_id:
                self._abort_event.set()
                return True
        return False

    async def abort_all_requests(self) -> int:
        """Abort every in-flight request and refuse new ones (terminal).

        Called by ProcessMemoryEnforcer on memory pressure.  Sets the
        terminal abort flag so a request arriving after this call is
        rejected at the entry-point check before touching dflash-mlx,
        and signals the per-call abort_event so the executor loop
        breaks out at its next checkpoint.  Forwards to the fallback
        engine too — both surfaces may have requests.
        """
        count = 0
        if (self._fallback_engine is not None
                and hasattr(self._fallback_engine, "abort_all_requests")):
            try:
                count += await self._fallback_engine.abort_all_requests()
            except Exception as exc:
                logger.warning(
                    f"DFlash fallback abort_all_requests failed: {exc}"
                )

        self._aborted_terminal = True
        with self._active_request_lock:
            if self._active_request_id is not None:
                self._abort_event.set()
                count += 1
        return count

    def get_stats(self) -> dict[str, Any]:
        return {
            "engine_type": "dflash",
            "model_name": self._model_name,
            "draft_model": self._draft_model_path,
            "max_dflash_ctx": self._max_dflash_ctx,
            "fallback_engine_type": self._fallback_engine_type,
            "in_fallback_mode": self._in_fallback_mode,
            "loaded": self._loaded,
        }

    def get_cache_stats(self) -> dict[str, Any] | None:
        if self._fallback_engine is not None:
            return self._fallback_engine.get_cache_stats()
        return None
