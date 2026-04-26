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
import logging
import os
import threading
from collections.abc import AsyncIterator
from typing import Any

import mlx.core as mx

from ..api.tool_calling import convert_tools_for_template
from ..api.utils import clean_special_tokens, detect_and_strip_partial
from ..exceptions import RequestAbortedError
from .base import BaseEngine, GenerationOutput

logger = logging.getLogger(__name__)

DEFAULT_MAX_DFLASH_CTX = 4096


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
        model_settings: Any | None = None,
        fallback_engine_type: str = "batched",
        scheduler_config: Any | None = None,
        process_memory_max_bytes: int = 0,
    ):
        self._model_name = model_name
        self._draft_model_path = draft_model_path
        self._draft_quant_bits = draft_quant_bits
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

        raw = os.environ.get("DFLASH_MAX_CTX", str(DEFAULT_MAX_DFLASH_CTX)).strip()
        try:
            self._max_dflash_ctx = max(1, int(raw))
        except ValueError:
            self._max_dflash_ctx = DEFAULT_MAX_DFLASH_CTX

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
            from dflash_mlx.runtime import load_target_bundle, load_draft_bundle

            model, tokenizer, meta = load_target_bundle(self._model_name)
            draft, draft_meta = load_draft_bundle(
                self._draft_model_path,
                quantize_draft=bool(self._draft_quant_bits),
            )
            return model, tokenizer, meta, draft

        result = await loop.run_in_executor(get_mlx_executor(), _load_models)
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

        self._loaded = True
        self._in_fallback_mode = False
        logger.info(
            f"DFlashEngine loaded: target={self._model_name}, "
            f"draft={self._draft_model_path}, "
            f"max_ctx={self._max_dflash_ctx}, "
            f"fallback={self._fallback_engine_type}"
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
            lambda: (mx.synchronize(), mx.clear_cache()),
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
                lambda: (mx.synchronize(), mx.clear_cache()),
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
    ) -> str:
        if hasattr(self._tokenizer_obj, "apply_chat_template"):
            is_partial = detect_and_strip_partial(messages)
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
    ) -> int:
        template_tools = convert_tools_for_template(tools) if tools else None
        prompt = self._apply_chat_template(
            messages, template_tools, chat_template_kwargs=chat_template_kwargs
        )
        return len(self._tokenizer_obj.encode(prompt))

    def _should_fallback(self, prompt_tokens: list[int]) -> bool:
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

            generator = stream_dflash_generate(
                target_model=self._target_model,
                tokenizer=self._executor_tokenizer,
                draft_model=self._draft_model,
                prompt="",
                max_new_tokens=max_tokens,
                stop_token_ids=stop_ids,
                prompt_tokens_override=prompt_tokens,
                temperature=temperature,
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
        from dflash_mlx.runtime import generate_dflash_once

        loop = asyncio.get_running_loop()
        stop_ids = get_stop_token_ids(self._tokenizer_obj)

        def _run():
            return generate_dflash_once(
                target_model=self._target_model,
                tokenizer=self._executor_tokenizer,
                draft_model=self._draft_model,
                prompt="",
                max_new_tokens=max_tokens,
                stop_token_ids=stop_ids,
                prompt_tokens_override=prompt_tokens,
                temperature=temperature,
            )

        self._enter_active(request_id)
        try:
            summary = await loop.run_in_executor(get_mlx_executor(), _run)
            # Discard the result if abort fired while the executor was
            # running — handler sees the typed abort instead of stale text.
            if self._abort_event.is_set():
                raise RequestAbortedError(
                    f"Request aborted during DFlash generation"
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
                            f"Request aborted during DFlash generation"
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
