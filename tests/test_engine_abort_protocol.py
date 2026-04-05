# SPDX-License-Identifier: Apache-2.0
"""Tests for the cooperative abort protocol on BaseNonStreamingEngine.

Issue 2 fix: every non-streaming engine (EmbeddingEngine, RerankerEngine,
STTEngine, TTSEngine, STSEngine) inherits ``abort_all_requests`` and
``_raise_if_aborted`` from BaseNonStreamingEngine, matching the contract
BatchedEngine/VLMBatchedEngine already provided. The enforcer can treat
every engine uniformly — no more hasattr guard.

Tests are deterministic: they coordinate via asyncio.Event and direct
state inspection, not sleeps or polling.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from omlx.engine.base import BaseNonStreamingEngine
from omlx.exceptions import RequestAbortedError


class _FakeEngine(BaseNonStreamingEngine):
    """Minimal concrete subclass used to exercise the abort protocol.

    Mirrors the shape of EmbeddingEngine / RerankerEngine: single public
    ``work()`` entry point that submits a blocking closure to a fake
    executor, then inspects ``_raise_if_aborted`` before returning the
    result.
    """

    def __init__(self, name: str = "fake-engine"):
        super().__init__()
        self._model_name = name
        self._started = False
        # asyncio.Event the test uses to release the "in-flight" work
        # deterministically from another task.
        self._work_released = asyncio.Event()
        # Latch so the test can observe when work() has entered the
        # "in-flight" region (i.e. awaiting the executor).
        self._work_in_flight = asyncio.Event()

    @property
    def model_name(self) -> str:
        return self._model_name

    async def start(self) -> None:
        self._started = True

    async def stop(self) -> None:
        self._started = False

    def get_stats(self) -> Dict[str, Any]:
        return {"model_name": self._model_name, "started": self._started}

    async def work(self) -> str:
        """Public entry point that mirrors real engines' shape.

        Pre-check → record in-flight → await a suspension point the test
        controls → post-check → return result. The abort flag may be
        set at either checkpoint.
        """
        if not self._started:
            raise RuntimeError("Engine not started. Call start() first.")
        self._raise_if_aborted()

        with self._active_lock:
            self._active_count += 1
        try:
            self._work_in_flight.set()
            # This await is the deterministic stand-in for
            # "loop.run_in_executor(get_mlx_executor(), _work_sync)".
            # The test drives the release.
            await self._work_released.wait()
            # After the executor returns, check the abort flag and
            # discard the result if the enforcer fired while we were
            # waiting.
            self._raise_if_aborted()
            return "work-result"
        finally:
            with self._active_lock:
                self._active_count -= 1


@pytest.fixture
async def started_fake_engine():
    engine = _FakeEngine()
    await engine.start()
    yield engine
    await engine.stop()


class TestBaseNonStreamingEngineAbortAPI:
    """Unit tests for the base class's abort primitive."""

    @pytest.mark.asyncio
    async def test_abort_all_requests_is_available_on_every_subclass(self):
        """BaseNonStreamingEngine.abort_all_requests is inherited by every
        concrete non-streaming engine — verified via MRO, not by calling
        each real constructor (which would require real models)."""
        from omlx.engine.embedding import EmbeddingEngine
        from omlx.engine.reranker import RerankerEngine
        from omlx.engine.stt import STTEngine
        from omlx.engine.sts import STSEngine
        from omlx.engine.tts import TTSEngine

        for cls in (
            EmbeddingEngine,
            RerankerEngine,
            STTEngine,
            STSEngine,
            TTSEngine,
        ):
            assert issubclass(cls, BaseNonStreamingEngine), (
                f"{cls.__name__} must inherit from BaseNonStreamingEngine "
                f"to get the uniform abort protocol"
            )
            method = getattr(cls, "abort_all_requests", None)
            assert method is not None, (
                f"{cls.__name__} missing abort_all_requests"
            )
            # Must resolve to the base-class implementation (or an
            # intentional override), not be absent.
            assert asyncio.iscoroutinefunction(method), (
                f"{cls.__name__}.abort_all_requests must be async"
            )

    @pytest.mark.asyncio
    async def test_abort_returns_active_count_and_sets_flag(self, started_fake_engine):
        """abort_all_requests captures the current active_count and sets
        the internal event."""
        engine = started_fake_engine

        assert engine.has_active_requests() is False
        assert engine._aborted.is_set() is False

        # Simulate 3 in-flight operations via direct counter mutation
        # (matches what the real engines do inside their _active_lock).
        with engine._active_lock:
            engine._active_count = 3

        count = await engine.abort_all_requests()

        assert count == 3
        assert engine._aborted.is_set() is True

    @pytest.mark.asyncio
    async def test_raise_if_aborted_noop_before_abort(self, started_fake_engine):
        """Before abort, _raise_if_aborted is a no-op."""
        started_fake_engine._raise_if_aborted()  # must not raise

    @pytest.mark.asyncio
    async def test_raise_if_aborted_raises_typed_error_after_abort(
        self, started_fake_engine
    ):
        """After abort, _raise_if_aborted raises RequestAbortedError
        with a message naming the engine."""
        await started_fake_engine.abort_all_requests()
        with pytest.raises(RequestAbortedError, match="fake-engine"):
            started_fake_engine._raise_if_aborted()


class TestCooperativeAbortBoundaries:
    """Tests that work() observes abort at both checkpoint boundaries."""

    @pytest.mark.asyncio
    async def test_pre_submit_abort_raises_before_executor_submission(
        self, started_fake_engine
    ):
        """If the engine is already aborted when the caller enters the
        public entry point, the pre-submit _raise_if_aborted check trips
        before any work is submitted — the in-flight latch must NOT fire.
        """
        engine = started_fake_engine
        await engine.abort_all_requests()

        with pytest.raises(RequestAbortedError):
            await engine.work()

        # The work function never reached the in-flight region.
        assert engine._work_in_flight.is_set() is False
        assert engine._active_count == 0

    @pytest.mark.asyncio
    async def test_in_flight_abort_discards_executor_result(
        self, started_fake_engine
    ):
        """Abort fires while work() is awaiting the executor. Once the
        "executor" completes, the post-submit _raise_if_aborted check
        trips and the result is discarded — the caller sees
        RequestAbortedError, not "work-result".
        """
        engine = started_fake_engine

        # Start work() as a background task so we can abort it mid-flight.
        work_task = asyncio.create_task(engine.work())

        # Deterministic synchronization: wait for work() to enter the
        # in-flight region (matches the real "executor future has been
        # submitted and is running" state).
        await engine._work_in_flight.wait()
        assert engine._active_count == 1

        # Fire the abort while work is in-flight.
        count = await engine.abort_all_requests()
        assert count == 1

        # Release the "executor" so work() resumes past its await.
        engine._work_released.set()

        # work() must surface the abort rather than returning the
        # completed result.
        with pytest.raises(RequestAbortedError, match="fake-engine"):
            await work_task

        # _active_count decrement still ran in finally.
        assert engine._active_count == 0

    @pytest.mark.asyncio
    async def test_abort_is_terminal_subsequent_calls_also_fail(
        self, started_fake_engine
    ):
        """The abort flag is not reset — a second work() after abort
        also raises immediately. (The engine is considered "dying" and
        will be unloaded by the enforcer; resetting the flag is
        explicitly out of scope per the base-class docstring.)
        """
        engine = started_fake_engine
        await engine.abort_all_requests()

        for _ in range(3):
            with pytest.raises(RequestAbortedError):
                await engine.work()


class TestEmbeddingEngineAbort:
    """End-to-end cooperative abort on the real EmbeddingEngine, with
    the MLX model dependency mocked out."""

    @pytest.mark.asyncio
    async def test_embed_after_abort_raises_request_aborted_error(self):
        """EmbeddingEngine.embed must call _raise_if_aborted on the
        pre-submit path — no executor work is done after abort."""
        from omlx.engine.embedding import EmbeddingEngine

        with patch(
            "omlx.engine.embedding.MLXEmbeddingModel"
        ) as mock_model_cls:
            mock_model = MagicMock()
            mock_model.load = MagicMock()
            mock_model.embed = MagicMock(return_value="unreachable")
            mock_model_cls.return_value = mock_model

            engine = EmbeddingEngine("fake-embed-model")
            # Bypass start() to avoid run_in_executor for model load.
            engine._model = mock_model

            await engine.abort_all_requests()

            with pytest.raises(RequestAbortedError, match="fake-embed-model"):
                await engine.embed(["hello"])

            # The underlying model.embed must NOT have been invoked —
            # the pre-submit _raise_if_aborted tripped first.
            mock_model.embed.assert_not_called()

    @pytest.mark.asyncio
    async def test_in_flight_embed_discards_result_on_abort(self):
        """Abort fires while the "executor" is running the embed. The
        closure returns its value, but EmbeddingEngine.embed must check
        the abort flag post-executor and raise instead of returning it.
        """
        from omlx.engine.embedding import EmbeddingEngine

        # Gate the executor submission on an asyncio.Event we control.
        release = asyncio.Event()
        in_flight = asyncio.Event()

        with patch(
            "omlx.engine.embedding.MLXEmbeddingModel"
        ) as mock_model_cls:
            mock_model = MagicMock()

            # Make model.embed a blocking call the test can release via
            # the asyncio.Event — wait() on a threading primitive would
            # coordinate, but asyncio.Event.wait is coroutine-only, so
            # use an asyncio.run_coroutine_threadsafe bridge instead.
            # Simpler: patch the executor submission entirely.
            mock_model.embed = MagicMock(return_value="real-result")
            mock_model_cls.return_value = mock_model

            engine = EmbeddingEngine("fake-embed-model")
            engine._model = mock_model

            # Patch run_in_executor to a controllable async gate.
            original_get_loop = asyncio.get_running_loop

            async def gated_executor(executor, func):
                in_flight.set()
                await release.wait()
                # Actually run the closure so the rest of embed() has a
                # real return value to inspect.
                return func()

            class _FakeLoop:
                def __init__(self, real):
                    self._real = real

                def __getattr__(self, name):
                    return getattr(self._real, name)

                def run_in_executor(self, executor, func):
                    return gated_executor(executor, func)

            def fake_get_running_loop():
                return _FakeLoop(original_get_loop())

            with patch(
                "omlx.engine.embedding.asyncio.get_running_loop",
                fake_get_running_loop,
            ):
                task = asyncio.create_task(engine.embed(["hello"]))
                await in_flight.wait()
                assert engine._active_count == 1

                # Abort mid-flight, then release the "executor".
                count = await engine.abort_all_requests()
                assert count == 1
                release.set()

                with pytest.raises(
                    RequestAbortedError, match="fake-embed-model"
                ):
                    await task

            # model.embed DID run (the closure completed), but its
            # result was discarded by the post-submit abort check.
            mock_model.embed.assert_called_once()
            assert engine._active_count == 0


class TestRerankerEngineAbort:
    """End-to-end cooperative abort on the real RerankerEngine."""

    @pytest.mark.asyncio
    async def test_rerank_after_abort_raises_request_aborted_error(self):
        from omlx.engine.reranker import RerankerEngine

        with patch(
            "omlx.engine.reranker.MLXRerankerModel"
        ) as mock_model_cls:
            mock_model = MagicMock()
            mock_model.load = MagicMock()
            mock_model.rerank = MagicMock(return_value="unreachable")
            mock_model_cls.return_value = mock_model

            engine = RerankerEngine("fake-rerank-model")
            engine._model = mock_model

            await engine.abort_all_requests()

            with pytest.raises(
                RequestAbortedError, match="fake-rerank-model"
            ):
                await engine.rerank(query="q", documents=["a", "b"])

            mock_model.rerank.assert_not_called()


class TestEngineSourceAbortCheckpoints:
    """Static invariant: every public MLX-bound entry point on every
    non-streaming engine must call _raise_if_aborted.

    This is a source-level scan rather than a runtime test. It stays
    green as long as every entry point that awaits run_in_executor
    surrounds that await with abort checks. It will flip red (and
    force a review) if someone adds a new entry point without the
    check, or removes an existing check during refactoring.
    """

    # (module path, method names that must contain _raise_if_aborted)
    CASES = [
        ("omlx/engine/embedding.py", ["embed"]),
        ("omlx/engine/reranker.py", ["rerank"]),
        ("omlx/engine/stt.py", ["transcribe"]),
        ("omlx/engine/tts.py", ["synthesize", "stream_synthesize"]),
        ("omlx/engine/sts.py", ["process"]),
    ]

    def test_every_public_entry_point_has_abort_check(self):
        import ast
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[1]
        missing: list[str] = []

        for rel_path, method_names in self.CASES:
            path = repo_root / rel_path
            source = path.read_text()
            tree = ast.parse(source, filename=str(path))

            for node in ast.walk(tree):
                if not isinstance(
                    node, (ast.AsyncFunctionDef, ast.FunctionDef)
                ):
                    continue
                if node.name not in method_names:
                    continue

                calls_abort_check = False
                for sub in ast.walk(node):
                    if (
                        isinstance(sub, ast.Call)
                        and isinstance(sub.func, ast.Attribute)
                        and sub.func.attr == "_raise_if_aborted"
                    ):
                        calls_abort_check = True
                        break
                if not calls_abort_check:
                    missing.append(f"{rel_path}::{node.name}")

        assert not missing, (
            "The following public entry points on non-streaming engines "
            "must call self._raise_if_aborted() at their MLX-executor "
            "boundaries, but no call was found:\n  "
            + "\n  ".join(missing)
            + "\nSee BaseNonStreamingEngine docstring for the contract."
        )
