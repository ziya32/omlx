# SPDX-License-Identifier: Apache-2.0
"""Model-free tests for the server-wide idle tracker (Fix E.1).

omlx is the single chokepoint every model request flows through (gateway-
leased OR direct), so its idle state is the authoritative "is the model server
actually idle?" signal the gateway admission gate confirms before admitting a
ZONE/DREAM cycle.

These tests exercise:
  * ``/v1/idle`` reports ``idle_seconds`` that grows while no inference runs,
    and ``active_requests`` / ``busy`` from the engine pool.
  * Simulating an inference (the finalizers ``use_engine`` /
    ``_with_engine_guard``, and the audio completion hook) refreshes
    ``_last_inference_at``.
  * Querying the idle endpoint does NOT reset ``_last_inference_at`` — only
    inference does (mirrors §D5's "/health is exempt from the lease").
  * The endpoint is auth-exempt (200 with no Bearer even when a key is set).

Everything is mocked — no real model, no real engine, no network.
"""

import asyncio
from contextlib import suppress
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import omlx.server as server
from omlx.server import ServerState, app

TEST_API_KEY = "test-api-key"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pool_with_active(n_active: int, n_leases: int | None = None):
    """Build a MagicMock engine pool.

    ``n_active`` requests are reported as actively COMPUTING via each engine's
    ``_output_collectors`` (the in-scheduler signal the ``/api/status`` walk
    aggregates). ``n_leases`` is the pool-wide ``total_active_uses`` — engine
    leases held OR awaited — and defaults to ``n_active``. Pass
    ``n_leases > n_active`` to simulate requests **blocked in get_engine**
    (lease acquired before the ``get_engine`` await, so they hold a lease but
    populate no collector and bump no ``_in_flight``)."""
    core = MagicMock(spec=["_output_collectors", "scheduler"])
    core._output_collectors = {f"req-{i}": None for i in range(n_active)}
    core.scheduler = None

    async_core = MagicMock(spec=["engine"])
    async_core.engine = core

    engine = MagicMock(spec=["_engine"])
    engine._engine = async_core

    entry = MagicMock(spec=["is_loading", "engine"])
    entry.is_loading = False
    entry.engine = engine

    pool = MagicMock(spec=["_entries", "total_active_uses"])
    pool._entries = {"model-a": entry}
    pool.total_active_uses = n_active if n_leases is None else n_leases
    return pool


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def server_state():
    """Fresh server state per test (no api_key unless a test sets one)."""
    state = ServerState()
    with patch("omlx.server._server_state", state):
        yield state


@pytest.fixture(autouse=True)
def _reset_in_flight():
    """Reset the module-level in-flight counter around every test.

    ``_in_flight_requests`` is a process-global; without this a leaked count
    from one test (or a balanced enter/exit that left it at a non-zero
    baseline) would bleed into the next. Asserts it is balanced at teardown so
    a missing decrement anywhere is caught as a test failure, not silently
    carried forward."""
    server._in_flight_requests = 0
    yield
    assert server._in_flight_requests == 0, (
        "in-flight counter leaked: " f"{server._in_flight_requests}"
    )


# ---------------------------------------------------------------------------
# /v1/idle endpoint
# ---------------------------------------------------------------------------


def test_idle_endpoint_reports_active_and_idle_seconds(client, server_state):
    """idle_seconds grows while no inference runs; active/busy come from the
    engine pool's in-flight count."""
    # No pool yet: active_requests == 0, not busy.
    base = 1000.0
    with patch("omlx.server._last_inference_at", base), patch(
        "omlx.server.time.monotonic", return_value=base + 42.0
    ):
        resp = client.get("/v1/idle")
    assert resp.status_code == 200
    data = resp.json()
    assert data["idle_seconds"] == pytest.approx(42.0)
    assert data["active_requests"] == 0
    assert data["busy"] is False

    # Advance the clock further with no inference: idle_seconds keeps growing.
    with patch("omlx.server._last_inference_at", base), patch(
        "omlx.server.time.monotonic", return_value=base + 100.0
    ):
        later = client.get("/v1/idle").json()
    assert later["idle_seconds"] == pytest.approx(100.0)
    assert later["idle_seconds"] > data["idle_seconds"]


def test_idle_endpoint_busy_when_active_requests(client, server_state):
    """busy=True and active_requests>0 when engines have in-flight work."""
    server_state.engine_pool = _pool_with_active(3)
    resp = client.get("/v1/idle")
    assert resp.status_code == 200
    data = resp.json()
    assert data["active_requests"] == 3
    assert data["busy"] is True
    # Even with a fresh inference timestamp, busy stays True while in-flight.
    assert data["idle_seconds"] >= 0.0


def test_idle_endpoint_not_busy_when_no_active_requests(client, server_state):
    """busy=False when the pool exists but nothing is in flight."""
    server_state.engine_pool = _pool_with_active(0)
    data = client.get("/v1/idle").json()
    assert data["active_requests"] == 0
    assert data["busy"] is False


def test_idle_endpoint_is_auth_exempt(client, server_state):
    """The probe must never 401: 200 with no Bearer even when a key is set."""
    server_state.api_key = "a-real-secret-key"
    # No Authorization header at all.
    resp = client.get("/v1/idle")
    assert resp.status_code == 200
    assert "idle_seconds" in resp.json()
    # A bogus key is also fine (endpoint doesn't check auth at all).
    resp2 = client.get("/v1/idle", headers={"Authorization": "Bearer wrong"})
    assert resp2.status_code == 200


def test_health_carries_idle_snapshot(client, server_state):
    """/health (also auth-exempt) surfaces the same idle snapshot."""
    # No engine pool → health's pool_status is None, but the idle snapshot is
    # always present (active_requests defaults to 0).
    server_state.engine_pool = None
    data = client.get("/health").json()
    assert data["status"] == "healthy"
    assert "idle" in data
    assert data["idle"]["active_requests"] == 0
    assert data["idle"]["busy"] is False
    assert "idle_seconds" in data["idle"]


# ---------------------------------------------------------------------------
# Probing idle must NOT count as inference
# ---------------------------------------------------------------------------


def test_idle_probe_does_not_reset_idle(client, server_state):
    """Hitting /v1/idle (and /health) must not touch _last_inference_at."""
    before = server.get_idle_snapshot  # sanity: helper exists
    assert callable(before)

    with patch("omlx.server.record_inference_activity") as rec:
        client.get("/v1/idle")
        client.get("/health")
    rec.assert_not_called()

    # End-to-end: with a frozen last-inference time, repeated probes report a
    # monotonically growing idle_seconds (never reset to ~0 by the probe).
    base = 5000.0
    with patch("omlx.server._last_inference_at", base):
        with patch("omlx.server.time.monotonic", return_value=base + 10.0):
            first = client.get("/v1/idle").json()["idle_seconds"]
        with patch("omlx.server.time.monotonic", return_value=base + 20.0):
            second = client.get("/v1/idle").json()["idle_seconds"]
    assert first == pytest.approx(10.0)
    assert second == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# record_inference_activity / get_idle_snapshot units
# ---------------------------------------------------------------------------


def test_record_inference_activity_advances_timestamp():
    """record_inference_activity() bumps _last_inference_at to 'now'."""
    with patch("omlx.server.time.monotonic", return_value=111.0):
        server.record_inference_activity()
    assert server._last_inference_at == 111.0

    with patch("omlx.server.time.monotonic", return_value=999.0):
        server.record_inference_activity()
    assert server._last_inference_at == 999.0


def test_get_idle_snapshot_resets_idle_after_inference(server_state):
    """Simulating an inference shrinks idle_seconds on the next snapshot."""
    server_state.engine_pool = None
    # 100s of idle, then an inference at t=200 → idle drops back to ~0.
    with patch("omlx.server._last_inference_at", 100.0), patch(
        "omlx.server.time.monotonic", return_value=200.0
    ):
        stale = server.get_idle_snapshot()
    assert stale["idle_seconds"] == pytest.approx(100.0)

    with patch("omlx.server.time.monotonic", return_value=200.0):
        server.record_inference_activity()  # _last_inference_at = 200.0
        fresh = server.get_idle_snapshot()
    assert fresh["idle_seconds"] == pytest.approx(0.0)


def test_idle_seconds_never_negative(server_state):
    """Clamp idle_seconds at 0 even if the clock appears to move backwards."""
    with patch("omlx.server._last_inference_at", 500.0), patch(
        "omlx.server.time.monotonic", return_value=499.0
    ):
        snap = server.get_idle_snapshot()
    assert snap["idle_seconds"] == 0.0


def test_count_active_requests_handles_partial_engines(server_state):
    """_count_active_requests tolerates None engines / missing cores."""
    # Pool is None → 0.
    server_state.engine_pool = None
    assert server._count_active_requests() == 0

    # Entry with engine=None contributes nothing; a populated one counts.
    empty = MagicMock(spec=["engine"])
    empty.engine = None

    # Entry whose engine has no async core (_engine is None) → skipped.
    no_async = MagicMock(spec=["_engine"])
    no_async._engine = None
    entry_no_async = MagicMock(spec=["engine"])
    entry_no_async.engine = no_async

    # Entry whose async core has no inner engine (.engine is None) → skipped.
    async_core_no_core = MagicMock(spec=["engine"])
    async_core_no_core.engine = None
    engine_no_core = MagicMock(spec=["_engine"])
    engine_no_core._engine = async_core_no_core
    entry_no_core = MagicMock(spec=["engine"])
    entry_no_core.engine = engine_no_core

    full = _pool_with_active(4)._entries["model-a"]
    pool = MagicMock(spec=["_entries", "total_active_uses"])
    pool._entries = {
        "empty": empty,
        "no-async": entry_no_async,
        "no-core": entry_no_core,
        "model-a": full,
    }
    pool.total_active_uses = 0  # isolate the collector walk (the partial-engine path)
    server_state.engine_pool = pool
    assert server._count_active_requests() == 4


# ---------------------------------------------------------------------------
# Finalizers refresh the tracker (the END-of-inference hook)
# ---------------------------------------------------------------------------


def test_with_engine_guard_records_inference_on_stream_finish():
    """_with_engine_guard bumps the idle tracker when the stream finishes."""
    pool = MagicMock(spec=["release_engine"])

    async def _gen():
        yield "a"
        yield "b"

    async def _drive():
        with patch("omlx.server.record_inference_activity") as rec:
            body = server._with_engine_guard(_gen(), pool, "model-a")
            chunks = [c async for c in body]
            return chunks, rec

    chunks, rec = asyncio.run(_drive())
    assert chunks == ["a", "b"]
    pool.release_engine.assert_called_once_with("model-a")
    rec.assert_called_once()


def test_with_engine_guard_records_inference_on_stream_error():
    """A mid-stream error still counts as inference (it ran, then failed)."""
    pool = MagicMock(spec=["release_engine"])

    async def _gen():
        yield "a"
        raise RuntimeError("boom")

    async def _drive():
        with patch("omlx.server.record_inference_activity") as rec:
            body = server._with_engine_guard(_gen(), pool, "model-a")
            with suppress(RuntimeError):
                async for _ in body:
                    pass
            return rec

    rec = asyncio.run(_drive())
    rec.assert_called_once()
    pool.release_engine.assert_called_once_with("model-a")


def test_with_engine_guard_does_not_record_when_never_iterated():
    """The never-iterated (client-disconnect) path releases the lease but is
    NOT inference — record_inference_activity must not fire there."""
    pool = MagicMock(spec=["release_engine"])

    async def _gen():
        yield "a"

    import gc

    with patch("omlx.server.record_inference_activity") as rec:
        body = server._with_engine_guard(_gen(), pool, "model-a")
        # Trigger the weakref finalizer without iterating the body (the
        # client-disconnect-before-stream path).
        del body
        gc.collect()
    # Lease still released by the finalizer, but no inference was recorded.
    pool.release_engine.assert_called_once_with("model-a")
    rec.assert_not_called()


def test_use_engine_records_inference_on_exit():
    """use_engine bumps the idle tracker when its block finishes."""
    pool = MagicMock(
        spec=["resolve_model_id", "acquire_engine", "release_engine"]
    )
    pool.resolve_model_id.return_value = "model-a"

    fake_engine = object()

    async def _fake_get_engine(model_id, engine_type, resolved_id=None):
        return fake_engine

    async def _drive():
        with patch("omlx.server.get_engine_pool", return_value=pool), patch(
            "omlx.server.get_engine", _fake_get_engine
        ), patch("omlx.server.record_inference_activity") as rec:
            async with server.use_engine("model-a", server.EngineType.RERANKER) as eng:
                assert eng is fake_engine
            return rec

    rec = asyncio.run(_drive())
    pool.acquire_engine.assert_called_once_with("model-a")
    pool.release_engine.assert_called_once_with("model-a")
    rec.assert_called_once()


# ---------------------------------------------------------------------------
# Audio (TTS/ASR/process) completion hook refreshes the tracker
# ---------------------------------------------------------------------------


def test_audio_record_request_refreshes_idle_tracker():
    """_record_audio_request (called by every audio inference path) bumps the
    server-wide idle tracker."""
    from omlx.api import audio_routes

    with patch("omlx.server.record_inference_activity") as rec, patch(
        "omlx.server_metrics.get_server_metrics"
    ):
        audio_routes._record_audio_request("whisper-large-v3")
    rec.assert_called_once()


def test_audio_record_request_bumps_idle_even_if_metrics_fail():
    """If server-metrics recording raises, the idle tracker is STILL bumped
    (the two are independent — a metrics blip must not hide audio activity
    from the idle gate)."""
    from omlx.api import audio_routes

    with patch(
        "omlx.server_metrics.get_server_metrics", side_effect=RuntimeError("metrics down")
    ), patch("omlx.server.record_inference_activity") as rec:
        # Must not raise despite the metrics failure.
        audio_routes._record_audio_request("whisper-large-v3")
    rec.assert_called_once()


# ---------------------------------------------------------------------------
# NON-STREAMING in-flight counting (Fix E.1, Item 2)
#
# The collector walk only sees a request during its transient scheduler step.
# A non-streaming chat/embedding/rerank request must still read as busy for its
# WHOLE duration via the explicit ``_in_flight_requests`` counter, so the
# dream/zone idle gate can never fire while one is mid-flight.
# ---------------------------------------------------------------------------


def test_enter_exit_inference_balance(server_state):
    """_enter/_exit move the in-flight counter and the snapshot reflects it,
    with NO loaded engine pool (so _output_collectors contributes nothing —
    this isolates the explicit counter)."""
    server_state.engine_pool = None
    assert server._count_active_requests() == 0

    server._enter_inference()
    # Counter alone (collectors empty) drives active_requests/busy.
    assert server._count_active_requests() == 1
    snap = server.get_idle_snapshot()
    assert snap["active_requests"] == 1
    assert snap["busy"] is True

    server._exit_inference()
    assert server._count_active_requests() == 0
    assert server.get_idle_snapshot()["busy"] is False


def test_exit_inference_clamps_at_zero(server_state):
    """A stray extra _exit_inference can never drive the counter negative
    (which would later mask a genuinely-busy server as idle)."""
    server_state.engine_pool = None
    server._exit_inference()  # underflow attempt with counter already 0
    assert server._in_flight_requests == 0
    assert server._count_active_requests() == 0

    # And a balanced pair still lands exactly on 0 afterward.
    server._enter_inference()
    server._exit_inference()
    server._exit_inference()  # extra exit — still clamped
    assert server._in_flight_requests == 0


def test_count_active_requests_takes_max_not_sum(server_state):
    """A single non-streaming token request appears in BOTH signals at once
    (the counter for its whole life + a collector during its scheduler step).
    _count_active_requests must report 1 (max), never 2 (sum)."""
    # One engine reporting one in-scheduler collector, AND one explicit
    # in-flight increment — same logical request seen twice.
    server_state.engine_pool = _pool_with_active(1)
    server._enter_inference()
    try:
        assert server._count_active_requests() == 1  # max(1, 1), not 1+1
    finally:
        server._exit_inference()

    # Many distinct requests: each bumps the counter, so the counter (≥ the
    # collector total) wins and reports true concurrency.
    server_state.engine_pool = _pool_with_active(1)
    server._enter_inference()
    server._enter_inference()
    server._enter_inference()
    try:
        # 3 explicit in-flight vs 1 collector → 3.
        assert server._count_active_requests() == 3
    finally:
        server._exit_inference()
        server._exit_inference()
        server._exit_inference()


def test_use_engine_marks_in_flight_during_non_streaming(server_state):
    """A non-streaming request held inside use_engine reads active_requests>0
    DURING the block and exactly 0 AFTER (the embedding/rerank/token-count
    path that never populates _output_collectors)."""
    server_state.engine_pool = None  # no collectors — counter is the only signal
    pool = MagicMock(
        spec=["resolve_model_id", "acquire_engine", "release_engine"]
    )
    pool.resolve_model_id.return_value = "model-a"
    fake_engine = object()

    async def _fake_get_engine(model_id, engine_type, resolved_id=None):
        return fake_engine

    observed = {}

    async def _drive():
        with patch("omlx.server.get_engine_pool", return_value=pool), patch(
            "omlx.server.get_engine", _fake_get_engine
        ):
            async with server.use_engine(
                "model-a", server.EngineType.EMBEDDING
            ) as eng:
                assert eng is fake_engine
                # In-flight WHILE the non-streaming op runs.
                observed["during"] = server._count_active_requests()
                observed["busy_during"] = server.get_idle_snapshot()["busy"]
            observed["after"] = server._count_active_requests()

    asyncio.run(_drive())
    assert observed["during"] == 1
    assert observed["busy_during"] is True
    assert observed["after"] == 0
    pool.acquire_engine.assert_called_once_with("model-a")
    pool.release_engine.assert_called_once_with("model-a")


def test_use_engine_drops_in_flight_on_error(server_state):
    """If the body inside use_engine raises, the in-flight count returns to 0
    (try/finally — no phantom-busy leak that would wedge the idle gate)."""
    server_state.engine_pool = None
    pool = MagicMock(
        spec=["resolve_model_id", "acquire_engine", "release_engine"]
    )
    pool.resolve_model_id.return_value = "model-a"

    async def _fake_get_engine(model_id, engine_type, resolved_id=None):
        return object()

    async def _drive():
        with patch("omlx.server.get_engine_pool", return_value=pool), patch(
            "omlx.server.get_engine", _fake_get_engine
        ):
            with pytest.raises(RuntimeError, match="boom"):
                async with server.use_engine("model-a", server.EngineType.LLM):
                    assert server._count_active_requests() == 1
                    raise RuntimeError("boom")

    asyncio.run(_drive())
    assert server._count_active_requests() == 0
    pool.release_engine.assert_called_once_with("model-a")


def test_use_engine_drops_in_flight_when_get_engine_raises(server_state):
    """If get_engine itself raises (load failure), the finally still releases
    the in-flight count — the increment happens before the await, so the
    decrement must run even though the body never executed."""
    server_state.engine_pool = None
    pool = MagicMock(
        spec=["resolve_model_id", "acquire_engine", "release_engine"]
    )
    pool.resolve_model_id.return_value = "model-a"

    async def _boom_get_engine(model_id, engine_type, resolved_id=None):
        raise RuntimeError("load failed")

    async def _drive():
        with patch("omlx.server.get_engine_pool", return_value=pool), patch(
            "omlx.server.get_engine", _boom_get_engine
        ):
            with pytest.raises(RuntimeError, match="load failed"):
                async with server.use_engine("model-a", server.EngineType.LLM):
                    pass

    asyncio.run(_drive())
    assert server._count_active_requests() == 0
    pool.release_engine.assert_called_once_with("model-a")


def test_with_engine_guard_marks_in_flight_while_streaming(server_state):
    """_with_engine_guard (used by BOTH true streams and the StreamingResponse-
    wrapped non-streaming chat build) reads busy WHILE its body is iterating,
    and 0 once exhausted."""
    server_state.engine_pool = None
    pool = MagicMock(spec=["release_engine"])
    seen = []

    async def _gen():
        # Record the in-flight count from inside the running stream.
        seen.append(server._count_active_requests())
        yield "a"
        seen.append(server._count_active_requests())
        yield "b"

    async def _drive():
        body = server._with_engine_guard(_gen(), pool, "model-a")
        return [c async for c in body]

    chunks = asyncio.run(_drive())
    assert chunks == ["a", "b"]
    # Busy (==1) at every point while the body was running.
    assert seen == [1, 1]
    # Drops back to 0 after the generator is exhausted.
    assert server._count_active_requests() == 0
    pool.release_engine.assert_called_once_with("model-a")


def test_with_engine_guard_drops_in_flight_on_stream_error(server_state):
    """A mid-stream error still decrements the in-flight counter (try/finally),
    so a failed request can't leave the server falsely 'busy'."""
    server_state.engine_pool = None
    pool = MagicMock(spec=["release_engine"])

    async def _gen():
        yield "a"
        raise RuntimeError("boom")

    async def _drive():
        body = server._with_engine_guard(_gen(), pool, "model-a")
        with suppress(RuntimeError):
            async for _ in body:
                assert server._count_active_requests() == 1
    asyncio.run(_drive())
    assert server._count_active_requests() == 0
    pool.release_engine.assert_called_once_with("model-a")


def test_with_engine_guard_never_iterated_does_not_count(server_state):
    """The never-iterated (client-disconnect-before-stream) path releases the
    lease via the weakref finalizer but must NOT touch the in-flight counter —
    no tokens ran, so it is not inference and the counter stays balanced."""
    server_state.engine_pool = None
    pool = MagicMock(spec=["release_engine"])

    async def _gen():
        yield "a"

    import gc

    body = server._with_engine_guard(_gen(), pool, "model-a")
    # Never iterate; trigger the finalizer by dropping the body.
    del body
    gc.collect()
    # Lease released by the finalizer, but the counter was never bumped.
    pool.release_engine.assert_called_once_with("model-a")
    assert server._count_active_requests() == 0


def test_idle_endpoint_busy_for_non_streaming_in_flight(client, server_state):
    """End-to-end through the HTTP route: while a non-streaming request is in
    flight (no scheduler collectors at all), GET /v1/idle reports
    active_requests>0 and busy=True — the exact regression Item 2 fixes."""
    server_state.engine_pool = None
    # Simulate a non-streaming request mid-flight by entering the in-flight
    # state the same way use_engine/_with_engine_guard do.
    server._enter_inference()
    try:
        data = client.get("/v1/idle").json()
        assert data["active_requests"] == 1
        assert data["busy"] is True
        # /health surfaces the same snapshot.
        health = client.get("/health").json()
        assert health["idle"]["busy"] is True
    finally:
        server._exit_inference()

    after = client.get("/v1/idle").json()
    assert after["active_requests"] == 0
    assert after["busy"] is False


def test_concurrent_in_flight_counts_are_correct(server_state):
    """Many overlapping non-streaming requests report the true concurrency,
    and decrement back to 0 cleanly (correct under concurrency)."""
    server_state.engine_pool = None

    async def _one(hold: float):
        server._enter_inference()
        try:
            await asyncio.sleep(hold)
        finally:
            server._exit_inference()

    peak = {"value": 0}

    async def _watch():
        # Sample the count repeatedly while the workers overlap.
        for _ in range(20):
            peak["value"] = max(peak["value"], server._count_active_requests())
            await asyncio.sleep(0.001)

    async def _drive():
        tasks = [asyncio.create_task(_one(0.01)) for _ in range(5)]
        watcher = asyncio.create_task(_watch())
        await asyncio.gather(*tasks, watcher)

    asyncio.run(_drive())
    assert peak["value"] == 5  # all five were counted simultaneously
    assert server._count_active_requests() == 0


# ---------------------------------------------------------------------------
# BLOCKED-IN-GET_ENGINE counting (fix-dreaming follow-up)
#
# acquire_engine (active_uses) fires BEFORE the get_engine await, but a request
# path's _enter_inference / _output_collectors only populate AFTER it. So a
# request blocked waiting for a loading / draining / exclusive-contended model
# holds an engine LEASE yet is invisible to _in_flight and the collector walk.
# /v1/idle must still read busy via the pool's total_active_uses, or the
# dream/zone gate fires into a wedged-but-not-computing server.
# ---------------------------------------------------------------------------


def test_total_active_uses_sums_leases_ignoring_pins():
    """The pool property sums every entry's active_uses; an idle PINNED model
    contributes 0 (pinning is a separate flag — no baseline to subtract)."""
    from omlx.engine_pool import EngineEntry, EnginePool

    def _entry(model_id, active_uses, pinned=False):
        return EngineEntry(
            model_id=model_id, model_path=f"/x/{model_id}",
            model_type="embedding", engine_type="embedding",
            estimated_size=0, active_uses=active_uses, is_pinned=pinned,
        )

    pool = EnginePool.__new__(EnginePool)  # bypass the model-dir scan; test the property
    pool._entries = {
        "busy": _entry("busy", 2),
        "pinned-idle": _entry("pinned-idle", 0, pinned=True),
        "one": _entry("one", 1),
    }
    assert pool.total_active_uses == 3  # 2 + 0 + 1 — the pin adds nothing
    pool._entries = {}
    assert pool.total_active_uses == 0


def test_count_active_requests_counts_blocked_in_get_engine(server_state):
    """A request BLOCKED in get_engine holds a lease (active_uses>0) but has not
    started computing — no _output_collectors, no _in_flight. The server must
    STILL report it via total_active_uses, else /v1/idle reads idle while the
    server is wedged (the regression that let a dream fire into a busy omlx)."""
    server_state.engine_pool = _pool_with_active(n_active=0, n_leases=2)
    assert server._in_flight_requests == 0
    assert server._count_active_requests() == 2
    snap = server.get_idle_snapshot()
    assert snap["active_requests"] == 2
    assert snap["busy"] is True


def test_idle_endpoint_busy_for_blocked_acquire(client, server_state):
    """End-to-end through HTTP: a request blocked waiting to ACQUIRE a model
    reads busy via GET /v1/idle (and /health) even though nothing computes."""
    server_state.engine_pool = _pool_with_active(n_active=0, n_leases=1)
    data = client.get("/v1/idle").json()
    assert data["active_requests"] == 1
    assert data["busy"] is True


def test_count_active_requests_max_includes_active_uses(server_state):
    """total_active_uses is the authoritative superset: one request seen as a
    lease AND a collector AND _in_flight stays 1 (max, not sum); and leases win
    when they exceed the computing signals (1 computing + 2 blocked-in-acquire)."""
    server_state.engine_pool = _pool_with_active(n_active=1, n_leases=1)
    server._enter_inference()
    try:
        assert server._count_active_requests() == 1
    finally:
        server._exit_inference()

    server_state.engine_pool = _pool_with_active(n_active=1, n_leases=3)
    assert server._count_active_requests() == 3
