# Review: enforcer eviction hardening diff

**Branch:** `features/more-models`
**Reviewed state:** working-tree changes on top of `2ebfdca` (10 files, +510/-161)
**Reviewer notes:** all issues below are verified by deterministic tests committed alongside this document. Test names and failure output are embedded in each section.

## Scope of the reviewed change

The diff hardens the race where `ProcessMemoryEnforcer` can tear a model out from under an in-flight request handler. Five coordinated pieces:

1. **`EngineEvictedError`** added (`omlx/exceptions.py`) and a new pool method **`EnginePool.ensure_engine_alive(model_id, engine_ref)`** (`omlx/engine_pool.py:906-922`) that raises it when an entry was unloaded.
2. **`EngineCore.generate()`** now raises the existing typed **`RequestAbortedError`** on error outputs (`omlx/engine_core.py:605-610`), and a new FastAPI exception handler translates it to HTTP 503 (`omlx/server.py:582-604`).
3. **`EngineCore.stream_outputs()`** swaps `raise RuntimeError(output.error)` → `break` after yielding the error `RequestOutput` (`omlx/engine_core.py:515-524`), on the rationale that raising would "abort the HTTP connection after headers are flushed".
4. **`ProcessMemoryEnforcer._check_and_enforce()`** simplified: the old 3-scenario matrix (evict-idle / keep-single-model / abort-then-keep) collapses into a single *"abort + unload any non-pinned LRU victim until under limit"* loop, with **no `active_uses > 0` skip** (`omlx/process_memory_enforcer.py:200-263`).
5. Tests updated/added to reflect the new policy.

The shape is right — typed exceptions, explicit post-await liveness checks, defined 503 semantics. Seven concrete problems remain, all verified.

---

## Issue summary

| # | Issue | Verdict | Severity | Fix status |
|---|---|---|---|---|
| 1 | Enforcer cascades evictions within a single tick | **Confirmed** | High | **Fixed** (one eviction per tick) |
| 2 | Non-abortable engines torn down mid-operation regardless of `active_uses` | **Confirmed** | High | **Fixed** (uniform cooperative abort on `BaseNonStreamingEngine`) |
| 3 | Dead `except EngineEvictedError` around `pool.get_engine()` calls | **Confirmed** | Low (cleanliness) | **Fixed** (dead branches + tests deleted) |
| 4 | Stopped `BatchedEngine` raises plain `RuntimeError` → falls through to HTTP 500 | **Confirmed** | Medium | **Fixed** (typed `RequestAbortedError` + terminal-stop on all engines) |
| 5 | Streaming abort leaks error text into assistant content + non-standard `finish_reason` | **Confirmed** | Medium | **Fixed** (Option A — `stream_outputs` raises `RequestAbortedError`; public/private error split) |
| 6 | `EnginePool.use_engine` acquire-after-await ordering | **Refuted** (dead code — method has no callers) | — | **Fixed** (deleted) |
| 7 | `server.use_engine` resolve-twice — acquire/release on a different id than the liveness check | **Confirmed** | Low | **Fixed** (resolved once, threaded via `resolved_id` parameter) |

---

## Issue 1 — Enforcer over-eviction within a single tick — FIXED

### Where it was
`omlx/process_memory_enforcer.py:242-263`

### What happens
```python
async with self._engine_pool._tracked_lock("process_memory_enforcer"):
    while mx.get_active_memory() > self._max_bytes:
        victim = self._engine_pool._find_drain_or_evict_candidate()
        ...
        await self._engine_pool._unload_engine(victim)
        continue
```

`_unload_engine` (`omlx/engine_pool.py:1539-1586`) intentionally does **not** synchronously reclaim Metal memory. It flips `entry.engine = None`, sets state to `UNLOADING`, and schedules `_deferred_engine_cleanup` as an independent task. The heavy `gc.collect()` + `mx.synchronize()` + `mx.clear_cache()` runs later on the MLX executor — by design, so the pool lock is not held while waiting for Metal.

Consequence: inside a single `_check_and_enforce()` tick, `mx.get_active_memory()` does **not** drop between iterations. The loop keeps finding the next non-pinned candidate (the last one's `entry.engine is None` is skipped by `_find_drain_or_evict_candidate`) and evicts it too. On realistic hardware a single tick evicts **every loaded non-pinned model** even when one would have been enough.

Existing tests (`test_two_models_both_inferring_evict_both`, `test_evicts_multiple_models`) pass only because `mx.get_active_memory.side_effect` is a hand-tuned list that drops memory between iterations — a nonphysical mock.

### Verification

`tests/test_process_memory_enforcer.py::TestOverEviction::test_constant_memory_evicts_at_most_one_per_tick`

Setup: three non-pinned models, `mx.get_active_memory` returns a constant `15 GB`, `fake_unload` realistically sets `entry.engine = None` but does **not** drop the memory reading.

Current failure (confirming the bug):
```
AssertionError: Enforcer over-evicted: unloaded 3 models in one tick when memory
reading was constant (deferred-cleanup semantics). Expected <= 1.
assert 3 <= 1
  +  where 3 = <AsyncMock name='mock._unload_engine'>.await_count
WARNING  Evicting non-pinned model 'model-a' to enforce process memory limit
WARNING  Evicting non-pinned model 'model-b' to enforce process memory limit
WARNING  Evicting non-pinned model 'model-c' to enforce process memory limit
```

### Recommended fix (deterministic, not time-based)

**Do not wait.** Just evict at most one victim per tick and let `_enforcement_loop` (`omlx/process_memory_enforcer.py:182-192`) drive the next check when it comes around:

```python
async with self._engine_pool._tracked_lock("process_memory_enforcer"):
    if mx.get_active_memory() <= self._max_bytes:
        return
    victim = self._engine_pool._find_drain_or_evict_candidate()
    if victim is None:
        # no-candidate diagnostic (throttled) — existing code
        return
    entry = self._engine_pool._entries.get(victim)
    if entry is None or entry.engine is None:
        return
    if hasattr(entry.engine, "abort_all_requests"):
        aborted = await entry.engine.abort_all_requests()
        if aborted > 0:
            logger.warning(f"Aborted {aborted} requests on '{victim}' before eviction")
    logger.warning(f"Evicting non-pinned model '{victim}' ...")
    await self._engine_pool._unload_engine(victim)
```

The next tick (driven by `_enforcement_loop`'s existing polling) will re-check memory and, if still over, evict another. This is the deterministic version of the old loop — it makes forward progress monotonically without assumptions about when Metal memory is actually reclaimed.

Alternatively, if the design wants multiple evictions per tick without a timer, make the loop condition **state-based**:

```python
committed = <sum of entry.estimated_size for loaded non-pinned entries>
while committed > projected_limit:
    victim = ...
    committed -= entry.estimated_size
    await self._engine_pool._unload_engine(victim)
```

Either variant is purely event/state-driven. Neither relies on `mx.get_active_memory()` dropping within a single tick.

### Test after fix
The committed test asserts `<= 1` eviction per tick. Any fix that bounds the per-tick eviction count to `<= 1` passes. A fix that uses the committed-memory-shadow variant would need a separate test updating the assertion.

### Applied fix

Took the first (break-after-one) approach. `_check_and_enforce` now picks a single victim, aborts its requests, unloads it, and returns. The existing `_enforcement_loop` polling drives subsequent ticks if memory is still over. Also updated two tests (`test_evicts_one_per_tick_across_multiple_ticks`, `test_two_models_both_inferring_evict_both_across_ticks`) that were asserting the old cascading behavior — they now call `_check_and_enforce()` twice and verify one eviction per call.

---

## Issue 2 — Non-abortable engines torn down mid-operation — FIXED

### Where it was
`omlx/process_memory_enforcer.py:250-262` (the `hasattr` guard), plus every `BaseNonStreamingEngine` subclass that lacked a cooperative abort primitive.

### What was happening

Only `BatchedEngine` (`omlx/engine/batched.py:771`) and `VLMBatchedEngine` (`omlx/engine/vlm.py:1873`) implemented `abort_all_requests`. `EmbeddingEngine`, `RerankerEngine`, `STTEngine`, `TTSEngine`, `STSEngine` did not.

The enforcer guarded on `hasattr(entry.engine, "abort_all_requests")` and silently skipped the abort for non-LLM engines, proceeding directly to `_unload_engine`. An in-flight `embed()`, `transcribe()`, `synthesize()`, or `process()` would have its engine unloaded mid-operation. A follow-up call on the stale reference would hit `RuntimeError("Engine not started. Call start() first.")` and fall through to HTTP 500 instead of 503. The enforcer was also unable to actually reclaim memory while these operations were in flight, compounding Issue 1.

### The fix: uniform cooperative abort on `BaseNonStreamingEngine`

The review considered two approaches — (a) have the enforcer skip non-abortable victims with `active_uses > 0`, or (b) give the non-batched engines a first-class cooperative abort primitive. **Option (b) landed** because it's the only approach that preserves the enforcer's memory-reclamation guarantees and delivers clean HTTP 503 to clients on in-flight aborts.

**1. `BaseNonStreamingEngine` gained an abort primitive** (`omlx/engine/base.py:266-356`):

```python
class BaseNonStreamingEngine(ABC):
    def __init__(self):
        self._active_count = 0
        self._active_lock = threading.Lock()
        # Terminal abort flag. asyncio.Event() is safe to construct
        # without a running loop in Python 3.10+.
        self._aborted = asyncio.Event()

    async def abort_all_requests(self) -> int:
        count = self._active_count
        self._aborted.set()
        return count

    def _raise_if_aborted(self) -> None:
        if self._aborted.is_set():
            raise RequestAbortedError(
                f"Engine for {self.model_name} has been aborted "
                f"due to memory pressure. Please retry the request."
            )
```

Python cannot preempt an MLX kernel that is already running on the executor thread, but the `async` wrapper that awaits the executor future *can* discard the result and raise `RequestAbortedError` to the handler. Memory reclamation happens naturally after the in-flight call finishes — the enforcer's subsequent `_unload_engine` + deferred cleanup handles `mx.clear_cache` on the same single-threaded MLX executor.

The abort is **terminal**: once fired, the engine refuses all operations until `stop()` runs. The enforcer always pairs `abort_all_requests` with `_unload_engine`, so a fresh request arriving after abort sees `entry.engine is None` via `ensure_engine_alive` and receives 503 before reaching the engine.

**2. Each concrete engine calls `_raise_if_aborted` at both boundaries** of every public method that touches the MLX executor:

- Entry point — right after the "engine started" guard.
- After each `await loop.run_in_executor(...)` — so in-flight work whose abort fired while the executor future was running discards the result and raises instead of returning stale output.

Concrete insertion points:
- `embedding.py` — `embed()` (pre + post)
- `reranker.py` — `rerank()` (pre + post)
- `stt.py` — `transcribe()` (pre), `_do_transcribe()` (after `_load_and_split`, after each chunk), `_transcribe_single()` (after executor)
- `tts.py` — `synthesize()` (pre + after create_gen, after each segment, after finalize); `stream_synthesize()` (pre + after create_gen, between segments, after each PCM encode)
- `sts.py` — `process()` (pre + post)

**3. The enforcer's `hasattr` guard is gone** (`omlx/process_memory_enforcer.py`):

```python
# Every engine implements abort_all_requests — BatchedEngine/VLMBatchedEngine
# via EngineCore, BaseNonStreamingEngine subclasses via the cooperative
# abort flag. No special casing.
aborted = await entry.engine.abort_all_requests()
if aborted > 0:
    logger.warning(f"Aborted {aborted} requests on '{victim}' before eviction")
```

### Verification

Three classes of tests, all deterministic (event-driven, no sleeps).

**`tests/test_process_memory_enforcer.py::TestUniformAbortProtocol`**

- `test_enforcer_aborts_every_victim_regardless_of_engine_kind` — enforcer unconditionally calls `abort_all_requests` on the eviction victim and then unloads it, regardless of whether the engine is LLM or non-streaming. **PASSES**.

**`tests/test_engine_abort_protocol.py::TestBaseNonStreamingEngineAbortAPI`** (base-class contract)

- `test_abort_all_requests_is_available_on_every_subclass` — MRO check that `EmbeddingEngine`, `RerankerEngine`, `STTEngine`, `STSEngine`, `TTSEngine` all inherit `abort_all_requests` from `BaseNonStreamingEngine` and it's `async`. **PASSES**.
- `test_abort_returns_active_count_and_sets_flag` — `abort_all_requests()` returns the current `_active_count` and sets `_aborted`. **PASSES**.
- `test_raise_if_aborted_noop_before_abort` / `test_raise_if_aborted_raises_typed_error_after_abort` — pre-abort no-op, post-abort raises `RequestAbortedError` naming the engine. **PASSES**.

**`tests/test_engine_abort_protocol.py::TestCooperativeAbortBoundaries`** (uses a `_FakeEngine(BaseNonStreamingEngine)` with an `asyncio.Event`-gated "executor" for deterministic in-flight control)

- `test_pre_submit_abort_raises_before_executor_submission` — abort before `work()` → pre-submit check trips, no executor work is done. **PASSES**.
- `test_in_flight_abort_discards_executor_result` — the key test. Start `work()` as a background task, wait on `_work_in_flight.wait()` for the task to enter the "executor future running" region, fire `abort_all_requests()` from another task, release the fake executor via `_work_released.set()`, assert that awaiting the task raises `RequestAbortedError` rather than returning `"work-result"`. Fully event-driven. **PASSES**.
- `test_abort_is_terminal_subsequent_calls_also_fail` — abort is sticky, three subsequent `work()` calls all raise. **PASSES**.

**`tests/test_engine_abort_protocol.py::TestEmbeddingEngineAbort` / `TestRerankerEngineAbort`** (end-to-end on the real engine classes with the MLX model dependency mocked)

- `test_embed_after_abort_raises_request_aborted_error` — `EmbeddingEngine.embed` called after `abort_all_requests` raises `RequestAbortedError` and the underlying `model.embed` is never invoked (pre-submit check tripped). **PASSES**.
- `test_in_flight_embed_discards_result_on_abort` — patches `asyncio.get_running_loop` with a fake loop whose `run_in_executor` is gated on `asyncio.Event`s, fires abort mid-flight, releases the gate, verifies `model.embed` *did* run but its result was discarded in favor of `RequestAbortedError`. **PASSES**.
- `RerankerEngine` has the analogous test.

**`tests/test_engine_abort_protocol.py::TestEngineSourceAbortCheckpoints`** (source-level invariant)

- `test_every_public_entry_point_has_abort_check` — AST walk of each engine module asserts that `embed`, `rerank`, `transcribe`, `synthesize`, `stream_synthesize`, and `process` all contain a call to `_raise_if_aborted`. Will fail loudly if someone adds a new entry point without the check or removes an existing check during refactoring. **PASSES**.

### Notes

- `has_active_requests()` is still the guard the drain monitor uses for TTL-based cooperative eviction; only the hard memory-pressure path needed `abort_all_requests`.
- Streaming TTS back-pressure: if the client reads slowly and the handler is blocked on `yield chunk`, the abort flag is set but not yet checked. Once the client reads the next chunk, the handler resumes, `_raise_if_aborted()` at the top of the loop fires, and the generator terminates. No deadlock.
- Fix deterministically composes with Issue 1: the enforcer still does at most one eviction per tick, it just no longer has a class of engines it silently refused to abort.

---

## Issue 3 — Dead `except EngineEvictedError` around `pool.get_engine()` — FIXED

### Where it was
- `omlx/server.py:725` — around `await pool.get_engine(model_id)`
- `omlx/api/audio_routes.py:148` — around `await pool.get_engine(resolved)`

### What's wrong

`EngineEvictedError` is raised in exactly one place in the tree: `EnginePool.ensure_engine_alive()` at `omlx/engine_pool.py:922`. `pool.get_engine()` never raises it. Both `except EngineEvictedError` branches are unreachable dead code.

The new tests `test_chat_completions_get_engine_evicted_returns_503` (`tests/integration/test_server_endpoints.py`) and `test_transcription_get_engine_evicted_returns_503` (`tests/integration/test_audio_reranker_endpoints.py`) exercise these branches only by monkeypatching `pool.get_engine` to raise — they produce false coverage.

### Verification

`tests/test_engine_pool.py::TestEngineEvictedErrorInvariant`

- `test_engine_evicted_error_only_raised_in_ensure_engine_alive` — AST scan of `omlx/engine_pool.py` confirms the only `raise EngineEvictedError` is inside `ensure_engine_alive`. **PASSES** (and stays green forever, catching any future regression that adds a raise site without reviewing the handler contracts).
- `test_pool_get_engine_does_not_raise_engine_evicted_error` — runtime: `pool.get_engine("no-such-model")` raises `ModelNotFoundError`, not `EngineEvictedError`. **PASSES**.

### Applied fix

Deleted the two dead `except EngineEvictedError` branches plus the two dead tests that only exercised them:

- `omlx/server.py:723-726` — the `except EngineEvictedError` right after `await pool.get_engine(model_id)` is gone. The live catch around `pool.ensure_engine_alive(...)` at `server.py:801-803` remains — that's the reachable path.
- `omlx/api/audio_routes.py:148-149` — same deletion in `_use_engine`. Live catch around `pool.ensure_engine_alive(...)` retained. Docstring updated to name the actual raise site.
- `tests/integration/test_server_endpoints.py` — `TestEngineEvictedErrorHandling::test_chat_completions_get_engine_evicted_returns_503` deleted.
- `tests/integration/test_audio_reranker_endpoints.py` — `TestAudioEngineEvictedErrorHandling::test_transcription_get_engine_evicted_returns_503` deleted.

The committed invariant tests in `tests/test_engine_pool.py::TestEngineEvictedErrorInvariant` still pass and will keep the invariant honest: the AST scan fails loudly if anyone adds a new `raise EngineEvictedError` outside `EnginePool.ensure_engine_alive`, forcing a review of the handler contracts at the same time.

**Stronger fix (still optional, not done):** move the liveness check *inside* `pool.get_engine`, under the lock, right before returning `entry.engine`. That closes the race at its root and makes `ensure_engine_alive` redundant (or only a defensive late check). Would require updating the invariant test to accept the new raise site. Parked for now — the `ensure_engine_alive` pattern works and is clearer for callers.

---

## Issue 4 — Stopped engine → `RuntimeError` → HTTP 500 — FIXED

### Where it was
- `omlx/engine/batched.py:436` (chat), `:508` (stream_generate), `:619` (generate), `:685` (stream_chat)
- `omlx/engine/vlm.py:1429`, `:1498`, `:1592`, `:1643` (analogous four entry points)
- Non-streaming engines (`embedding.py`, `reranker.py`, `stt.py`, `tts.py`, `sts.py`) had the same class of bug via `if self._model is None: raise RuntimeError("Engine not started...")` firing after `stop()` nulled the model.

### What happens

`ensure_engine_alive` is a lock-free point-in-time check. Between it returning OK and the handler calling `engine.chat()`, the enforcer can evict and the deferred cleanup can run `engine.stop()`. When `chat()` is then invoked, it sees `self._stopped == True` and raises:

```python
raise RuntimeError(f"BatchedEngine for {self._model_name} has been stopped")
```

This plain `RuntimeError` bypasses the new `@app.exception_handler(RequestAbortedError)` at `server.py:582` and falls through to `unhandled_exception_handler` at `server.py:610`, which returns HTTP 500.

### Verification

`tests/integration/test_server_endpoints.py::TestStoppedEngineRuntimeError`
- `test_chat_runtime_error_returns_500_today` — **PASSES**, captures the current broken behavior. Server log:
  ```
  ERROR omlx.server:server.py:613 POST /v1/completions → 500 (unhandled):
      BatchedEngine for test-model has been stopped
  ```
- `test_chat_runtime_error_should_return_503_after_fix` — **XFAIL**, asserts the post-fix 503.
- `test_completions_runtime_error_returns_500_today` / `_should_return_503_after_fix` — same pair for `/v1/completions`.

`tests/integration/test_server_endpoints.py::TestBatchedEngineStoppedRaisesTypedError` — 4 unit-level tests that pass today documenting the plain `RuntimeError` raise, plus 1 xfail asserting the typed exception should be raised.

### Applied fix

The fix has two parts — one for the two `_stopped`-flagged LLM engine classes and one for the model-guard pattern on every `BaseNonStreamingEngine` subclass.

**1. `BatchedEngine` and `VLMBatchedEngine`.** All four `if self._stopped: raise RuntimeError(...)` sites in each class now raise `RequestAbortedError` with a message naming the model and the root cause. The `start()` check ("has been stopped and cannot be restarted") is intentionally left as `RuntimeError` — that's a programming error, not a client-facing race. `request_aborted_handler` at `server.py:582` translates the typed exception to HTTP 503 automatically.

```python
# batched.py and vlm.py, at each of the four runtime entry points
if self._stopped:
    raise RequestAbortedError(
        f"Engine for {self._model_name} has been stopped "
        f"due to memory pressure. Please retry the request."
    )
```

**2. `BaseNonStreamingEngine` — terminal `stop()`.** The non-streaming engines did not have a `_stopped` flag; they relied on `self._model is None` after `stop()`. A handler racing with stop would hit `raise RuntimeError("Engine not started")` and land on HTTP 500.

The fix is to treat `stop()` as a terminal form of abort and flip the check order at each entry point:

- New helper on `BaseNonStreamingEngine` (`omlx/engine/base.py`): `_mark_stopped()` sets the existing cooperative abort flag (`self._aborted`). Concrete engines call it at the top of their `stop()` method, **before** clearing `self._model`.
- Every public entry point (`embed`, `rerank`, `transcribe`, `synthesize`, `stream_synthesize`, `process`) now calls `self._raise_if_aborted()` **before** the `self._model is None` guard. A handler racing with stop sees the typed `RequestAbortedError` (→ 503) instead of the plain `RuntimeError` (→ 500).

The "genuinely never started" case is preserved: if `start()` was never called at all, `self._aborted` is clean, `_raise_if_aborted()` is a no-op, and the `self._model is None` guard fires with the original `RuntimeError("Engine not started")`. This distinction is covered by a dedicated regression test.

The `_mark_stopped()` helper composes cleanly with Issue 2's abort protocol: `abort_all_requests()` and `stop()` both latch the same `_aborted` event, so the cooperative check at each entry point serves both code paths uniformly.

### Verification

**`tests/integration/test_server_endpoints.py::TestStoppedEngineReturns503`** (server-level, 2 tests)

- `test_chat_on_stopped_engine_returns_503` — mock `engine.chat` to raise `RequestAbortedError`, POST `/v1/chat/completions`, assert 503 + OpenAI-shaped error body. **PASSES**.
- `test_completions_on_stopped_engine_returns_503` — same for `/v1/completions` via `engine.generate`. **PASSES**.

**`tests/integration/test_server_endpoints.py::TestBatchedEngineStoppedRaisesTypedError`** (unit-level, 4 tests)

- `test_chat_on_stopped_engine_raises_request_aborted_error` — construct a `BatchedEngine`, set `_stopped=True`, call `chat` → `RequestAbortedError` naming the model. **PASSES**.
- Same for `generate`, `stream_generate`, `stream_chat`. **PASSES**.

**`tests/integration/test_server_endpoints.py::TestBaseNonStreamingEngineStoppedRaisesTypedError`** (unit-level, 3 tests)

- `test_embed_after_stop_raises_request_aborted_error` — instantiate `EmbeddingEngine` with a mocked `MLXEmbeddingModel`, call `stop()`, then `embed(...)` → `RequestAbortedError` naming the model. Proves that `_mark_stopped()` sets `_aborted` before `_model = None` and that the reordered `_raise_if_aborted()` check fires first. **PASSES**.
- `test_rerank_after_stop_raises_request_aborted_error` — same for `RerankerEngine`. **PASSES**.
- `test_embed_still_raises_runtime_error_when_never_started` — construct a fresh `EmbeddingEngine` without ever starting it, call `embed(...)`, assert `RuntimeError("Engine not started")` still fires. Confirms the "programming error" case is not masked by the fix. **PASSES**.

The `tests/test_engine_abort_protocol.py::TestEngineSourceAbortCheckpoints` AST-scan test continues to pass: every public entry point still contains a `_raise_if_aborted` call (they moved from post-model-guard to pre-model-guard, which the AST scan doesn't care about — it just requires presence).

---

## Ancillary find — `BoundarySnapshotSSDStore.cleanup_all` filesystem race

While running the full unit suite after the Issue 4 fix landed, `tests/test_boundary_snapshot_store.py::TestBoundarySnapshotSSDStore::test_cleanup_all_drains_queue` failed intermittently. Investigating revealed a pre-existing thread-safety bug in `omlx/cache/boundary_snapshot_store.py` that was unrelated to the enforcer work but sat in the same region of concurrent code.

### What was happening

`BoundarySnapshotSSDStore.cleanup_all()` drained the write queue, cleared `_pending_writes`, then ran `shutil.rmtree` + `mkdir` on the snapshot directory. But it never serialized with the background writer thread: items the writer had **already dequeued** were not cancelled, and the writer's `mkdir + write + rename` sequence raced with cleanup's `rmtree + mkdir`. When the writer's mkdir landed inside the freshly-recreated snapshot dir, a `req-1/` subdirectory (and sometimes a `.safetensors` file) leaked past the cleanup.

`cleanup_all` also unconditionally `self._cancelled_requests.clear()`'d, wiping the only mechanism that would otherwise tell the writer to skip a stale item.

### Deterministic reproduction

`tests/test_boundary_snapshot_store.py::TestBoundarySnapshotSSDStore::test_cleanup_all_serializes_with_in_flight_write_item` drives the race by hand:

1. Shut down the store's writer thread so the test owns processing.
2. Call `save()` — adds to `_pending_writes`, enqueues an item.
3. Dequeue the item manually.
4. Run `cleanup_all()` — drains the (empty) queue, clears state, rmtrees, recreates the snapshot dir.
5. Call the newly-extracted `_process_one_write_item(item)` on the stale item.
6. Assert the snapshot dir is empty.

Fully deterministic — no sleeps, no races, no retries. The original `test_cleanup_all_drains_queue` kept its thread-based shape but now passes consistently thanks to the fix.

### The fix

Two coordinated changes in `omlx/cache/boundary_snapshot_store.py`:

1. **`_io_lock`** — a new `threading.Lock` held by the writer across its entire per-item critical section (`cancel check + mkdir + write + rename + pending_writes cleanup`) and by `cleanup_all` / `cleanup_request` across their filesystem mutations. Serializes the two paths deterministically.
2. **Staleness check** — inside the writer's `_io_lock` block, re-check `pw_key in self._pending_writes`. If cleanup wiped the entry between dequeue and lock acquisition, skip the item. This catches the case where the writer acquires `_io_lock` *after* a cleanup has completed — under `_io_lock` alone, such a writer would still run its `mkdir + write` against the freshly-recreated directory.

The writer loop also got a small refactor: the per-item body moved into `_process_one_write_item(item)` so tests can drive it synchronously without racing the background thread.

### Validation

Each configuration was run 20 times against the new deterministic test:

| Configuration | Result |
|---|---|
| Full fix | **20 / 20 pass** |
| Buggy baseline (HEAD) | **20 / 20 fail** (AttributeError — refactor absent) |
| Partial fix (refactor + `_io_lock`, staleness check removed) | **20 / 20 fail** (`AssertionError: Stale write item leaked files into cleanup_all'd snapshot_dir`) |

The third row is the key discriminator: removing *only* the staleness check while keeping the refactor and lock still deterministically reproduces the leak, proving the test specifically exercises the staleness-check logic rather than any ambient thread-safety win from the lock alone.

---

## Issue 5 — Streaming abort leaks error into assistant content + non-standard `finish_reason` — FIXED

### Where it was
- `omlx/engine_core.py:515-524` — the `raise` → `break` change from the original diff
- `omlx/server.py:2776-2814` — `stream_chat_completion` treats `output.new_text` as content without checking `output.error`
- `omlx/server.py:2654-2673` — `stream_completion` same pattern
- `omlx/engine/base.py:12-30` — `GenerationOutput` dataclass drops the `error` field, so even if handlers wanted to check it, it's not propagated

### What happens

When the enforcer calls `abort_all_requests`, `engine_core.abort_all_requests` pushes a `RequestOutput` onto the collector with:

```python
new_text = "\n\n[Error: Request aborted: process memory limit exceeded. "
           "Increase --max-process-memory or reduce context size.]"
finish_reason = "error"
error = "Request aborted: process memory limit exceeded. ..."
```

`stream_outputs` yields this and then breaks (new behavior). The yielded output flows through `BatchedEngine.stream_generate` → `GenerationOutput(..., finish_reason="error", ...)` — the `error` field is **dropped** at the `GenerationOutput` boundary.

The handler at `stream_chat_completion` iterates `async for output in engine.stream_chat(...)` and emits `output.new_text` as a plain content delta. It does not check `output.finish_reason == "error"`. The final chunk carries `finish_reason="error"`, which is not one of the OpenAI-standard values (`stop`, `length`, `tool_calls`, `content_filter`, `function_call`).

Client sees:
1. Operational error text embedded in assistant content, rendered as if the model said it.
2. An internal CLI flag name (`--max-process-memory`) leaked to API clients.
3. A non-standard `finish_reason` that strict OpenAI clients may reject.

The justification comment at `engine_core.py:519-524` — "raising here would propagate through StreamingResponse and abort the connection after headers are flushed" — is **incorrect** for the affected handlers. Both `stream_chat_completion` (`server.py:2815`) and `stream_completion` (`server.py:2674`) wrap their `async for` loops in `except Exception as e: yield error; yield [DONE]; return` blocks that would cleanly handle a raise.

### Verification

`tests/integration/test_server_endpoints.py::TestStreamingAbortErrorLeak`

- `test_chat_stream_leaks_abort_error_into_content` — **PASSES**, documents current leak. Asserts (a) a content delta chunk contains `"Error: Request aborted"`, (b) the final chunk's `finish_reason == "error"`, (c) `--max-process-memory` flag name leaks into content.
- `test_chat_stream_does_not_leak_error_into_content_after_fix` — **XFAIL**.
- `test_chat_stream_finish_reason_is_valid_openai_value_after_fix` — **XFAIL**.

### Applied fix (Option A + public/private error split)

Two coordinated changes, both in `omlx/engine_core.py`:

**1. `stream_outputs` raises instead of yielding error output** (`omlx/engine_core.py` inside the `stream_outputs` loop):

```python
if output.error:
    # Raise a typed exception that the FastAPI streaming handlers
    # catch in their `except Exception` blocks.
    raise RequestAbortedError(output.error)

yield output
```

Critically, the raise happens **before** the `yield`. The handler's `async for` never sees the error output, so there's no opportunity to emit it as a content delta. The handler's existing `except Exception as e:` block (verified in all four handlers — `stream_completion`, `stream_chat_completion`, `stream_anthropic_messages`, `stream_responses_api`) catches the exception, emits a proper SSE error event, emits `[DONE]`, and returns cleanly. The handler's final-chunk `finish_reason` logic is bypassed entirely, so the non-standard `finish_reason="error"` can no longer leak.

**2. Public/private error split in `abort_all_requests`**:

```python
public_error = (
    "Request aborted due to server memory pressure. "
    "Please retry the request."
)
collector.put(
    RequestOutput(
        request_id=rid,
        finished=True,
        finish_reason="error",
        new_text="",      # no accidental content leak
        error=public_error,
    )
)
```

Operator-only hints (CLI flag names like `--max-process-memory`, internal variable names) no longer reach clients. Operators still see the full story in the server logs via the existing `logger.warning(...)` call. `new_text` is explicitly empty so that any handler that might still iterate the output (e.g. a misbehaving custom backend) doesn't accidentally leak the error as assistant content.

The same contract applies to the single-request `abort_request()` path: it also sets `error="Request aborted"` on its RequestOutput, so `stream_outputs` will raise uniformly regardless of which abort path fired. The test `test_abort_request_wakes_blocked_stream_outputs` was updated to reflect this.

### Why Option A and not Option B

The review considered two paths (see original version of this file for the full analysis). Option A — making `stream_outputs` the single source of truth that raises on error — was chosen because:

- It keeps the error-handling logic in one place (the engine core), instead of duplicating a "check for error" pattern across every streaming handler.
- The existing `except Exception` blocks in all four handlers were already wired to emit proper SSE error events, verified by source inspection before the fix landed. No handler changes were needed.
- `GenerationOutput` can keep its existing shape — no new `error` field to thread through `batched.stream_generate` / `vlm.stream_generate` etc.
- The contract is simpler to reason about: "if the engine hands you a `RequestOutput`, it's a real token; anything else comes via exception".

### Verification

**Unit-level** (`tests/test_engine_core.py`):
- `TestEngineCoreErrorPropagation::test_stream_outputs_raises_request_aborted_error_on_error_output` — builds an `EngineCore`, injects an error `RequestOutput` into the collector, iterates `stream_outputs`, asserts `RequestAbortedError` is raised and **no output was yielded first**. Against the buggy `yield + break` version, this fails with `DID NOT RAISE` — a direct discriminator.
- `TestEngineCoreAbortRequest::test_abort_request_wakes_blocked_stream_outputs` — updated to assert the consumer terminates via `RequestAbortedError` rather than receiving a yielded abort output.
- `TestEngineCoreErrorPropagation::test_generate_raises_request_aborted_error` — unchanged; the non-streaming `generate()` path already raised.
- `TestEngineCoreAbortAllRequests::test_abort_all_requests` — updated to assert the new public error contract: `error` field contains sanitized text without operator hints, `new_text` is empty.

**Integration-level** (`tests/integration/test_server_endpoints.py::TestStreamingAbortSurfacesCleanSSEError`):
- `test_chat_stream_aborted_has_no_content_leak` — no content delta carries `"Request aborted"` or `"memory"`.
- `test_chat_stream_aborted_emits_clean_sse_error_chunk` — response ends with `data: [DONE]`; exactly one SSE chunk carries a top-level `error` key; its message contains the public abort text and the word "retry".
- `test_chat_stream_aborted_public_error_has_no_operator_leaks` — response body contains neither `--max-process-memory` nor internal variable names like `_max_bytes`.
- `test_chat_stream_aborted_no_nonstandard_finish_reason` — no chunk carries a non-standard `finish_reason`; only `stop`/`length`/`tool_calls`/`content_filter`/`None` appear.
- `test_completion_stream_aborted_emits_clean_sse_error_chunk` — the same invariants hold for `/v1/completions` streaming via `stream_completion`.

All tests deterministic (mocks simulate the post-fix contract: `stream_chat` / `stream_generate` raise `RequestAbortedError`). No timers, no retries.

---

## Issue 6 — `EnginePool.use_engine` ordering — REFUTED (dead code) — FIXED

### Finding

`EnginePool.use_engine` at `omlx/engine_pool.py:954` had zero callers in the tree. Verified by `rg "\.use_engine\(" omlx/ tests/`:
- `server.use_engine` is a separately defined function in `omlx/server.py:810` that calls `pool.acquire_engine` / `get_engine` / `pool.release_engine` directly — it is **not** a wrapper around `pool.use_engine`.
- `audio_routes._use_engine` is its own local helper.
- The only references to `pool.use_engine` were the method definition, the docstring example, and comment cross-references.

The acquire-after-await ordering concern was real **in theory**, but since no code path exercised the method, there was no observable impact. No test was ever written for it.

### Applied fix

Deleted the method outright. Also cleaned up two stale comment cross-references that mentioned it:

1. The "Engine use-counting" block comment above `acquire_engine` no longer says "(or `use_engine` context manager)". It now correctly points at `server.use_engine` and adds a note that the enforcer deliberately does NOT respect `active_uses` on its hard-limit path — a correction that ties in with the Issue 2 fix (the old comment still said "process_memory_enforcer → skips victims with `active_uses > 0`", which has been false since Issue 2 landed).
2. `_find_drain_or_evict_candidate`'s docstring now reads "acquired the engine via `acquire_engine`" instead of "via `acquire_engine/use_engine`".

The `from contextlib import asynccontextmanager` import stays — still used by `_tracked_lock`.

No tests added or removed. Full unit suite continues to pass with identical counts.

---

## Issue 7 — `server.use_engine` resolve-twice mismatch — FIXED

### Where it was
- `omlx/server.py` `server.use_engine` — resolved once locally to take a lease, then `server.get_engine` resolved again internally.
- `omlx/server.py` `create_completion`, `create_chat_completion`, anthropic `create_message`, Responses API `create_response` — same pattern: `resolve_model_id()` at module level, then `get_engine_for_model(request.model)` which re-resolved via `server.get_engine`.

If the alias map changed between the two calls (settings reload, hot-swap), `acquire`/`release` operated on `alias-a` while `ensure_engine_alive` checked `alias-b`. The lease was balanced but guarded the wrong entry; the liveness check ran against an entry that was never leased.

### Applied fix

Added an optional `resolved_id: str | None = None` parameter to `server.get_engine` and threaded it through every convenience wrapper (`get_engine_for_model`, `get_embedding_engine`, `get_reranker_engine`, `get_asr_engine`, `get_tts_engine`):

```python
async def get_engine(
    model_id: str | None = None,
    engine_type: EngineType = EngineType.LLM,
    resolved_id: str | None = None,
) -> ...:
    ...
    if resolved_id is not None:
        model_id = resolved_id
    else:
        model_id = pool.resolve_model_id(
            model_id, _server_state.settings_manager
        )
```

When the caller has already resolved, it passes `resolved_id=<resolved>` and `get_engine` skips the second `pool.resolve_model_id` call.

Updated all callers that pre-resolve + take a lease to pass the resolved id through:
- `server.use_engine` — threads through its locally resolved `resolved_id`.
- `create_completion` (line 2009) — `get_engine_for_model(request.model, resolved_id=resolved_model)`.
- `create_chat_completion` (line 2179) — same.
- `create_message` (anthropic, line 3383) — same.
- `create_response` (Responses API, line 3743) — same.

Other callers that do **not** pre-resolve remain unchanged and naturally resolve exactly once:
- `create_embeddings` (line 1818) — calls `get_embedding_engine(request.model)` without a pre-resolve. Single resolve happens inside `get_engine`. Not a Issue-7 case (no race), left as-is.
- The audio routes (`omlx/api/audio_routes.py`) use their own local `_use_engine` helper that calls `pool.get_engine` directly and was already single-resolve-by-construction — untouched.

Admin endpoints (`omlx/admin/benchmark.py`, `omlx/admin/accuracy_benchmark.py`, `omlx/admin/routes.py`) call `pool.get_engine(raw_id)` without alias resolution at all — a separate pre-existing concern, not Issue 7.

### Verification

`tests/test_server.py::TestUseEngineResolveOnce` (replaces the old `TestUseEngineResolveRace` which documented the buggy behavior):

- `test_resolve_called_once_and_all_ops_see_same_id` — rigs `pool.resolve_model_id` with `side_effect=["alias-a", "SHOULD-NOT-BE-CALLED"]`, runs `async with server.use_engine("my-alias", EngineType.LLM)`, asserts (a) `resolve_model_id.call_count == 1`, (b) `acquire_engine`, `ensure_engine_alive`, `release_engine` all received `"alias-a"`, (c) `pool.get_engine.await_args.args[0] == "alias-a"`. The second side_effect value is never consumed, proving the fix collapsed two resolves into one.
- `test_chat_completion_handler_resolves_once` — covers the handler path: simulates what a chat handler does (call `resolve_model_id()` at module level, then `get_engine_for_model(request.model, resolved_id=resolved_model)`), asserts that `pool.resolve_model_id` is called exactly once across the entire flow and that `pool.get_engine` receives the pre-resolved id.

Both tests pass deterministically. The old buggy-behavior test was deleted since it asserted the two-resolves-with-mismatch state that no longer exists.

---

## Prioritized recommendation

Fix in this order, each is small and independently reviewable:

1. **Issue 4** (trivial): swap `RuntimeError` → `RequestAbortedError` in `BatchedEngine` / `VLMBatchedEngine` `_stopped` checks. Closes the narrow race end-to-end.
2. **Issue 1** (trivial): bound the enforcer to one eviction per tick.
3. **Issue 2** (small): skip non-abortable victims with `active_uses > 0`.
4. **Issue 5** (small): revert `break` → raise, or propagate `error` field through `GenerationOutput` and have handlers route it as SSE error. Sanitize the enforcer's public error message.
5. **Issue 7** (small): resolve-once in `server.use_engine`.
6. **Issue 3** (trivial): delete dead `except EngineEvictedError` branches and their fake tests.
7. **Issue 6** (trivial): delete unused `EnginePool.use_engine`.

All recommended fixes are deterministic — no timers, no polling, no retries. They move the system from *"hope the timing works out"* to *"the invariant holds by construction"*.

---

## Committed verification tests

| File | Tests added |
|---|---|
| `tests/test_process_memory_enforcer.py` | `TestOverEviction::test_constant_memory_evicts_at_most_one_per_tick`; `TestNonAbortableEngineProtection::test_evicts_non_abortable_engine_with_active_uses`; `TestNonAbortableEngineProtection::test_skips_non_abortable_with_active_uses_and_falls_back` (xfail) |
| `tests/test_engine_pool.py` | `TestEngineEvictedErrorInvariant::test_engine_evicted_error_only_raised_in_ensure_engine_alive`; `TestEngineEvictedErrorInvariant::test_pool_get_engine_does_not_raise_engine_evicted_error` |
| `tests/integration/test_server_endpoints.py` | `TestStoppedEngineRuntimeError` (2 pass + 2 xfail); `TestBatchedEngineStoppedRaisesTypedError` (4 pass + 1 xfail); `TestStreamingAbortErrorLeak` (1 pass + 2 xfail) — all `@pytest.mark.integration` |
| `tests/test_server.py` | `TestUseEngineResolveRace::test_resolve_changes_between_calls_leaks_counters` |

Each "current behavior" test documents the bug and will need to be deleted or inverted in the fix commit. Each xfail test documents the intended post-fix behavior and will auto-promote when the fix lands — if they promote unexpectedly during future work, CI will fail loudly, forcing a review.
