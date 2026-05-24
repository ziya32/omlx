# AGENTS.md — Metal buffer-pool serialization (read before touching Metal/cache code)

## The invariant

omlx is designed so **all GPU/Metal work runs on one thread** — the single-worker
`get_mlx_executor()` (`engine_core.py`). Inference, model loads, and
`mx.clear_cache()`/`mx.synchronize()` are all dispatched there, so MLX's Metal
buffer pool is never mutated by two threads at once.

**That invariant is violated on purpose by the async cache-save paths.** To keep
inference fast, the KV-cache SSD save runs `_extract_tensor_bytes()`
(`cache/paged_ssd_cache.py`) on a *separate* executor — it reads live Metal
buffers via the Python buffer protocol (`bytes(memoryview(arr))`) off the MLX
thread. Same for `cache/boundary_snapshot_store.py` and
`cache/vision_feature_cache.py` (both call `_extract_tensor_bytes`).

## The hazard

If a **buffer-pool mutator** runs on the MLX thread while one of those readers is
mid-read, the pool is reclaimed/moved out from under the reader → Metal buffer
corruption → the next GPU command buffer aborts with
`kIOGPUCommandBufferCallbackError*` (Timeout *or* OutOfMemory) →
`std::runtime_error` → **SIGABRT kills the whole process** (taking every loaded
model + in-flight request with it).

Mutators are not just `clear_cache`/`synchronize` — **a model load allocates
buffers, which makes the allocator reclaim cached buffers too.** This is what
crashed `TestGatewayStress` (2026-05-23): a phase-2 KV save racing the embedding
model load. Same class as issue #1106.

## The rule (how the fix works — follow it)

`mx_buffer_lock.py` exposes one process-wide reentrant lock,
`mx_buffer_access_lock`, plus two helpers. **Every party that touches the Metal
buffer pool must hold it:**

- **Off-thread buffer READS** → already covered: the lock is taken *inside*
  `_extract_tensor_bytes()`, the single chokepoint for all readers. New off-thread
  code that reads `mx.array` bytes (memoryview / buffer protocol) must also hold
  `mx_buffer_access_lock`.
- **clear_cache / synchronize** → call **`locked_sync_and_clear_cache()`**, never
  raw `mx.synchronize()` + `mx.clear_cache()`. (Already applied across
  `engine_pool.py`, `scheduler.py`, `engine/{stt,sts,tts,reranker,dflash,vlm}.py`.)
- **Model loads** → wrap the load dispatched to `get_mlx_executor()` with
  **`run_locked(fn)`** (e.g. `run_in_executor(get_mlx_executor(), lambda: run_locked(self._model.load))`).

### Do / Don't
- ✅ Route any new `clear_cache`/`synchronize` through `locked_sync_and_clear_cache()`.
- ✅ Wrap any new model load with `run_locked(...)`.
- ✅ Hold `mx_buffer_access_lock` around any new off-thread Metal-buffer read.
- ❌ Don't call raw `mx.clear_cache()`/`mx.synchronize()` anywhere a concurrent
  off-thread buffer read could be in flight.
- ❌ Don't assume "all Metal is on one thread" — the async cache-save paths break
  that; the lock is what makes them safe.
- ⚠️ It's an `RLock` (re-entrant): nesting (e.g. the sync store path already
  holding it, then `_extract_tensor_bytes` re-taking it) is fine. Keep hold times
  short — only the actual buffer read / pool mutation, never disk I/O.

## Improving it
Better than this lock would be eliminating the cross-thread access entirely —
e.g. extract KV bytes to host memory *on the MLX thread* at save-submit time and
let only the disk write be async. That removes the reader/mutator race at the
cost of moving the (bounded) memcpy onto the inference thread. If you do that, the
lock can be dropped — but verify with the reproducer first.

## Verify before/after changes
`nanobot/tests/e2e/test_kv_save_load_race_e2e.py` reproduces the race
(large pinned-model KV phase-2 save vs an embedding-model load) in ~4 min and
aborts unfixed omlx; it must stay green. The full `TestGatewayStress` is the
end-to-end guard.
