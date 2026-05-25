"""Shared serialization for Metal buffer-pool access.

A single process-wide lock that mediates between:

  * threads that READ Metal buffers via the Python buffer protocol off the
    inference thread — notably the async phase-2 KV-cache SSD save
    (``prefix_cache._save_blocks_phase2`` -> ``paged_ssd_cache.save_block`` ->
    ``_extract_tensor_bytes``), and
  * any thread that issues a buffer-pool-reclaiming MLX op
    (``mx.clear_cache()`` / ``mx.synchronize()``) — the scheduler inference
    thread AND the engine-pool loader/eviction paths
    (``_signal_exclusive_idle``, post-load cache release, deferred cleanup).

Without this, ``clear_cache`` can reclaim the underlying Metal buffer pool
while another thread is mid-read, corrupting an in-flight GPU command buffer.
That surfaces non-deterministically as a SIGABRT (#1106) or a Metal
command-buffer error (``kIOGPUCommandBufferCallbackError*``: OutOfMemory /
Timeout) that aborts the whole omlx process.

Previously the lock lived in ``scheduler.py`` and only guarded the synchronous
store path; the async phase-2 save and the engine-pool clear paths were
unguarded. This module hoists it so every party can share the one instance
without import cycles (it imports nothing from omlx).
"""

from __future__ import annotations

import threading

# Reentrant: a single logical operation may acquire it more than once
# (e.g. the synchronous store path already holds it around store_cache, which
# calls save_block -> _extract_tensor_bytes, which re-acquires it).
mx_buffer_access_lock = threading.RLock()


def locked_sync_and_clear_cache() -> None:
    """``mx.synchronize()`` + ``mx.clear_cache()`` under the buffer-access lock.

    Use this everywhere a buffer-pool reclaim runs off the request path
    (engine-pool load/evict/idle teardown) so it cannot race a concurrent
    buffer read. Synchronizing before clearing also prevents releasing buffers
    still referenced by in-flight command buffers (issue #300).
    """
    import mlx.core as mx

    with mx_buffer_access_lock:
        mx.synchronize()
        mx.clear_cache()


def locked_free_and_clear(drop=None) -> None:
    """Drop a model reference, ``gc.collect()``, then ``synchronize`` +
    ``clear_cache`` — all under the buffer-access lock.

    MUST be run on the MLX executor thread (via ``run_in_executor``). Model
    eviction frees the victim's MLX arrays; if that free runs on the
    asyncio event-loop thread (the old ``self._model = None; gc.collect()``
    in each engine ``stop()``) it races buffer allocations from in-flight
    generation on the executor thread, corrupting the in-flight GPU work
    (garbled TTS audio / bad inference) — same hazard class as the
    ``clear_cache`` race (#85). Running the drop+gc here, on the executor and
    under the lock, serializes the free with generation and every other
    buffer-pool party.
    """
    import gc

    import mlx.core as mx

    with mx_buffer_access_lock:
        if drop is not None:
            drop()
        gc.collect()
        mx.synchronize()
        mx.clear_cache()


def run_locked(fn):
    """Run ``fn()`` while holding the buffer-access lock; return its result.

    Wrap reclaim-triggering MLX-executor work — model loads above all — so it
    cannot run concurrently with an off-thread Metal buffer read (phase-2
    KV-cache save / vision-feature save). A model load allocates large buffers,
    which makes the MLX allocator reclaim *cached* buffers; doing that mid-read
    corrupts the in-flight GPU command buffer. Locking ``clear_cache`` alone is
    insufficient because allocation is itself a reclaim trigger. RLock, so
    nesting with ``locked_sync_and_clear_cache`` / ``_extract_tensor_bytes`` is
    safe.
    """
    with mx_buffer_access_lock:
        return fn()
