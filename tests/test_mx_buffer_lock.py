"""Unit tests for the Metal buffer-pool serialization lock (mx_buffer_lock).

Fast and GPU/Metal-free: they verify the lock primitive + helper contracts and
guard the AGENTS.md invariant (off-thread buffer reads and clear/synchronize
mutators must share one lock; the fixed paths must not re-introduce an
unguarded clear_cache dispatch). See omlx/AGENTS.md and the end-to-end
reproducer nanobot/tests/e2e/test_kv_save_load_race_e2e.py.
"""

import inspect
import re
import threading
from pathlib import Path

import pytest

from omlx import mx_buffer_lock
from omlx.mx_buffer_lock import (
    locked_sync_and_clear_cache,
    mx_buffer_access_lock,
    run_locked,
)


def _held_by_another_thread() -> bool:
    """True iff the lock is currently held (a *fresh* thread can't acquire it).

    RLock is owned per-thread, so acquiring from a separate thread fails while
    the lock is held by anyone — which is exactly the cross-thread exclusion the
    buffer lock provides.
    """
    result: dict[str, bool] = {}

    def _probe() -> None:
        got = mx_buffer_access_lock.acquire(blocking=False)
        result["got"] = got
        if got:
            mx_buffer_access_lock.release()

    t = threading.Thread(target=_probe)
    t.start()
    t.join()
    return not result["got"]


def test_lock_is_reentrant_and_shared_with_scheduler():
    # Reentrant: the same thread may acquire it more than once (the sync store
    # path holds it, then _extract_tensor_bytes re-acquires).
    assert mx_buffer_access_lock.acquire(blocking=False)
    try:
        assert mx_buffer_access_lock.acquire(blocking=False)
        mx_buffer_access_lock.release()
    finally:
        mx_buffer_access_lock.release()

    # The scheduler must use the SAME instance (it imports it as an alias).
    from omlx import scheduler

    assert scheduler._mx_buffer_access_lock is mx_buffer_access_lock


def test_run_locked_holds_lock_during_call_and_returns_result():
    assert not _held_by_another_thread()  # free before

    observed: dict[str, bool] = {}

    def work() -> int:
        observed["held"] = _held_by_another_thread()
        return 42

    assert run_locked(work) == 42
    assert observed["held"] is True       # held for the duration of the call
    assert not _held_by_another_thread()  # released after


def test_run_locked_releases_lock_on_exception():
    def boom() -> None:
        raise ValueError("boom")

    with pytest.raises(ValueError):
        run_locked(boom)
    assert not _held_by_another_thread()  # released even on error


def test_locked_sync_and_clear_cache_holds_lock_around_mx_calls(monkeypatch):
    """It must synchronize THEN clear, both while holding the buffer lock."""
    import mlx.core as mx

    calls: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        mx, "synchronize",
        lambda *a, **k: calls.append(("sync", _held_by_another_thread())),
    )
    monkeypatch.setattr(
        mx, "clear_cache",
        lambda *a, **k: calls.append(("clear", _held_by_another_thread())),
    )

    locked_sync_and_clear_cache()

    assert [name for name, _ in calls] == ["sync", "clear"]   # sync before clear
    assert all(held for _, held in calls)                     # lock held for both
    assert not _held_by_another_thread()                      # released after


def test_extract_tensor_bytes_reader_holds_the_buffer_lock():
    """The single off-thread buffer-read chokepoint must take the lock."""
    from omlx.cache.paged_ssd_cache import _extract_tensor_bytes

    src = inspect.getsource(_extract_tensor_bytes)
    assert "mx_buffer_access_lock" in src, (
        "_extract_tensor_bytes must read Metal buffers under mx_buffer_access_lock "
        "(see AGENTS.md); off-thread reads racing clear_cache abort the process."
    )


# Paths whose clear_cache/synchronize were routed through locked_sync_and_clear_cache.
# A raw `lambda: (mx.synchronize(), mx.clear_cache())` dispatched to an executor
# here is the exact unguarded pattern that crashed multi-model serving.
_GUARDED_PATHS = [
    "engine_pool.py",
    "scheduler.py",
    "engine/embedding.py",
    "engine/stt.py",
    "engine/sts.py",
    "engine/tts.py",
    "engine/reranker.py",
    "engine/dflash.py",
    "engine/vlm.py",
]
_UNGUARDED_LAMBDA = re.compile(
    r"lambda:\s*\(\s*mx\.synchronize\(\)\s*,\s*mx\.clear_cache\(\)\s*\)"
)


def test_no_unguarded_clear_cache_lambda_on_fixed_paths():
    pkg_root = Path(mx_buffer_lock.__file__).parent
    offenders = [
        rel for rel in _GUARDED_PATHS
        if _UNGUARDED_LAMBDA.search((pkg_root / rel).read_text())
    ]
    assert not offenders, (
        "Unguarded 'lambda: (mx.synchronize(), mx.clear_cache())' found in "
        f"{offenders}. Route it through locked_sync_and_clear_cache() so the "
        "buffer-pool reclaim can't race an off-thread read (see AGENTS.md)."
    )
