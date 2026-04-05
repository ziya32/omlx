# Pinned Model Pre-Request Eviction

**Status**: Draft v4
**Date**: 2026-04-04

## Problem

When a large VLM model is pinned in memory, remaining headroom is gradually consumed by smaller non-pinned models that load on demand and stay resident. When a large VLM request arrives (many images, long context), the runtime memory needed for vision encoding and KV cache exceeds what's available because the smaller models are still occupying memory.

### Why current mechanisms fail

| Mechanism | Why it doesn't help |
|---|---|
| `_prepare_memory_for()` | Only runs when loading an UNLOADED model. The pinned VLM is already ACTIVE, so this never fires for VLM requests. |
| `_init_vision_limits()` | Computed **once** at VLM load time from `mx.get_active_memory()`. If smaller models load *after* the VLM, the vision budget is stale — it reflects headroom that no longer exists. |
| `ProcessMemoryEnforcer` | Reactive. Polls every 1s and evicts after memory *already* exceeds the limit. By then, vision encoding is mid-allocation and the system is in OOM territory or swapping. |
| Prefill memory guard | Aborts the **VLM request itself** when memory is high, rather than evicting the smaller models. Exactly backwards. |
| TTL | Reduces the window but doesn't eliminate it. A VLM request can arrive while a small model is actively serving or within its TTL window. |

### Concrete scenario

```
Timeline:
  t0  VLM loaded (pinned), 20GB weights. 12GB headroom.
  t1  Small LLM-A loaded on demand, 4GB. 8GB headroom remaining.
  t2  Small LLM-B loaded on demand, 3GB. 5GB headroom remaining.
  t3  VLM request arrives: 8 images, needs ~9GB for vision + KV.
      Only 5GB available → OOM or swap death spiral.
```

The system has no way to say: "A request is about to run on the pinned VLM and needs more headroom than what's available. Evict non-pinned models *before* starting inference."

---

## Design

### Core idea

Add **pre-request eviction** to `get_engine()` for pinned models. When a request arrives for a pinned model, proactively evict **all** non-pinned models — both idle and active — before returning the engine to the caller. Idle models are unloaded immediately; active models are drained (in-flight requests finish gracefully or time out, then unloaded).

### New per-model setting: `exclusive`

```python
@dataclass
class ModelSettings:
    exclusive: bool = False  # When True, evict all non-pinned models on request
```

Only meaningful when `is_pinned` is also True. If a model is pinned but not exclusive, existing behavior is preserved (non-pinned models coexist freely and are only evicted for model *loading* pressure).

Setting `exclusive: true` communicates: "This model's runtime headroom is more important than keeping other models warm."

### Configuration

```yaml
# model_settings.yaml
models:
  Qwen2.5-VL-72B-Instruct-4bit:
    pinned: true
    exclusive: true       # ← new
  SmolLM2-360M-Instruct:
    pinned: false
    ttl_seconds: 300
```

---

## Implementation Plan

### Overview

```
get_engine(model_id)
  ├─ fast path: state == ACTIVE, engine is not None
  │   ├─ [existing] update last_access, return engine
  │   └─ [NEW] if entry.is_pinned and entry.exclusive:
  │       ├─ _clear_for_exclusive(entry) → evict idle, drain active
  │       ├─ if drains pending → wait outside lock, re-enter loop
  │       └─ recalculate vision limits, return engine
  └─ (other states: LOADING, DRAINING, UNLOADED — unchanged)
```

### Step 1: Add `exclusive` field to `ModelSettings`

**File**: `omlx/model_settings.py`

```python
@dataclass
class ModelSettings:
    # ... existing fields ...
    exclusive: bool = False
```

Propagate to `EngineEntry`:

```python
@dataclass
class EngineEntry:
    # ... existing fields ...
    exclusive: bool = False  # Evict all non-pinned on request
```

Set during `discover_models()` / `apply_settings()` from the settings manager, alongside `is_pinned`.

---

### Step 2: Add `_clear_for_exclusive()` method to `EnginePool`

**File**: `omlx/engine_pool.py`

New method called under `self._lock`. Returns `None` if all non-pinned models are already gone, or an `asyncio.Event` the caller must wait on (then re-enter the loop).

```python
async def _clear_for_exclusive(
    self, pinned_entry: EngineEntry
) -> asyncio.Event | None:
    """Evict all non-pinned models to maximize headroom for a pinned model.

    Called under self._lock when a request arrives for a pinned+exclusive model.

    Returns:
        None — all non-pinned models cleared, caller may proceed.
        asyncio.Event — a drain or unload is in progress. Caller must
            await this event outside the lock, then re-enter get_engine().
    """
    for mid, e in self._entries.items():
        if e.is_pinned or e is pinned_entry:
            continue

        # Already transitioning — collect an event to wait on
        if e.state == EngineState.DRAINING:
            return e.drain_complete
        if e.state == EngineState.UNLOADING:
            return e.unload_complete

        # Not loaded — skip
        if e.engine is None or e.state != EngineState.ACTIVE:
            continue

        # Loaded non-pinned model — must go
        has_work = (
            self._engine_has_active_work(e.engine) or e.active_uses > 0
        )
        if has_work:
            # Active requests → drain gracefully
            self._start_drain(mid)
            logger.info(
                f"Exclusive headroom: draining '{mid}' "
                f"(active_uses={e.active_uses})"
            )
            return e.drain_complete
        else:
            # Idle → unload immediately
            logger.info(f"Exclusive headroom: evicting idle '{mid}'")
            await self._unload_engine(mid, reason="exclusive_headroom")
            # Continue loop — may need to evict more models.
            # But _unload_engine sets state to UNLOADING and defers
            # cleanup, so the next iteration will see UNLOADING and
            # return unload_complete if Metal cleanup isn't done yet.

    # All non-pinned models cleared (or never existed)
    return None
```

**Key design choices:**

1. **One eviction per lock acquisition.** When a drain is started, the method returns the `drain_complete` event immediately rather than trying to start all drains at once. This is because:
   - Drains are asynchronous — we can't hold the lock while waiting.
   - The caller re-enters the `get_engine()` loop, re-acquires the lock, and `_clear_for_exclusive()` is called again. On the next iteration it finds the first model DRAINING (returns its event) or already UNLOADED (moves on to the next non-pinned model).
   - This converges: each iteration removes one model from the "needs clearing" set.

2. **Respects existing drain mechanics.** Uses `_start_drain()` which launches `_drain_monitor()` with the existing timeout and abort logic. No new drain semantics needed.

3. **Unload before drain check.** Idle models are evicted first (same iteration), then if any busy model remains, its drain event is returned. This ensures we don't waste a round-trip waiting for an idle model that could have been freed immediately.

---

### Step 3: Modify `get_engine()` fast path

**File**: `omlx/engine_pool.py`, in `get_engine()` method

Current fast path (lines 542-545):

```python
if entry.state == EngineState.ACTIVE and entry.engine is not None:
    entry.last_access = time.time()
    return entry.engine
```

Modified:

```python
if entry.state == EngineState.ACTIVE and entry.engine is not None:
    entry.last_access = time.time()

    # Exclusive pinned models: clear non-pinned models before
    # returning the engine so inference has maximum headroom.
    if entry.is_pinned and entry.exclusive:
        wait_event = await self._clear_for_exclusive(entry)
        if wait_event is not None:
            # Must wait for drain/unload — fall through to the
            # wait-outside-lock section, then re-enter loop.
            event = wait_event
            wait_target = "exclusive_headroom"
            # Don't set should_load — model is already loaded.
            # Break out of the lock block to enter the wait path.
        else:
            # All clear — recalculate vision limits if VLM
            self._refresh_vision_limits(entry)
            return entry.engine
    else:
        return entry.engine
```

The existing wait-outside-lock code (lines ~654-705) already handles arbitrary events with timeout + retry. The only change needed is to recognize `wait_target == "exclusive_headroom"` for logging purposes.

After wake-up, the `while True` loop re-enters, hits the fast path again, calls `_clear_for_exclusive()` again. If more models need clearing, it waits again. If all clear, it returns the engine.

---

### Step 4: Recalculate vision limits after eviction

**File**: `omlx/engine_pool.py`

New helper method. **Does NOT call `_init_vision_limits()`** because that
calls `mx.clear_cache()` directly on the event loop thread, which is unsafe
when the MLX executor is running concurrent Metal operations (see
Concurrency Review — Issue #2). Instead, computes the budget from committed
memory, which is purely arithmetic and always consistent under the lock.

```python
def _refresh_vision_limits(self, entry: EngineEntry) -> None:
    """Recalculate vision limits from committed memory state.

    Called under self._lock after _clear_for_exclusive() returns None
    (all non-pinned models cleared). Uses _committed_memory() rather
    than mx.get_active_memory() to avoid Metal calls under the lock.
    """
    engine = entry.engine
    if engine is None or not hasattr(engine, 'vision_chunk_budget_pixels'):
        return

    enforcer = self._process_memory_enforcer
    max_bytes = getattr(enforcer, 'max_bytes', 0) if enforcer else 0
    if not max_bytes:
        return

    committed = self._committed_memory()
    headroom = max(0, max_bytes - committed)

    engine.vision_chunk_budget_pixels = int(
        headroom * self._VISION_SAFETY_FACTOR
        / self._VISION_BYTES_PER_PIXEL
    ) if headroom > 0 else 0

    entry._vision_limits_cache = None  # Invalidate cached limits

    logger.info(
        "Vision limits refreshed: chunk_budget_pixels=%d "
        "(headroom=%.1fGB, committed=%.1fGB, limit=%.1fGB)",
        engine.vision_chunk_budget_pixels,
        headroom / 1e9,
        committed / 1e9,
        max_bytes / 1e9,
    )
```

Called in the fast path after `_clear_for_exclusive()` returns `None` (all models cleared). This ensures the VLM's `vision_chunk_budget_pixels` reflects the actual available memory, not the stale value from initial load.

---

### Step 5: Coalescing concurrent requests

Multiple VLM requests arriving simultaneously should not each independently trigger a full eviction pass. The existing mechanics handle this naturally:

1. First request enters `get_engine()`, acquires lock, calls `_clear_for_exclusive()`, starts drains.
2. Second request enters `get_engine()`, acquires lock (after first releases it), calls `_clear_for_exclusive()`. The models are already DRAINING — it returns the existing `drain_complete` event.
3. Both requests wait on the same event.
4. When drain completes, both wake up, re-enter loop, both see all-clear, both return the engine.

No special coalescing logic needed — the lock serialization + shared events provide it.

---

## Interaction with Existing Mechanisms

### ProcessMemoryEnforcer

The enforcer continues to run as a safety net. Pre-request eviction reduces the frequency of enforcer interventions because memory pressure is resolved *before* inference starts, not after. The enforcer's emergency abort remains as defense-in-depth for edge cases (e.g., a VLM request that exceeds even the full memory budget).

### TTL

TTL continues to work for non-exclusive models. For exclusive models, TTL is largely redundant — non-pinned models are evicted on every request anyway. No conflict.

### `_prepare_memory_for()`

Gains one additional early gate (Step 9b): non-pinned loads defer while an exclusive model is actively inferring, regardless of memory math. This protects the exclusive model's scheduler throughput from MLX executor contention during weight loading. Remaining memory-accounting logic is unchanged. Pre-request eviction handles runtime memory for already-loaded models; the early gate and Step 9 together handle the reverse direction. The three are complementary.

### Prefill memory guard

Still active as a last resort. With pre-request eviction, it fires less often because headroom is maximized before inference begins. When it does fire, it's for genuinely oversized requests that exceed the system's total capacity.

---

## State Machine Additions

No new states. The existing state machine is sufficient:

```
                        _clear_for_exclusive()
                              │
    ┌─────────────────────────┼─────────────────────────────┐
    │                         │                             │
    ▼                         ▼                             ▼
 ACTIVE (idle)            ACTIVE (busy)              DRAINING/UNLOADING
    │                         │                             │
    │ _unload_engine()        │ _start_drain()              │ (already
    │                         │                             │  transitioning)
    ▼                         ▼                             │
 UNLOADING ──► UNLOADED   DRAINING ──► UNLOADING ──► UNLOADED
                              │                             │
                              └──── wait on event ──────────┘
                                         │
                                         ▼
                              get_engine() re-enters loop
                              _clear_for_exclusive() again
                              all clear → return engine
```

---

## Request Flow (End-to-End)

```
1. HTTP request for pinned+exclusive VLM model
      │
2. server.get_engine(vlm_model_id)
      │
3. pool.get_engine(vlm_model_id)
      │
4. Lock acquired. Entry is ACTIVE.
      │
5. entry.is_pinned and entry.exclusive → _clear_for_exclusive()
      │
      ├─ SmallLLM-A is ACTIVE, idle → _unload_engine("A", reason="exclusive_headroom")
      │   A transitions: ACTIVE → UNLOADING (deferred cleanup launched)
      │
      ├─ SmallLLM-B is ACTIVE, busy (active_uses=1) → _start_drain("B")
      │   B transitions: ACTIVE → DRAINING (drain_monitor launched)
      │   Returns B.drain_complete event
      │
6. Lock released. Caller awaits drain_complete.
      │
      ├─ Meanwhile: B's in-flight request finishes naturally.
      │   drain_monitor detects no active work → _unload_engine("B")
      │   B transitions: DRAINING → UNLOADING → UNLOADED
      │   drain_complete.set()
      │
7. Caller wakes. Re-enters get_engine() loop.
      │
8. Lock acquired. Entry is still ACTIVE.
      │
9. _clear_for_exclusive() again:
      │
      ├─ A is UNLOADING → return A.unload_complete
      │   (Metal cleanup still in progress)
      │
      │   OR
      │
      ├─ A is UNLOADED, B is UNLOADED → return None (all clear)
      │
10. If None: _refresh_vision_limits() → return engine.
    If event: release lock, wait, re-enter loop (go to step 8).
      │
11. VLM inference starts with maximum memory headroom.
      │
12. Vision encoding + KV cache allocation succeeds.
```

---

## Latency Impact

**Worst case**: A VLM request arrives while a non-pinned model is mid-inference with a long generation. The VLM request waits up to `drain_timeout` (default 120s) for the drain to complete.

**Typical case**: Non-pinned models are idle (no active requests). Eviction is immediate (sub-millisecond state transition + deferred Metal cleanup). The VLM request may wait for one `unload_complete` event (~100-500ms for `mx.synchronize() + mx.clear_cache()`).

**Best case**: No non-pinned models loaded. `_clear_for_exclusive()` returns `None` immediately. Zero added latency.

**Amortization**: Only the *first* VLM request after non-pinned models load pays the eviction cost. Subsequent VLM requests find no models to evict (until new non-pinned requests load them again).

---

## Edge Cases

### 1. All non-pinned models are busy

`_clear_for_exclusive()` starts drains for all of them. The VLM request waits for each drain to complete, one at a time (each loop iteration waits on one event). In the worst case, this is `N * drain_timeout` if all models time out.

**Mitigation**: The drain timeout (120s) is already configurable. For latency-sensitive VLM workloads, a shorter drain timeout (e.g., 30s) can be set globally.

### 2. Non-pinned model requests arrive during VLM inference

This is the reverse direction of the core problem and needs careful handling. See the dedicated section below: **[Non-Pinned Request Handling During Exclusive Inference](#non-pinned-request-handling-during-exclusive-inference)**.

### 3. Multiple pinned+exclusive models

If two models are both pinned and exclusive, they will not evict each other (both are pinned). They coexist normally with the headroom split between them. The exclusive flag only affects non-pinned models.

### 4. Exclusive model with no active requests

If no requests are arriving for the exclusive model, non-pinned models load and unload normally. The exclusive eviction only triggers when a request arrives. Between requests, the system operates as before.

### 5. Vision limits become 0

If the system has very little memory after loading the VLM, `_init_vision_limits()` may calculate `vision_chunk_budget_pixels = 0`. This means all vision encoding is rejected (images can't be processed). This is correct — it's better to reject with a clear error than to OOM.

---

## Non-Pinned Request Handling During Exclusive Inference

### The problem (reverse direction)

After `_clear_for_exclusive()` evicts all non-pinned models and the VLM begins inference, a request for a non-pinned small model arrives. What happens?

**Current behavior (broken):**

```
1. get_engine("SmallLLM") → entry.state == UNLOADED
2. _prepare_memory_for(entry):
   ├─ _committed_memory() = VLM estimated_size only (e.g. 20GB)
   │   (KV cache + vision buffers NOT counted — only model weights)
   ├─ committed + small_model_size ≤ max_model_memory → passes ✓
   └─ Falls through to _check_process_memory(entry)
3. _check_process_memory(entry):
   ├─ mx.get_active_memory() = 28GB (VLM weights + KV + vision)
   ├─ projected = 28GB + 4GB = 32GB > max_bytes (30GB) → too high
   ├─ _find_drain_or_evict_candidate() → None (VLM is pinned)
   ├─ No draining/loading/unloading models
   ├─ No cleanup tasks
   ├─ Fallback: committed(20GB) + small(4GB) = 24GB ≤ 30GB → passes ✓
   │   "Metal residual may be reclaimed during load. Proceeding cautiously."
   └─ Returns None → load proceeds
4. Small model loads → actual memory = 32GB+ → OOM or swap
```

The committed-memory fallback at line 1000-1013 is designed for Metal allocator lag (freed buffers not yet reclaimed). But here the memory is *legitimately* in use by the VLM's KV cache. The fallback incorrectly allows the load.

Even if the fallback rejects (projected > max_bytes + 25% headroom), the result is an immediate `InsufficientMemoryError` (HTTP 507). The caller gets a failure with no option to retry — it doesn't know the VLM will finish and free memory soon.

**What we need:** Non-pinned requests should **wait** for the exclusive model to finish its active requests, then load when memory is actually available.

---

### Design: Exclusive-idle notification

Add an event-based mechanism so non-pinned model requests can wait for the exclusive model's runtime memory to be freed.

#### New field on `EngineEntry`

```python
@dataclass
class EngineEntry:
    # ... existing fields ...
    exclusive_idle: asyncio.Event | None = None  # Signaled when active_uses → 0
```

#### Lifecycle of `exclusive_idle`

```
acquire_engine():
  if entry.exclusive and entry.active_uses == 0:
      entry.exclusive_idle = asyncio.Event()   # ← create fresh event
  entry.active_uses += 1

release_engine():
  entry.active_uses -= 1
  if entry.exclusive and entry.active_uses == 0:
      # Schedule async: mx.synchronize + mx.clear_cache, then signal
      asyncio.get_running_loop().create_task(
          _signal_exclusive_idle(entry)
      )
```

The event is created when the exclusive model goes from idle→busy (first request acquired) and signaled when it goes back to idle (last request released). Each busy→idle transition creates a fresh event so stale signals don't leak across cycles.

#### Metal cache clearing before signal

When the VLM request finishes, the scheduler frees KV arrays and calls `mx.synchronize(generation_stream)`. But Metal's buffer cache still holds the pages — `mx.get_active_memory()` stays high. We must call `mx.clear_cache()` before signaling waiters so they see the actual freed memory.

```python
async def _signal_exclusive_idle(self, entry: EngineEntry) -> None:
    """Clear Metal cache and signal that the exclusive model is idle.

    Runs as a fire-and-forget task from release_engine(). Must run
    on the MLX executor to safely call mx.synchronize/clear_cache.
    """
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        get_mlx_executor(),
        lambda: (mx.synchronize(), mx.clear_cache()),
    )
    if entry.exclusive_idle is not None:
        entry.exclusive_idle.set()
```

#### Intercept in `_check_process_memory()`

Add a check **before** the committed-memory fallback (before line 995):

```python
# --- NEW: wait for exclusive models instead of falling through ---
# If an exclusive model has active requests, its KV cache / vision
# buffers are consuming memory that will be freed when it finishes.
# Wait for it rather than loading now (which would OOM) or rejecting
# with 507 (which gives the caller no retry path).
for mid, e in self._entries.items():
    if (
        e.exclusive
        and e.is_pinned
        and e.active_uses > 0
        and e.exclusive_idle is not None
    ):
        logger.info(
            f"Waiting for exclusive '{mid}' to finish "
            f"({e.active_uses} active) before loading "
            f"'{entry.model_id}'"
        )
        return e.exclusive_idle

# [existing committed-memory fallback below]
```

When the non-pinned request wakes from `exclusive_idle`:
1. Re-enters `get_engine()` loop → `_prepare_memory_for()` → `_check_process_memory()`
2. `mx.get_active_memory()` is now lower (KV freed + cache cleared)
3. `projected = lower_memory + small_model_size` → fits → load proceeds
4. If VLM started another request in the meantime → `exclusive_idle` is a fresh un-set event → wait again

---

### Full timeline with both directions

```
t0  VLM loaded (pinned+exclusive). 20GB weights. 12GB headroom.
t1  SmallLLM loaded on demand (4GB). 8GB headroom.

t2  VLM request R1 arrives.
    get_engine("VLM"):
      _clear_for_exclusive() → SmallLLM is idle → evict immediately.
      SmallLLM: ACTIVE → UNLOADING → UNLOADED.
      _refresh_vision_limits() → budget recalculated with full headroom.
      Returns VLM engine.
    VLM starts vision encoding + prefill (uses ~9GB KV + vision).
    mx.get_active_memory() = 29GB.

t3  SmallLLM request R2 arrives (while VLM is mid-inference).
    get_engine("SmallLLM"):
      entry.state == UNLOADED → _prepare_memory_for()
      _committed_memory(20GB) + 4GB = 24GB ≤ max(30GB) → passes weight check
      _check_process_memory():
        mx.get_active_memory() = 29GB
        projected = 29GB + 4GB = 33GB > 30GB → too high
        No eviction candidates (VLM is pinned)
        → NEW: finds VLM has active_uses=1 and exclusive_idle event
        → returns exclusive_idle event
      Caller waits on exclusive_idle.

t4  VLM request R1 finishes.
    release_engine("VLM") → active_uses: 1 → 0.
    _signal_exclusive_idle():
      mx.synchronize() + mx.clear_cache()
      → mx.get_active_memory() drops from 29GB to ~20GB
      exclusive_idle.set()

t5  SmallLLM request R2 wakes.
    Re-enters get_engine() loop → _prepare_memory_for():
      _check_process_memory():
        mx.get_active_memory() = 20GB
        projected = 20GB + 4GB = 24GB ≤ 30GB → fits ✓
    SmallLLM loads. Serves request R2.

t6  VLM request R3 arrives (while SmallLLM is serving R2).
    get_engine("VLM"):
      _clear_for_exclusive() → SmallLLM is busy (active_uses=1)
      → _start_drain("SmallLLM"), returns drain_complete.
    Caller waits on drain_complete.
    SmallLLM finishes R2 → drain_monitor unloads it → drain_complete.set().
    VLM caller wakes → _clear_for_exclusive() returns None → proceed.
```

---

### Throughput vs starvation: optimal strategy

The naive exclusive approach has a starvation risk: if VLM requests stream in continuously, non-pinned requests wait indefinitely (each time they wake, a new VLM request has already re-acquired exclusivity).

#### Three-layer strategy

**Layer 1: Natural windowing (default, no config)**

Most VLM workloads are bursty — a request takes 5-30s to complete, then there's a gap before the next one. Non-pinned requests naturally fill these gaps:

```
VLM:   ████████░░░░████████░░░░████████
Small:          ██░░        ██░░
                ↑ window     ↑ window
```

No special logic needed. The exclusive-idle event + `get_engine()` retry loop handles this automatically. This is sufficient for most real workloads.

**Layer 2: Bounded wait with timeout**

Non-pinned requests waiting on `exclusive_idle` are subject to the existing `max_wait_timeout` (default 300s). If the VLM is continuously busy for 5 minutes, the non-pinned request times out with a `ModelLoadingError` (HTTP 504) that includes a descriptive message:

```
"Model 'SmallLLM' timed out waiting for exclusive model 'VLM' to
become idle (waited 300s). The VLM has continuous active requests.
Retry later or increase max_wait_timeout."
```

This prevents infinite starvation — non-pinned requests get a clear signal to retry later rather than hanging forever.

**Layer 3: Starvation guard (configurable, `exclusive_max_hold`)**

For workloads where VLM requests are near-continuous and non-pinned models must still be served, add an optional `exclusive_max_hold` setting (default: disabled/0):

```yaml
models:
  Qwen2.5-VL-72B-Instruct-4bit:
    pinned: true
    exclusive: true
    exclusive_max_hold: 120  # seconds: max continuous exclusive hold
```

When `exclusive_max_hold > 0`, the exclusive model tracks how long it has continuously held exclusivity (i.e., time since `active_uses` was last 0). When this duration exceeds `exclusive_max_hold`:

- `_clear_for_exclusive()` is **skipped** for the next VLM request.
- Non-pinned models are allowed to load and coexist.
- The VLM request proceeds without eviction (may get reduced vision budget).
- After the non-pinned model finishes and is released, the hold timer resets.

```python
# In get_engine() fast path, before calling _clear_for_exclusive():
if entry.exclusive and entry._exclusive_hold_start > 0:
    hold_duration = time.time() - entry._exclusive_hold_start
    if hold_duration > entry.exclusive_max_hold:
        logger.info(
            f"Exclusive hold expired for '{model_id}' "
            f"({hold_duration:.0f}s > {entry.exclusive_max_hold}s), "
            f"allowing non-pinned models to coexist"
        )
        # Skip eviction — return engine without clearing
        self._refresh_vision_limits(entry)
        return entry.engine
```

Tracking:
```python
# In acquire_engine(), when exclusive model goes 0 → 1:
entry._exclusive_hold_start = time.time()

# In release_engine(), when exclusive model goes 1 → 0:
entry._exclusive_hold_start = 0.0
```

This gives a guaranteed window for non-pinned requests every `exclusive_max_hold` seconds, even under continuous VLM load.

#### Why not more complex scheduling?

Alternatives considered and rejected:

| Alternative | Rejected because |
|---|---|
| Per-request memory estimation for VLM | Unreliable: vision memory depends on image layout, model architecture, dynamic tile counts. False negatives cause OOM; false positives waste capacity. |
| Load small models during VLM decode phase | KV cache is still allocated during decode. Headroom available is small and unpredictable. Risk of OOM mid-generation (worse than upfront rejection). |
| Request priority queue with preemption | Adds a scheduling layer orthogonal to the engine pool. High complexity for marginal throughput gain over natural windowing. |
| Time-sliced round-robin between models | Defeats the purpose of exclusive mode (maximizing VLM headroom). Model loading/unloading overhead per time slice is too high. |

The three-layer strategy is chosen because:
- Layer 1 handles 90%+ of real workloads with zero config
- Layer 2 provides a safety net (timeout) with zero config
- Layer 3 is opt-in for the rare continuous-VLM-load case

---

### Implementation: additional steps

#### Step 6: Add `exclusive_idle` to `EngineEntry`

**File**: `omlx/engine_pool.py`

```python
@dataclass
class EngineEntry:
    # ... existing fields ...
    exclusive_idle: asyncio.Event | None = None
    _exclusive_hold_start: float = 0.0
```

#### Step 7: Modify `acquire_engine()` and `release_engine()`

**File**: `omlx/engine_pool.py`

```python
def acquire_engine(self, model_id: str) -> None:
    entry = self._entries.get(model_id)
    if entry is not None:
        if entry.exclusive and entry.active_uses == 0:
            # Entering exclusive hold — create fresh event for waiters
            entry.exclusive_idle = asyncio.Event()
            entry._exclusive_hold_start = time.time()
        entry.active_uses += 1

def release_engine(self, model_id: str) -> None:
    entry = self._entries.get(model_id)
    if entry is not None and entry.active_uses > 0:
        entry.active_uses -= 1
        if entry.exclusive and entry.active_uses == 0:
            entry._exclusive_hold_start = 0.0
            # CRITICAL: Capture the event reference NOW, not later.
            # acquire_engine() may replace entry.exclusive_idle with
            # a fresh Event before _signal_exclusive_idle() runs.
            # See "Concurrency Review — Issue #1" below.
            event_to_signal = entry.exclusive_idle
            # Fire async task: clear Metal cache, then signal waiters
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(
                    self._signal_exclusive_idle(event_to_signal)
                )
            except RuntimeError:
                # No running loop (shutdown) — signal directly
                if event_to_signal is not None:
                    event_to_signal.set()
```

#### Step 8: Add `_signal_exclusive_idle()` method

**File**: `omlx/engine_pool.py`

```python
async def _signal_exclusive_idle(
    self, event: asyncio.Event | None
) -> None:
    """Clear Metal buffer cache and signal a specific event.

    Runs as a fire-and-forget task spawned by release_engine().
    Takes the EVENT directly (not the entry) to avoid the capture
    race where acquire_engine() replaces entry.exclusive_idle
    before this task runs.

    Must run mx.synchronize + mx.clear_cache on the MLX executor
    so that mx.get_active_memory() reflects the freed KV/vision
    buffers before waiters re-check.
    """
    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            get_mlx_executor(),
            lambda: (mx.synchronize(), mx.clear_cache()),
        )
    except Exception as e:
        logger.warning(
            f"Metal cache clear failed in exclusive idle signal: {e}"
        )
    finally:
        if event is not None:
            event.set()
```

#### Step 9: Modify `_check_process_memory()` to wait on exclusive idle

**File**: `omlx/engine_pool.py`

Insert before the committed-memory fallback (before current line 995):

```python
# If an exclusive model has active requests, its runtime memory
# (KV cache, vision buffers) will be freed when it finishes.
# Wait for it rather than loading into contested memory or
# rejecting with an immediate 507.
for mid, e in self._entries.items():
    if (
        e.exclusive
        and e.is_pinned
        and e.active_uses > 0
        and e.exclusive_idle is not None
    ):
        logger.info(
            f"Waiting for exclusive model '{mid}' to finish "
            f"({e.active_uses} active request(s)) before loading "
            f"'{entry.model_id}'"
        )
        return e.exclusive_idle
```

#### Step 9b: Early-gate non-pinned loads in `_prepare_memory_for()`

**File**: `omlx/engine_pool.py`

Step 9 handles the memory-contention case: we wait on `exclusive_idle`
when a non-pinned load would exceed the process memory limit. But there
is a second, orthogonal reason to wait — **MLX executor contention**.

The MLX executor is a single-threaded `ThreadPoolExecutor`. Every Metal
operation — weight loading, scheduler steps, `mx.clear_cache`, vision
encoding — is serialized onto that one thread. Loading a new model
takes seconds and occupies the executor for the entire duration. If
an exclusive VLM is mid-inference when a non-pinned load is admitted,
the VLM's scheduler steps queue behind the weight-load work and decode
throughput tanks even when the memory math would pass.

Insert an early gate at the **top** of `_prepare_memory_for()`, before
any memory checks:

```python
async def _prepare_memory_for(
    self, entry: EngineEntry
) -> asyncio.Event | None:
    # Don't load non-pinned models while an exclusive model is actively
    # inferring.  Model loading monopolizes the single MLX executor
    # thread, starving the exclusive model's scheduler steps.  Wait for
    # the exclusive model to finish instead.
    if not entry.is_pinned:
        for mid, e in self._entries.items():
            if e.exclusive and e.is_pinned:
                if e.active_uses > 0 and e.exclusive_idle is not None:
                    logger.info(
                        f"Deferring load of '{entry.model_id}': "
                        f"exclusive model '{mid}' has "
                        f"{e.active_uses} active request(s)"
                    )
                    return e.exclusive_idle

    # ... existing memory checks (UNLOADING wait, LOADING serialization,
    #     committed-memory check, _check_process_memory, etc.)
```

**Why this is separate from Step 9:**

| | Step 9 (`_check_process_memory`) | Step 9b (`_prepare_memory_for` gate) |
|---|---|---|
| Rationale | Memory contention | MLX executor contention |
| Fires when | Projected memory > limit | Always, for any non-pinned load |
| What it protects | Avoid OOM on the load | Avoid starving the VLM scheduler |

The two hooks complement each other. Step 9b catches the common case —
a small model load arriving during VLM decode where memory would fit
but executor contention would hurt VLM throughput. Step 9 remains as
the memory-safety fallback for the tighter case where the load also
overflows the memory budget.

**Interaction with Layer 2 (bounded wait):** Both hooks return the
same `exclusive_idle` event, so the waiter is bounded by the same
`max_wait_timeout` mechanism. Starvation protection from the
three-layer strategy applies identically to both hooks.

#### Step 10 (optional): Add `exclusive_max_hold` setting

**File**: `omlx/model_settings.py`

```python
@dataclass
class ModelSettings:
    # ... existing fields ...
    exclusive: bool = False
    exclusive_max_hold: int = 0  # 0 = disabled (hold indefinitely)
```

**File**: `omlx/engine_pool.py` — in `get_engine()` fast path, before `_clear_for_exclusive()`:

```python
if entry.exclusive_max_hold > 0 and entry._exclusive_hold_start > 0:
    hold = time.time() - entry._exclusive_hold_start
    if hold > entry.exclusive_max_hold:
        logger.info(
            f"Exclusive hold expired for '{model_id}' "
            f"({hold:.0f}s > {entry.exclusive_max_hold}s)"
        )
        self._refresh_vision_limits(entry)
        return entry.engine
```

---

## Files Changed

| File | Change |
|---|---|
| `omlx/model_settings.py` | Add `exclusive`, `exclusive_max_hold` to `ModelSettings` |
| `omlx/engine_pool.py` | Add `exclusive`, `exclusive_idle`, `_exclusive_hold_start` fields to `EngineEntry` |
| `omlx/engine_pool.py` | Add `_clear_for_exclusive()` method |
| `omlx/engine_pool.py` | Add `_refresh_vision_limits()` helper |
| `omlx/engine_pool.py` | Add `_signal_exclusive_idle()` method |
| `omlx/engine_pool.py` | Modify `get_engine()` fast path (exclusive eviction + hold expiry) |
| `omlx/engine_pool.py` | Modify `acquire_engine()` / `release_engine()` (exclusive idle lifecycle) |
| `omlx/engine_pool.py` | Modify `_check_process_memory()` (wait on exclusive idle — memory-contention hook, Step 9) |
| `omlx/engine_pool.py` | Early gate in `_prepare_memory_for()` (wait on exclusive idle — MLX executor-contention hook, Step 9b) |
| `omlx/engine_pool.py` | Wire `exclusive` from settings in `discover_models()` / `apply_settings()` |
| `tests/` | Unit + integration tests |
| `tests/integration/test_exclusive_pinned_e2e.py` | E2E: concurrent VLM + embedding + reranker + audio with pinned VLM |

---

## Testing Plan

### Unit tests — exclusive eviction (VLM request direction)

1. **Idle eviction**: Pin model A (exclusive). Load A and B. Request A. Assert B is unloaded before engine returned.
2. **Drain eviction**: Pin model A (exclusive). Load A and B. Start inference on B. Request A. Assert B enters DRAINING, then UNLOADED after drain.
3. **Multiple non-pinned**: Pin A (exclusive). Load A, B, C. Request A. Assert B and C both evicted (may take multiple loop iterations).
4. **Vision limits refresh**: Pin VLM (exclusive). Load VLM and small LLM. Request VLM. Assert `vision_chunk_budget_pixels` is recalculated with more headroom after eviction.
5. **Coalescing**: Pin A (exclusive). Load A and B (busy). Send two concurrent requests to A. Assert only one drain is started for B, both requests wait on same event.
6. **No-op when clean**: Pin A (exclusive). Load only A. Request A. Assert `_clear_for_exclusive()` returns None, no eviction, no delay.
7. **Non-exclusive pinned**: Pin A (not exclusive). Load A and B. Request A. Assert B is NOT evicted.

### Unit tests — non-pinned request waiting (reverse direction)

8. **Wait for exclusive idle (memory hook, Step 9)**: Pin A (exclusive) with active request, projected memory for B would overflow. Request B (non-pinned, unloaded). Assert B's `_check_process_memory()` returns `exclusive_idle`, not immediate 507.
8b. **Wait for exclusive idle (executor hook, Step 9b)**: Pin A (exclusive) with active request, memory headroom is plenty for B. Request B (non-pinned, unloaded). Assert B's `_prepare_memory_for()` returns `exclusive_idle` at the early gate — before any memory check runs — because loading B would contend with A's scheduler steps on the MLX executor.
9. **Wake and load**: Pin A (exclusive). Start request on A. Request B (waits). Finish A's request. Assert `exclusive_idle` fires, B loads successfully.
10. **Metal cache cleared before signal**: Mock `mx.clear_cache`. Assert it is called in `_signal_exclusive_idle` before `exclusive_idle.set()`.
11. **Re-wait on new VLM request**: Pin A (exclusive). B is waiting on `exclusive_idle`. A finishes (B wakes), but new A request arrives immediately. Assert B waits again on fresh `exclusive_idle` event.
12. **Timeout**: Pin A (exclusive) with long-running request. Request B. Assert B times out after `max_wait_timeout` with `ModelLoadingError`, not 507.
13. **exclusive_max_hold expiry**: Pin A (exclusive, max_hold=10s). A holds exclusivity for 15s. New A request arrives. Assert `_clear_for_exclusive()` is skipped, non-pinned models allowed to coexist.
14. **Hold timer reset**: Pin A (exclusive, max_hold=60s). A goes idle (active_uses → 0). Assert `_exclusive_hold_start` resets to 0. Next A request starts fresh hold timer.

### Integration tests

15. **End-to-end VLM → small → VLM cycle**: Start server with pinned exclusive VLM + small LLM. Send VLM request (completes). Send small LLM request (loads, completes). Send VLM request again (evicts small LLM, completes). Assert all three succeed.
16. **Concurrent VLM + small**: Send VLM request and small LLM request simultaneously. Assert small LLM waits for VLM to finish, then loads and completes. No 507.
17. **Latency under load**: Measure added latency for VLM requests when 0, 1, 2 non-pinned models need eviction. Ensure idle eviction adds < 1s, drain eviction adds < drain_timeout.
18. **Starvation guard**: Set `exclusive_max_hold: 5`. Send continuous VLM requests. Send small LLM request. Assert small LLM eventually gets served within ~5s of the hold expiry window.
19. **Event capture race**: Pin A (exclusive). Start request on A. B waits on `exclusive_idle`. A finishes → signal task created. BEFORE signal task runs, new A request arrives (`acquire_engine` creates fresh event). Assert OLD event (B's waiter) is still signaled correctly. Assert NEW event is un-set.

---

## Concurrency Review

Thorough analysis of deadlock, livelock, and race conditions — both introduced by this design and pre-existing.

### Locking model summary

The engine pool uses a single `asyncio.Lock` (`self._lock`) accessed via `_tracked_lock(caller)` with a 60s acquire timeout. All state mutations happen under this lock. Events are waited on **outside** the lock (INV-1). The lock is never held across an `await` on an event — callers release the lock, await, then re-acquire on the next loop iteration.

The MLX executor (`get_mlx_executor()`) is a global single-thread `ThreadPoolExecutor`. All Metal GPU operations must be serialized onto it. The event loop thread and the MLX executor thread are distinct — only the executor runs Metal operations.

### Issue #1: exclusive_idle event capture race (FIXED above)

**Severity**: HIGH — would cause non-pinned requests to block until `max_wait_timeout` (300s).

**The bug (in v2 draft)**: `_signal_exclusive_idle(entry)` reads `entry.exclusive_idle` after an async delay (`mx.synchronize + mx.clear_cache`). If `acquire_engine()` replaces the event during the delay, the old waiters never get signaled.

```
T0: VLM active_uses: 1→0. release_engine() creates task: _signal_exclusive_idle(entry)
T1: New VLM request. acquire_engine(): active_uses: 0→1.
    entry.exclusive_idle = asyncio.Event()  ← NEW event, overwrites old
T2: _signal_exclusive_idle() runs. Reads entry.exclusive_idle → gets NEW event.
    Sets NEW event (wrong!). Old event never set. Old waiters block 300s.
```

**Fix (v3)**: Capture the event reference at `release_engine()` time, pass directly to `_signal_exclusive_idle(event)`. The task signals the captured reference regardless of what `entry.exclusive_idle` points to later.

**Verification**: Even if `acquire_engine()` fires between T0 and T2, old waiters get signaled promptly. New waiters wait on the fresh event. No leak.

---

### Issue #2: _refresh_vision_limits() calling mx.clear_cache() on event loop thread (FIXED above)

**Severity**: MEDIUM — potential Metal race, corrupted vision budget.

**The bug (in v2 draft)**: `_refresh_vision_limits()` called `_init_vision_limits()` which calls `mx.clear_cache()` directly on the event loop thread (vlm.py:385). This is safe at initial load time (MLX executor idle) but NOT on every VLM request — the MLX executor may be running a scheduler step for another model's `EngineCore`.

`mx.clear_cache()` is documented as thread-safe (uses internal mutex), and only frees unreferenced buffers. So no crash risk. But calling it from the event loop while the executor is mid-operation may evict cached buffers the executor was about to reuse, causing unnecessary re-allocation churn and inaccurate `get_active_memory()` readings.

**Fix (v3)**: `_refresh_vision_limits()` computes the budget from `_committed_memory()` (purely arithmetic, under the lock, no Metal calls). This is more accurate post-eviction anyway — we know exactly what's loaded.

---

### Issue #3: get_engine() wait_target classification for new event types

**Severity**: LOW — logging and timeout extension logic.

The existing `get_engine()` wait path (line 658) classifies events as binary:

```python
wait_target = "ready_event" if event is entry.ready_event else "drain_complete"
```

The new `exclusive_idle` and `unload_complete` events from `_clear_for_exclusive()` get misclassified as `"drain_complete"`. This causes:

1. **Misleading logs**: "waiting on drain_complete" when actually waiting on exclusive_idle.
2. **Timeout extension logic** (lines 676-686): Checks `_find_draining_entry(event)` which searches for an entry whose `drain_complete` matches the event. Won't match `exclusive_idle` → extension doesn't trigger → normal timeout fires.

The second point is actually **correct behavior**: we WANT `exclusive_idle` waits to respect `max_wait_timeout`, not extend indefinitely. But the logs are confusing.

**Fix**: Generalize wait_target:

```python
if event is entry.ready_event:
    wait_target = "ready_event"
elif event is getattr(entry, 'exclusive_idle', None):
    wait_target = "exclusive_idle"  # from _clear_for_exclusive
else:
    # Could be drain_complete or unload_complete from
    # _clear_for_exclusive or _prepare_memory_for
    wait_target = "drain_or_unload"
```

Note: the `event` at this point might be from a *different* entry (the evicted model's `drain_complete`), not the requesting model's entry. So checking `entry.exclusive_idle` won't match in that case — but that's correct, the wait_target should be "drain_or_unload" for those.

For the case where `_check_process_memory()` returns an `exclusive_idle` event from a different entry, wait_target will be "drain_or_unload" which is imprecise but harmless for logging.

---

### Issue #4: _clear_for_exclusive() + _unload_engine() state mutation during iteration

**Severity**: LOW — functionally correct.

`_clear_for_exclusive()` iterates `self._entries.items()` and calls `_unload_engine()` for idle models within the same iteration. `_unload_engine()` mutates the entry's state (ACTIVE → UNLOADING) and sets `engine = None`, but does NOT add/remove keys from `_entries`. Python dict iteration is safe as long as the dict's size doesn't change during iteration. So this is correct.

After `_unload_engine()`, the loop continues to the next entry. If an entry we already visited changed state, we don't revisit it (forward iteration). If we haven't visited it yet, we'll see the updated state — but only already-processed entries get mutated. No issue.

---

### Issue #5: MLX executor contention with _signal_exclusive_idle()

**Severity**: LOW — latency, not deadlock.

`_signal_exclusive_idle()` submits `mx.synchronize() + mx.clear_cache()` to the MLX executor. If the executor is busy (model loading, scheduler step), the task queues behind. Waiters on `exclusive_idle` are delayed until the executor completes its current work.

This is not a deadlock because:
- The executor thread is making progress (not blocked on the event loop)
- The event loop thread is not blocked (the `_signal_exclusive_idle()` coroutine is awaiting, not blocking)
- The waiter in `get_engine()` is awaiting `event.wait()`, not holding the lock

**Worst case**: If a model load is in progress (~5-30s), the signal is delayed by that duration. Non-pinned requests wait longer. Bounded by the load duration + `max_wait_timeout`.

**Mitigation**: None needed. The executor is a serialization point by design (all Metal ops must be single-threaded). The delay is inherent to the architecture.

---

### Issue #6: Interaction between _clear_for_exclusive() and ProcessMemoryEnforcer

**Severity**: NONE — fully serialized by the lock.

Both `_clear_for_exclusive()` (called from `get_engine()` under lock) and `_check_and_enforce()` (acquires lock) can unload the same model. But they share the same `asyncio.Lock`, so they're serialized:

- If `_clear_for_exclusive()` unloads model B first, the enforcer sees B in UNLOADING/UNLOADED state and skips it.
- If the enforcer unloads model B first, `_clear_for_exclusive()` sees B in UNLOADING state and returns `unload_complete` for the caller to wait on.

No double-unload or conflict.

---

### Pre-existing issues — reviewed against current code

All four flagged issues were reviewed against the actual asyncio execution model and the codebase's established patterns. **None require code changes.**

#### Pre-1: _drain_monitor crash path mutates state without lock (line 1138) — FALSE POSITIVE

**Flagged concern**: Line 1138 sets `entry.state = EngineState.UNLOADED` without the lock inside the double-exception fallback. Could race with concurrent `get_engine()` readers.

**Verdict**: **Not a real bug.** Safe for three reasons:

1. **asyncio is single-threaded.** `entry.state = EngineState.UNLOADED` is a synchronous attribute assignment with no `await`. No other coroutine can interleave between read and write. Context switching only occurs at `await` points.

2. **Consistent with codebase pattern.** Naked field mutations outside the lock are used throughout:
   - `acquire_engine()` line 765: `entry.active_uses += 1` (no lock)
   - `release_engine()` line 774: `entry.active_uses -= 1` (no lock)
   - `_unload_engine()` line 1247: `entry.engine = None` (no lock for this assignment)
   
   The pool lock protects *compound decisions* (read state + decide + transition), not individual field writes.

3. **Waiters re-check under lock.** After `drain_complete.set()` (line 1143, always fires), the `get_engine()` waiter wakes, re-enters the while loop, acquires the lock (line 537), and re-checks `entry.state` (line 542/561). By that time, line 1138 has already completed.

4. **Intentional recovery design.** This path only executes when:
   - The main `_drain_monitor()` loop crashes with an exception (line 1124)
   - AND the lock-acquisition attempt at line 1128 itself times out or fails
   
   A bare state assignment is the correct last-resort fallback. Adding complexity here (non-blocking lock attempts, engine clearing) would make the crash recovery path more fragile, not less.

No fix needed.

---

#### Pre-2: Enforcer holds lock while calling abort_all_requests() (lines 266, 289) — FALSE POSITIVE

**Flagged concern**: `_check_and_enforce()` holds the pool lock while calling `await entry.engine.abort_all_requests()`, blocking `get_engine()` callers.

**Verdict**: **Not a real performance issue.** Despite being `async def`, `abort_all_requests()` contains zero awaits:

```python
# engine_core.py lines 409-445 — the full implementation:
async def abort_all_requests(self) -> int:
    request_ids = list(self._output_collectors.keys())
    for rid in request_ids:
        self.scheduler.abort_request(rid)  # → set.add(rid), O(1)
        collector = self._output_collectors.get(rid)
        if collector is not None:
            collector.put(RequestOutput(...))  # queue put, O(1)
        event = self._finished_events.get(rid)
        if event is not None:
            event.set()  # signal, O(1)
    self._pending_cleanups.update(request_ids)
    return len(request_ids)
```

- **No awaits**: The function runs synchronously to completion despite `async def`.
- **No Metal operations**: No `mx.synchronize()`, no GPU calls.
- **O(N) where N = active requests**: `scheduler.abort_request(rid)` is just `self._pending_abort_ids.add(request_id)` (scheduler.py line 3178).
- **Total lock hold**: ~2-10ms for a typical case (10-50 active requests). The 60s lock-acquire timeout is orders of magnitude larger.

**Moving abort outside the lock would introduce TOCTOU bugs:** `_unload_engine()` Phase 1 sets `entry.engine = None` under the lock. If we released the lock before calling `abort_all_requests()`, the engine reference would already be None. We'd need to capture it before unload — but then `_deferred_engine_cleanup()` could `stop()` and garbage-collect the engine while we're aborting on it.

The current code is correct: check `entry.engine is not None` (line 264), call `abort_all_requests()` (line 266), then `_unload_engine()` (line 276) — all under one lock hold, with the heavy cleanup deferred. This is exactly the intended two-phase unload design.

No fix needed.

---

#### Pre-3: Drain extends indefinitely when active_uses stuck (lines 1090-1100) — ACCEPTED RISK

**Flagged concern**: If `release_engine()` is never called (leaked acquire), `active_uses` stays > 0 and `_drain_monitor()` extends the drain forever.

**Verdict**: Real scenario but bounded by existing mechanisms and not worsened by this design.

- `get_engine()` waiters time out at `max_wait_timeout` (300s) — they are not blocked forever.
- The `_with_engine_guard()` helper in server.py (line 806) wraps all streaming responses and releases the engine in its finally block, covering normal completion, errors, and client disconnect.
- A truly "leaked" acquire would require a code bug in a request handler that bypasses `_with_engine_guard()` AND doesn't have a finally block. This is a programming error, not a runtime race.
- Adding a `DRAIN_ABSOLUTE_TIMEOUT` that force-unloads despite `active_uses > 0` would abort active user requests that are legitimately running — worse than waiting.

The exclusive eviction path uses the same drain mechanics. The VLM request waiter is bounded by `max_wait_timeout`. No additional fix needed.

---

#### Pre-4: active_uses modification without lock — FALSE POSITIVE

`acquire_engine()` and `release_engine()` modify `active_uses` without the lock. **Not a real issue** because:

1. **asyncio is single-threaded.** These are synchronous methods with no `await`. No interleaving is possible.
2. **The MLX executor thread does not touch `active_uses`.** Only event-loop code reads/writes it.
3. **The lock protects compound decisions** (read active_uses + read engine_has_active_work + decide to unload), not individual field writes.

No fix needed.

---

### Potential livelock analysis

#### Scenario A: VLM and non-pinned request chase each other

```
T0: VLM request → _clear_for_exclusive() → evicts SmallLLM
T1: SmallLLM request → _check_process_memory() → waits on exclusive_idle
T2: VLM finishes → exclusive_idle fires → SmallLLM loads
T3: New VLM request → _clear_for_exclusive() → drains SmallLLM
T4: SmallLLM request → _check_process_memory() → waits on exclusive_idle
... repeat
```

**Is this a livelock?** No — each cycle makes progress (both requests complete). The small model loads and serves between VLM requests. Throughput is reduced (small model reloads each cycle), but no starvation — both models get served.

**When does this degrade?** If VLM requests are faster than the load-serve-evict cycle for the small model, the small model's request latency grows (most time spent waiting/loading). But it still converges: each attempt loads the model and serves at least one request.

#### Scenario B: Multiple non-pinned models compete for the idle window

```
T0: VLM finishes → exclusive_idle fires
T1: SmallLLM-A and SmallLLM-B both wake, both enter _prepare_memory_for()
T2: SmallLLM-A acquires lock first, loads (enough memory for one model)
T3: SmallLLM-B re-checks: not enough memory → finds no victim → waits
```

What does SmallLLM-B wait on? In `_check_process_memory()`:
- `mx.get_active_memory()` shows VLM + SmallLLM-A
- No eviction candidates (SmallLLM-A just loaded, VLM is pinned)
- No exclusive model active (VLM has `active_uses == 0`)
- Falls through to committed-memory check or `InsufficientMemoryError`

**Risk**: SmallLLM-B gets a 507 if it can't fit alongside SmallLLM-A and the VLM. This is correct — the system genuinely doesn't have enough memory for three models. Not a livelock.

#### Scenario C: _clear_for_exclusive() never converges

Each call to `_clear_for_exclusive()` processes one model (evict or start drain) then returns. The `get_engine()` loop calls it again on re-entry. Could the set of non-pinned models change between iterations (new models loading)?

No — while the VLM's `get_engine()` is in the while loop, any new model load would require `_prepare_memory_for()` which acquires the lock. The VLM request is also acquiring the lock. They're serialized. Between lock acquisitions, no new model can transition to ACTIVE because loading requires the lock.

`_clear_for_exclusive()` converges in at most `N` iterations where N = number of non-pinned models (each iteration removes one from ACTIVE → DRAINING or UNLOADING). **No livelock.**

---

### Deadlock-freedom argument

The design uses a single lock + events pattern. For a deadlock to occur, one of these must hold:

1. **Nested lock acquisition**: The same `asyncio.Lock` acquired twice by the same coroutine. This doesn't happen — every lock acquisition is in a `_tracked_lock()` context manager, and the code never nests them (the lock is released before awaiting events, and re-acquired on the next loop iteration).

2. **Lock held while waiting on an event that requires the lock to be signaled**: This doesn't happen — events are always waited on OUTSIDE the lock (INV-1). The drain_complete event is signaled by `_drain_monitor()` which acquires the lock independently. The `exclusive_idle` event is signaled by `_signal_exclusive_idle()` which doesn't acquire the lock at all.

3. **Circular wait between lock and MLX executor**: The lock is never held while waiting on the MLX executor in the new code. `_signal_exclusive_idle()` runs as a free task (no lock). `_refresh_vision_limits()` is under the lock but does no Metal calls (uses `_committed_memory()`).

4. **Event never signaled**: All events have guaranteed signaling paths:
   - `drain_complete`: signaled in `_drain_monitor()`'s finally block (INV-3)
   - `unload_complete`: signaled in `_deferred_engine_cleanup()`'s finally block
   - `exclusive_idle`: signaled in `_signal_exclusive_idle()`'s finally block
   - All are additionally bounded by `max_wait_timeout` (300s) in `get_engine()`

**Conclusion**: No deadlocks introduced by this design. All pre-existing patterns (Pre-1 through Pre-4) were reviewed and confirmed safe under asyncio's single-threaded execution model.
