# Rebase plan: `features/more-models` → `origin/main` (v0.3.9rc1)

**Status:** in progress
**Authored:** 2026-05-19

## Context

| Ref | SHA | Note |
|---|---|---|
| merge-base | `cb33a76` | "fix(patches): align mlx-vlm Qwen3.5/3.6 forward with mlx-lm cache semantics" |
| `origin/main` | `f6f4269` | v0.3.9rc1 — 214 commits ahead of merge-base (199 files, +38011/-4844) |
| `features/more-models` | `bbba911` | 72 commits ahead of merge-base (150 files, +35047/-3871) |

`git cherry` finds **0 of the 72 feature commits patch-equivalent** to anything on
`origin/main` — a plain `git rebase` replays all 72 and auto-skips none. Note: local
`main` is stale (105 behind `origin/main`); the rebase targets `origin/main`.

This is the next in the established `backup/features/more-models-pre-{036,038,rc1}-rebase`
sequence → **`pre-039-rebase`**.

## Strategy

Granular interactive `git rebase --onto origin/main <merge-base>`, preserving history.
`rerere` enabled; conflict style `zdiff3`. Squash-the-remainder is the documented
fallback if conflict volume becomes unmanageable.

## Safety

```
git branch backup/features/more-models-pre-039-rebase   # pre-rewrite snapshot
git config rerere.enabled true
git config merge.conflictStyle zdiff3
```
Rollback at any point: `git rebase --abort` (mid-rebase) or
`git reset --hard backup/features/more-models-pre-039-rebase`.

## Decision 1 — DROP 15 superseded commits (take main's copy)

| Class | Commit | Subject | Why drop |
|---|---|---|---|
| identical | `00076e6` | fix(gemma4): drop stray channel close, delegate tool-call markup | patch-identical to main `23bd304` |
| identical | `a7b8b6a` | fix(api): accept JSON-object string from native tool parsers | patch-identical to main `1cd4531` |
| identical | `c85b9a1` | [codex] add Russian admin localization (#977) | patch-identical to main `84dd857` |
| identical | `b768715` | chore(audio): drop one-off TTS demo script | patch-identical to main `2d2d1a9` |
| identical | `1858edd` | fix(uploader): match oQ models by substring | patch-identical to main `42749b3` |
| identical | `5673c31` | fix(updater): pick latest stable release (#981) | patch-identical to main `3351134` |
| version | `c75c7cd` | chore: bump version to 0.3.8 | superseded; main at 0.3.9rc1 |
| version | `f252917` | chore: bump version to 0.3.8rc1 | superseded |
| version | `7844f15` | chore: bump version to 0.3.8.dev3 | superseded |
| differ→main | `22440be` | perf(cache): Async store_cache + SSD evict unlink | near-identical to main `af97a0f`; main canonical |
| differ→main | `0b7f930` | fix(gemma4): Preserve audio_tower in oQ output | near-identical to main `5a55eb0` |
| differ→main | `bbf0f90` | fix(anthropic): repair text/thinking transitions | same fix; main `c5b91b5`. **Post-rebase: re-check feature's extra `except (json.JSONDecodeError, AttributeError)`** |
| differ→main | `74e5922` | fix: clean errors for non-LLM/STT (#826) | main `2549237` is superset |
| differ→main | `ec1de8d` | Add native TTS streaming (#951) | main `f7d136a` is superset |
| differ→main | `f157e85` | fix(server): gate mcp/audio behind verify_api_key | main `697f63d` is a **different impl**. **Post-rebase: verify routers actually gated** |

## Decision 2 — KEEP feature's qwen3_5 work (ahead of main)

Feature's `f9f9d9e` (skip per-layer GPU syncs) is *more advanced* than main's `89ee4a9`
(adds `force_text_only_rope()` + `scalar_offset` branch); main never evolved it further.
**Keep & replay:** `f9f9d9e` + `81d383c` (drop bug-masking fallbacks) + `0790ac5`
(original-call fallback).

## Decision 3 — Cache prefix-race: PORT `bbba911`, do not drop

`bbba911` (split `store_cache` → sync register + async write) fixes a race **introduced
by `22440be`**, which we drop in favor of main's `af97a0f`. But main's `af97a0f`
submits the *whole* `store_cache` (incl. in-memory block-hash registration) to the
background executor — the **same root defect** `bbba911` fixes. Main's `#1298`
(`e4d07b8`, pending-write-buffer) fixes an *unrelated* hot-cache↔SSD race in
`paged_ssd_cache.py` and does **not** cover this.

→ Main carries the prefix-cache regression latent. `bbba911` will not apply cleanly
(`prefix_cache.py` +385 / `scheduler.py` +91 both heavily rewritten on main: N-tuple V3
`5b00c42`, `af97a0f` executor wiring). **Re-express the Phase-1-sync / Phase-2-async
split against main's `BlockAwarePrefixCache` + `af97a0f` scheduler. Keep its regression
tests** (`TestRegression_22440be_BackToBackPrefixCacheHit`). `cb15a40` (test-mock fix)
likely moot — main's `af97a0f` already adds `_pending_async_removes`/`_drain_*`; verify
against main's `test_abort_drain.py`, reconcile or drop.

## Decision 4 — Memory enforcer: KEEP BOTH, complementary (#1 reconcile zone)

Main and the feature branch solve **different halves** and neither subsumes the other.

- **Feature branch has NO `phys_footprint`** (grep `phys_footprint|proc_pid_rusage|RUSAGE_INFO|jetsam|libproc` → empty). It budgets on `mx.get_active_memory()` alone — blind to the **95 GB** IOAccelerator/Metal gap (31B+32k) that main's `37c73a0` exists to close. Feature needs main's metric.
- **Main lacks** eviction-time correctness: one-eviction-per-tick, cooperative abort on non-LLM engines (embed/rerank/stt/tts/sts), abort-pinned-in-flight, clean streaming-abort SSE, resolve-once. Feature has all of these (`docs/enforcer-eviction-review.md`, `102fe6b`).

| Capability | main v0.3.9rc1 | feature |
|---|---|---|
| `phys_footprint` jetsam metric | ✅ core | ❌ absent |
| 2-watermark soft/hard + `get_pressure_level()` | ✅ | ❌ single `_max_bytes` |
| Queue cap → `SchedulerQueueFullError` → 503+Retry-After | ✅ | ❌ |
| Chunked prefill bounds prefill peak (#1224) | ✅ | ❌ |
| Predictive generation memory guard (defer/WAIT) | ❌ | ✅ `scheduler.py:3281` |
| Eviction-race hardening, all engine types | partial | ✅ extensive |
| Cooperative abort on non-LLM engines | ❌ LLM-only | ✅ `BaseNonStreamingEngine` |
| Abort pinned-model in-flight, keep weights | ❌ | ✅ `102fe6b` |
| One-eviction-per-tick (deferred-cleanup-aware) | ❌ | ✅ Issue 1 |
| Clean streaming-abort SSE (no operator-hint leak) | ❓ | ✅ Issue 5 |

**Plan:**
1. Take main's memory feature as base — drop none: `37c73a0`, `196d667`, `11e6ea7`, `c003b2e`, `d736bfd`, `2dcc53a`.
2. Replay feature enforcer-hardening on top — additive: `102fe6b` + Issue-1/2/3/4/5/7 (`engine/base.py` cooperative-abort protocol, `ensure_engine_alive`/`EngineEvictedError`, one-per-tick, typed-503, clean SSE, resolve-once).
3. **Critical:** retarget feature's eviction/guard logic to main's `max(active, phys_footprint)` combined metric (`process_memory_enforcer.py:156`), not bare `mx.get_active_memory()`.
4. Hand-merge overlapping methods (not either-or): `_check_and_enforce` (main 2-watermark + feature one-per-tick/uniform-abort/pinned-abort), `exceptions.py` (union `SchedulerQueueFullError` + `EngineEvictedError`), `scheduler.py` (main admission-pause/queue-cap + feature generation-guard/deferred-abort), `server.py` (both 503 paths), `memory_monitor.py` (main KV+SDPA estimator), `engine/base.py` (feature net-new).
5. Verify feature Issue 5 (sanitized abort error / valid `finish_reason`) against main's abort path during resolve.

## Conflict hotspots (85 files changed on both sides)

| File | main churn | feat churn |
|---|---|---|
| `omlx/scheduler.py` | 2872 | 1046 |
| `omlx/server.py` | 582 | 2798 |
| `omlx/engine_pool.py` | 133 | 2172 |
| `omlx/cache/prefix_cache.py` | 887 | 645 |
| `omlx/engine/vlm.py` | 441 | 1044 |
| `omlx/api/audio_routes.py` | 413 | 1034 |
| `omlx/engine/dflash.py` | 738 | 247 |
| `omlx/process_memory_enforcer.py` | 237 | 324 |
| `omlx/engine/tts.py` · `stt.py` | high | high |

## Remaining ~54 commits

Genuinely-unique feature work (more-models, dflash routing, cancel endpoint, enforcer,
VLM mRoPE, test rebases). Replay as-is; resolve hotspot conflicts once each (rerere).

## Execution

```
git rebase -i --onto origin/main cb33a76 features/more-models   # drop the 15 via GIT_SEQUENCE_EDITOR
# resolve hotspots; reconcile Decision 3 & 4 zones by hand
# build + run suite (esp. test_prefix_cache.py, bbba911 regression tests,
#   test_process_memory_enforcer.py, test_scheduler_admission.py, test_proc_memory.py)
git push --force-with-lease
```
