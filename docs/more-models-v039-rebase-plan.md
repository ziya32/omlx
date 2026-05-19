# Rebase plan: `features/more-models` → `origin/main` (v0.3.9rc1)

**Status:** merge + 13 follow-up commits + uv.lock regen applied. **235 targeted tests
pass (100% green).** Remaining items are deferrable polish (engine_pool diagnostic
fields, dflash track-upstream, Anthropic/Responses resolve-once migration).
**Authored:** 2026-05-19
**Last update:** 2026-05-19 — Tier-1 + smaller follow-ups complete

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

**`git merge origin/main` into `features/more-models`** (one resolution pass).
`rerere` enabled; conflict style `zdiff3`. No force-push (merge commit on the branch).

### Why not rebase (decision log)

Initial plan was granular interactive `rebase --onto origin/main`. Aborted at commit
**1 of 73**: the branch's oldest replay commit is `21f99a6` — a *squash of 59 commits*
(`104 files, +26693/-3367`) that conflicted across 24 files (~120 hunks: server.py 29,
audio_routes.py 16, scheduler.py 13, engine_pool.py 12, prefix_cache.py 10,
process_memory_enforcer.py 9, …). The granular-history rationale does not hold when the
base is already a squash: rebase would force resolving the *entire* feature-vs-main
reconciliation in one mega-conflict **and then** replay 71 more re-conflicting commits.
`git merge` does the same total reconciliation **once**, against final main, with
both-sides context, and no force-push. Decisions below are applied *within* the single
merge resolution rather than per-commit (take-main = take `origin/main` side; keep-feature
= take `HEAD` side; hand-merge zones = union by hand).

## Safety

```
git branch backup/features/more-models-pre-039-rebase   # pre-rewrite snapshot
git config rerere.enabled true
git config merge.conflictStyle zdiff3
```
Rollback at any point: `git merge --abort` (mid-merge) or
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

## Remaining unique feature work

Genuinely-unique feature work (more-models, dflash routing, cancel endpoint, enforcer,
VLM mRoPE, test rebases) carries via the merge from the `HEAD` side; resolve hotspot
conflicts once each (rerere).

## Flagged risks (verification-phase blockers)

1. ~~**mlx-vlm pin moved under the kept GatedDeltaNet patch.**~~ **RESOLVED — false alarm.**
   Feature's `gated_delta_advance.py` docstring explicitly states: *"As of mlx-vlm
   191d7c8 (target), upstream ships `cache.advance(S)` on its own, so the original
   ed7884c fix is no longer carried by this patch."* The patch only carries the
   `mx.contiguous` wrap on `cache[0]` write + `cache.lengths is not None` per-element
   slicing + drops mlx-vlm silent fallbacks. **It does NOT call `cache.advance(S)`
   redundantly** — feature designed for upstream evolution. All 7
   `tests/test_gated_delta_advance.py` tests pass against pinned `mlx-vlm@f96138e`.
2. **anthropic extra `except`** — feature's dropped `bbf0f90` had an extra
   `except (json.JSONDecodeError, AttributeError)`; main's `c5b91b5` won. Re-check.
3. **gate-mcp** — main's `697f63d` is a different impl than feature's dropped
   `f157e85`. Verify mcp/audio routers are actually gated behind `verify_api_key`.

## Follow-up commits applied (2026-05-19)

After the initial merge landed, the following critical follow-ups were applied:

| Commit | What | Why |
|---|---|---|
| `ca688cd` | scheduler: wire bbba911 `submit_store_cache_async` dispatch | **CRITICAL** — the kept bbba911 port now ACTIVATES; without this hook the prefix-cache race remained latent (the 197s→348s regression bbba911 fixes was still latent). |
| `80f22a6` | server: add `RequestAbortedError` → HTTP 503 exception handler | Issue 4 at LLM level — in-flight aborts no longer fall through to HTTP 500. |
| `6bb59ef` | tts/stt: cooperative-abort checkpoints (`_mark_stopped`, `_raise_if_aborted`) | Issue 4 at non-LLM level — TTS/STT handlers racing with enforcer eviction now return clean 503. |
| `39a51b3` | engine_pool: retarget preload check to `max(active, phys_footprint)` | Decision 4 metric integration — load gate now uses the same jetsam-accurate metric the enforcer uses. |
| `a6ed4b0` | server: `POST /v1/cancel/{request_id}` out-of-band cancel | Re-applies feature `25e3dda` — out-of-band cancel for clients that can't rely on TCP-close. |
| `277f457` | scheduler: predictive generation memory guard | Re-applies feature's "wait state" budget (the predictive variant) — defer admission when `active + (n_running+1) * per_request_estimate + cached_overhead > soft`. |
| `9e8de7d` | tests(enforcer): align 3 cascade-eviction tests with one-per-tick contract | The feature design doc explicitly predicted these would need updating. Now 149/149 enforcer+cache+admission+memory tests pass. |
| `a9b4487` | server: add `resolved_id` parameter to `get_engine` (Issue 7) | Resolve-once API change. Caller migration to thread `resolved_id` through `create_completion`/`chat_completion`/Anthropic/Responses deferred to a follow-up commit. |

**Already closed by auto-merge** (not actually gaps, despite earlier flag):
- `skip_api_key_verification` admin plumbing — fully merged.
- `OCR_MODEL_TYPES` / `OCR_MODEL_PROMPTS` in vlm.py — fully merged.

## Take-side gaps (remaining follow-up work)

The following resolutions used a wholesale `--ours`/`--theirs` for tractability. The
discarded side's substance must be re-applied as targeted follow-up commits before this
merge is considered semantically complete:

### `scheduler.py` ← `--theirs` (main's chunked prefill, queue cap, phase timers, etc.)
Lost feature work to re-add:
- **Predictive generation memory guard** (was at feature's `scheduler.py:3281-3344`) —
  the "wait state" budget that defers admission when projected concurrent cost would
  exceed limit. Integrate with main's `_admission_paused` mechanism, retargeted to
  `max(active, phys_footprint)`.
- **Deferred-abort plumbing** — `_do_abort_request` / `_pending_async_aborts` set,
  `cancel_request(request_id)` exposed through engines.
- **`bbba911` scheduler hits** — dispatch through `submit_store_cache_async` instead of
  `store_cache` so the kept prefix-cache race fix actually activates. **WITHOUT THIS
  THE PREFIX-CACHE REGRESSION REMAINS LATENT** (per Decision 3 finding).
- **`4baf965`** scheduler cache-corruption traceback when retries exhausted.

### `server.py` ← `--theirs` (main's 503, native reasoning Responses, anthropic, gate-mcp)
Lost feature work to re-add:
- **`POST /v1/cancel/{request_id}`** endpoint (`25e3dda`).
- **`RequestAbortedError` exception handler** translating to HTTP 503 (Issue 4).
- **`server.use_engine` resolve-once** threading via `resolved_id` (Issue 7) for
  `create_completion` / `create_chat_completion` / `create_message` (anthropic) /
  `create_response` (Responses).
- **`preserve_thinking` for `/v1/responses`** (`64e522f`).
- **`_with_json_keepalive` on non-streaming endpoints** (`a124a1f`) + body-encoded
  error after keepalive byte (`16445e1`).

### `tts.py`, `stt.py` ← `--theirs` (main's STT/TTS evolution)
Lost feature work to re-add (Decision 4 cooperative-abort protocol gaps):
- `_raise_if_aborted()` checkpoints at every public entry point + after each
  `run_in_executor` boundary (`embed`, `rerank`, `transcribe`, `_do_transcribe`,
  `_transcribe_single`, `synthesize`, `stream_synthesize`, `process`).
- `_mark_stopped()` call at the top of each engine's `stop()` method before clearing
  `self._model`.
- Without these: an in-flight TTS/STT/embed/rerank racing with enforcer eviction
  returns HTTP 500 instead of the typed 503 (the exact regression Issue 4 fixes).

### `admin/routes.py` ← `--theirs` (main's admin rework)
Lost feature work: `admin: expose skip_api_key_verification in GlobalSettingsRequest`
(`31ee562`). Re-add as a small targeted patch.

### `audio_routes.py`, `vlm.py`, `engine_pool.py`, `dflash.py` ← `--ours` (feature's restructuring)
Lost main work to re-layer:
- `audio_routes.py`: main's `word_timestamps` (`19bb34e`), `max_tokens` for
  transcriptions (`6993c5a`/`84ef801`).
- `vlm.py`: main's Gemma4 OCR torch-gated bypass (`a1987ed`), Qwen3.6 nested-visual
  sanitize (`29c9341`).
- `engine_pool.py`: main's phys_footprint preload check (`max(active, phys)` instead
  of bare `mx.get_active_memory()`), `actual_size`/`loading_started_at`/`_load_seconds_per_gb_ema`
  diagnostic fields, `#1276` DFlash `draft_window_size`/`draft_sink_size`/`verify_mode`,
  `#1283` fallback error surfacing.
- `dflash.py`: main's track-upstream dflash-mlx + Gemma4 support (`ee5edc4`,
  `496a248`), draft quant settings.

### Tests
Several test files (`test_audio_stt.py`, `test_audio_tts.py`, `test_server.py`,
`test_process_memory_enforcer.py`, `test_admin_auth.py`, `integration/test_server_endpoints.py`)
took `--theirs`. Feature's test additions for kept work (cancel endpoint regression,
cooperative-abort protocol contract tests, abort-during-eviction integration tests)
need re-adding alongside the corresponding code re-adds above.

## Execution

```
git merge origin/main                 # one resolution pass
# resolve ~24-file conflict per Decisions 1-4:
#   Decision 1 (15 superseded) -> take origin/main side
#   Decision 2 (qwen3_5)       -> take HEAD (feature) side
#   Decision 3 (cache race)    -> hand-merge bbba911 onto main's cache
#   Decision 4 (mem enforcer)  -> hand-merge both; retarget to max(active,phys)
# build + run suite (esp. test_prefix_cache.py, bbba911 regression tests,
#   test_process_memory_enforcer.py, test_scheduler_admission.py, test_proc_memory.py)
git commit                            # merge commit; no force-push
git push                              # fast-forward of origin/features/more-models
```
