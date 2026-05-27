# v0.3.12 Integration: `features/more-models` ← upstream `v0.3.12`

**Status:** merge complete; default unit suite **5038 passed, 0 failed**
(136 slow/integration deselected). Slow + integration (real-model) sweep in progress.
**Authored:** 2026-05-26

## Context

The fork last merged upstream at `cb7effd` (v0.3.9-1). Upstream then shipped
**v0.3.10 → v0.3.12** — 36 commits, mostly stability fixes, including a **rewrite
of the memory subsystem** (drop the `max_*_memory` sliders → `memory_guard_tier`
dropdown + dynamic ceiling + adaptive prefill throttle + Metal-cap clamp). Goal:
merge all v0.3.12 fixes, dedup against the feature branch's own stability work,
keep the suite green.

| Ref | SHA | Note |
|---|---|---|
| merge-base | `cb7effd` | origin/main, v0.3.9-1 (already on the feature branch) |
| incoming | `v0.3.12` = `1fb36ce` | 36 commits (`cb7effd..v0.3.12`); upstream/main `786dfe2` is +1 Homebrew bump, excluded |
| merge commit | `b8ca89d` | `git merge v0.3.12` (rerere + zdiff3), 19 conflicts |
| follow-ups | `ddf779a` (omlx prod), `36eea99` (tests) | post-merge regression fixes |
| backup | `backup/features/more-models-pre-v0312-merge` (`f97a11f`) | pre-merge tip |

**User decision:** adopt upstream's tier-based memory model (sliders → tier dropdown).

## Dedup decisions (A–G)

- **A. Memory enforcer** — adopt upstream's tier rewrite (`memory_guard_tier`,
  `get_final_ceiling()` = min(static, dynamic, metal_cap), `mx.set_wired_limit`,
  active-mem reclaim, `_walk_store_cache_caps`). Re-layered feature hardening onto
  `_check_and_enforce`: one-eviction-per-tick `break` (feature `_unload_engine` defers
  the heavy free), `_find_drain_or_evict_candidate` + busy-defer, pinned in-flight abort
  (102fe6b), peak observability (`_peak_memory_bytes`/`_overage_count`/`reset_peak`),
  `_model_size_bytes` plumbing. `self._max_bytes` → `ceiling` (3 sites).
- **B. Scheduler** — kept both memory guards (complementary phases): feature's
  decode-step admission guard + upstream's prefill `_adaptive_chunk_size` throttle.
  Reconciled the store-cache submit collision: feature's `submit_store_cache_async` +
  `_cleanup_after_phase2`, wrapped in upstream's `_StoreCacheGate` (#1383).
- **C. engine_pool** — kept the feature pool (`--ours`, avoids a Frankenstein auto-merge);
  removed the `max_model_memory` ctor param; added `_current_ceiling()` (callback +
  enforcer fallback); migrated all `_max_model_memory` reads **and** the
  `getattr(enforcer, "max_bytes", 0)` vision-limit/`_check_process_memory` sites
  (6, easy to miss — would have silently returned 0). Server sets `_get_final_ceiling`.
  **Kept the 25% admission KV headroom** (feature's own safety; reasoned deviation
  from the plan's "drop it", which targeted upstream's separate admission path).
- **D. Per-request MLX clear (#684) vs buffer lock** — adopted upstream's
  `base._finish_activity()` per-request clear across the 6 engines, but **routed its
  body through `locked_sync_and_clear_cache`** so it can't bypass the buffer lock.
  Kept feature `run_locked` loads, `locked_free_and_clear` stops, abort protocol,
  TTS `_next_seg`. `paged_ssd_cache`: kept the lock + added `ssd_write_drops` (1b666af).
- **E. VLM/DFlash/MTP/engine_core** — `--ours` + inject for vlm.py (d673ea3 `for_vlm=True`
  call-site, 7d640c1 whole-request cache read-back; 4ba94e6 Qwen3.6 MoE sanitize +
  d673ea3 `for_vlm` param auto-merged in model_loading.py). dflash.py hand-merged
  (915190d lifecycle self-heal + d0f60ec multimodal fallback grafted onto the feature's
  track-upstream `start`/`stop`/`_evict`, preserving the #85 free-race). engine_core.py
  auto-merged cleanly: feature's `AsyncEngineCore.scheduler` property **and** 3ec2016
  guards both present; kept the sanitized `public_error`. server.py: `is_dflash_vlm`
  re-injected on both extraction blocks.
- **F. settings** — dropped `max_*_memory` + `get_max_model_memory_bytes` +
  `_adaptive_system_reserve`; kept `memory_guard_tier`, `DiscoverySettings`, host=0.0.0.0.
- **G. Clean upstream takes** — integrations env-scrub, ms-downloader, hf xet,
  tool_calling, responses tag-free (#1348), oq processor_config, admin UI, anthropic
  stream indices, version → 0.3.12.

## Resolution polarity (19 conflicts)

| Polarity | Files |
|---|---|
| upstream-base + re-layer feature | `process_memory_enforcer.py`, `settings.py`, `cli.py`, `admin/routes.py` (memory UI) |
| feature-base + re-inject upstream (`--ours`) | `engine_pool.py`, `engine/vlm.py`, `server.py` |
| per-hunk combine | `scheduler.py`, `engine/{base,embedding,reranker,sts,stt,tts}.py`, `engine_core.py`, `engine/dflash.py` |
| tests | `test_settings.py` (theirs), `test_engine_pool.py`/`test_process_memory_enforcer.py`/`test_embedding.py` (combine) |

## Post-merge regression fixes (`ddf779a`, `36eea99`)

Production (`ddf779a`): scheduler `import threading` (feature dropped it when it moved
to the shared buffer lock; upstream's `_StoreCacheGate` needs it); enforcer
`_walk_store_cache_caps` every tick (was gated to the ok-branch); the 6
`getattr(enforcer, "max_bytes", 0)` → `get_final_ceiling()` sites; `extract_thinking`
`start_in_thinking` param restored (lost to the #1348 auto-merge; server passes it for
native-reasoning, default False keeps #1348's tag-free→content); F401 cleanup.

Tests (`36eea99`): `EnginePool(max_model_memory=X)` → `_make_pool(ceiling=X)` helper
(7 files); enforcer mocks → `.get_final_ceiling.return_value`; cache-clear patch targets
→ `base.locked_sync_and_clear_cache` (per-request) / `engine.<x>.locked_free_and_clear`
(stop — the old gc+locked_sync patches were a latent feature-branch test bug);
`ModelTooLargeError.max_memory` → `.ceiling`; `pool._current_model_memory` → property;
`test_ms_downloader` AsyncMock import.

## Verification

- **Default unit sweep:** `pytest -q -m "not slow and not integration"` → **5038 passed, 0 failed**.
- **20/20 critical modules import clean** against the merged tree.
- **Slow + integration:** `pytest -m "slow or integration"` (136 real-model tests on
  `/Volumes/SD-1TB/hot-models`). Early results: reranker/ASR/TTS e2e + grammar_live PASS.
  Known watch items: `test_asr_long_audio` setup ERROR (module-scoped `server_app` +
  `loop_scope="session"` async fixture — likely pre-existing pytest-asyncio fragility);
  `test_json_schema[qwen]` flaky (sampling variance, passes on retry).

## Rollback

`git reset --hard backup/features/more-models-pre-v0312-merge` (pre-merge tip), or
`git reset --hard b8ca89d` to keep the merge but drop the follow-up fixes.
