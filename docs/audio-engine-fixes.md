# Audio Engine Fixes Plan

This document covers the implementation plan for fixing all identified issues with audio engine support in omlx, organized by priority.

## Background

Audio engines (ASR/TTS) were added in Phase 0 (`b724c54`) and correctly follow the `BaseNonStreamingEngine` pattern with proper MLX executor usage. However, several gaps remain in TTL safety, executor contention, API completeness, and observability.

## How Concurrent Multi-Model Requests Work Today

All MLX GPU operations go through a **single-threaded executor** (`ThreadPoolExecutor(max_workers=1)` in `engine_core.py`). This is required because MLX uses a module-level Metal stream -- concurrent GPU ops from different threads cause Metal command buffer races and segfaults.

**LLM-to-LLM concurrency** works well: each model's `EngineCore._engine_loop()` calls `scheduler.step()` on the executor (generating one token per active request via continuous batching), then yields via `await asyncio.sleep(0)`. Multiple models naturally interleave at token boundaries.

**Audio-to-LLM concurrency is the gap**: TTS does `list(model.generate(...))` as a single executor submission -- holding it for the entire synthesis (potentially 10+ seconds). All LLM `scheduler.step()` calls queue behind it. ASR has the same problem with `generate_transcription()`.

**Chunked yielding** (Phase 2) fixes this by splitting the monolithic executor hold into N small holds (one per audio segment), letting LLM steps interleave between segments.

---

## Phase 1: Active Audio Operation Tracking

**Severity:** HIGH
**Problem:** `check_ttl_expirations()` (`engine_pool.py:668-675`) only checks `BatchedEngine` for active requests via `_output_collectors`. Audio engines mid-transcription/synthesis are not detected as active and can be evicted, corrupting the in-flight operation.

### Changes

**`engine/tts.py` + `engine/asr.py`:**
- Add `self._active_operations: int = 0` to `__init__`
- Add `@property def active_operations(self) -> int`
- Increment before `run_in_executor` in `synthesize()`/`transcribe()`, decrement in `finally`
- Include in `get_stats()`

**`engine_pool.py` (line 668):**
- Extend active-work detection:
  ```python
  # After the existing BatchedEngine check:
  elif hasattr(entry.engine, 'active_operations'):
      if entry.engine.active_operations > 0:
          has_active = True
  ```

---

## Phase 2: Chunked Yielding for Executor Contention

**Severity:** HIGH
**Problem:** TTS `_synthesize_sync()` calls `list(model.generate(...))` which collects ALL segments before returning, holding the MLX executor for the entire duration and blocking all LLM token generation.

### Technical Approach

`model.generate()` is already a Python generator that yields audio segments one at a time. Instead of consuming it eagerly, process one segment per executor submission:

```python
async def synthesize(self, ...):
    loop = asyncio.get_running_loop()

    # Step 1: Create generator on executor (quick)
    gen = await loop.run_in_executor(get_mlx_executor(), _create_generator)

    # Step 2: Consume one segment at a time, yielding executor between each
    segments = []
    while True:
        seg = await loop.run_in_executor(get_mlx_executor(), lambda: next(gen, None))
        if seg is None:
            break
        segments.append(seg)

    # Step 3: Concatenate and encode WAV (quick)
    return await loop.run_in_executor(get_mlx_executor(), _finalize, segments)
```

Each `next(gen)` generates one audio segment (~0.5-2s of work), then releases the executor so LLM `scheduler.step()` calls can run.

**ASR:** Keep as single executor call for now -- `generate_transcription()` is not a generator, and transcription is typically much faster than TTS synthesis.

### Files
- `engine/tts.py` -- replace `_synthesize_sync` with chunked async pattern

---

## Phase 3: Streaming TTS Output

**Severity:** HIGH
**Problem:** Client must wait for entire audio to be generated before playback. mlx-audio's `model.generate()` yields segments that could be streamed.

### Changes

**`engine/tts.py`:**
- Add `async def stream_synthesize()` that yields `SpeechOutput` per segment (builds on Phase 2's chunked architecture)
- Each segment is individually `mx.eval`'d and converted to WAV/PCM on the executor

**`server.py`:**
- Add `stream` query parameter to `POST /v1/audio/speech`
- When streaming, return `StreamingResponse` with raw PCM chunks
- Keep non-streaming WAV as default for backward compatibility

**WAV header consideration:** Streaming raw WAV requires either:
- A WAV header with unknown data length (`0xFFFFFFFF`) upfront (not all clients handle this)
- Raw PCM format (no header) with `audio/pcm` content type
- Recommended: support streaming only for `response_format=pcm`, keep non-streaming for WAV

---

## Phase 4: Memory Headroom Fix

**Severity:** MEDIUM
**Problem:** 25% KV-cache headroom (`engine_pool.py:319`) applied to all models, but audio/embedding/reranker models don't use KV caches.

### Change

**`engine_pool.py` (line 319):**
```python
if entry.model_type in ("asr", "tts", "embedding", "reranker"):
    kv_headroom = 0
else:
    kv_headroom = int(entry.estimated_size * 0.25)
```

---

## Phase 5: OpenAI API Gaps

**Severity:** MEDIUM

### 5a. Audio Format Conversion (response_format)

Currently only WAV is supported. Add mp3/opus/flac via `ffmpeg` subprocess (widely available on macOS).

**Files:**
- `api/audio_models.py` -- validate accepted formats
- `engine/tts.py` -- add `_convert_audio(wav_bytes, format)` using `ffmpeg` subprocess
- `server.py` -- pass format through, set correct `media_type` in response

### 5b. Speed Parameter

Check if `mlx-audio`'s `model.generate()` accepts a speed parameter. If not, post-process via sample rate manipulation (change sample rate without resampling for simple pitch-preserving speedup, or use `librosa` for proper time stretching).

### 5c. Verbose JSON Transcription Response

Preserve segment data from `result.segments` in `TranscriptionOutput` (currently discarded).

**Files:**
- `api/audio_models.py` -- add `TranscriptionSegment` and `VerboseTranscriptionResponse` models
- `engine/asr.py` -- include `segments` in `TranscriptionOutput`
- `server.py` -- add `response_format` form field; return verbose/text/srt/vtt as appropriate

### 5d. Prompt Parameter for Whisper

Add `prompt: str | None` parameter to `transcribe()`, pass as `initial_prompt` to `generate_transcription()` if supported by mlx-audio.

**Files:**
- `engine/asr.py` -- add `prompt` param
- `server.py` -- extract `prompt` from form data

---

## Phase 6: Audio-Specific Model Settings

**Severity:** MEDIUM
**Problem:** No per-model audio configuration (default voice, language, etc.) unlike LLM settings.

### Changes

**`model_settings.py`:**
```python
default_voice: Optional[str] = None
default_instruct: Optional[str] = None
default_language: Optional[str] = None
default_response_format: Optional[str] = None
```

**`server.py`:**
- In `create_speech()`, read model settings for default voice/instruct when request uses defaults
- In `create_transcription()`, read model settings for default language

**`admin/routes.py`:**
- Add audio fields to `ModelSettingsRequest`

---

## Phase 7: Enhanced Stats/Telemetry

**Severity:** MEDIUM
**Problem:** `get_stats()` returns only `{"model_name", "loaded"}` -- no operational metrics.

### Changes

**`engine/tts.py` + `engine/asr.py`:**
- Add counters: `_total_operations`, `_total_audio_seconds`, `_total_processing_seconds`
- Update in `synthesize()`/`transcribe()`
- Expose in `get_stats()`

**`server.py`:**
- Include audio stats in `/api/status` response

---

## Phase 8: ASR get_languages()

**Severity:** LOW
**Problem:** TTS exposes `get_speakers()` but ASR has no equivalent for listing supported languages.

### Changes

**`engine/asr.py`:**
- Add `get_languages()` method that queries the tokenizer for supported language tokens

**`server.py`:**
- Add `GET /v1/audio/languages` endpoint or include in an existing discovery endpoint

---

## Phase 9: Error Handling

**Severity:** LOW
**Problem:** Audio errors surface as raw mlx-audio exceptions without context.

### Changes

**`exceptions.py`:**
```python
class AudioError(OMLXError): ...
class InvalidAudioFormatError(AudioError): ...
class VoiceCloningError(AudioError): ...
```

**`engine/tts.py` + `engine/asr.py`:**
- Wrap `model.generate()` / `generate_transcription()` with specific error handling
- Validate ref_audio for voice cloning, audio file format for transcription

**`server.py`:**
- Catch audio exceptions, return 400/422 with descriptive messages

---

## Phase 10: Admin Panel Audio Configuration

**Severity:** LOW
**Problem:** Admin UI has no audio-specific settings.

### Changes

**`admin/templates/dashboard/_modal_model_settings.html`:**
- Add "asr" and "tts" to model type dropdown
- Add conditional audio settings section (default voice dropdown, default language)

**`admin/static/js/dashboard.js`:**
- Fetch speakers/languages dynamically when an audio model is selected

---

## Implementation Order

```
Sprint 1:  Phase 1 (TTL fix) + Phase 4 (headroom)       -- small, high impact, no deps
Sprint 2:  Phase 2 (chunked yielding) + Phase 7 (stats)  -- core concurrency fix
Sprint 3:  Phase 3 (streaming TTS) + Phase 6 (settings)  -- builds on Phase 2
Sprint 4:  Phase 5 (API gaps)                             -- depends on Phase 3
Sprint 5:  Phase 8 + 9 + 10 (polish)                     -- low priority
Sprint 6:  Phase 11 (nanobot updates)                     -- depends on Phase 5
```

---

## Testing Plan

### Unit Tests

| Phase | Test | File |
|-------|------|------|
| 1 | `active_operations` increments during `synthesize()`, decrements after (including on error) | `tests/test_audio_engines.py` |
| 1 | `active_operations` increments during `transcribe()`, decrements after | `tests/test_audio_engines.py` |
| 1 | `check_ttl_expirations` skips audio engine with `active_operations > 0` | `tests/test_engine_pool.py` |
| 1 | `check_ttl_expirations` unloads audio engine with `active_operations == 0` after TTL | `tests/test_engine_pool.py` |
| 2 | TTS chunked yielding calls executor multiple times (once per segment + finalize) | `tests/test_audio_engines.py` |
| 2 | TTS produces identical output to pre-chunking implementation | `tests/test_audio_engines.py` |
| 3 | `stream_synthesize()` yields multiple `SpeechOutput` chunks | `tests/test_audio_engines.py` |
| 3 | Each streamed chunk contains valid audio data | `tests/test_audio_engines.py` |
| 4 | Loading ASR/TTS model uses 0% headroom | `tests/test_engine_pool.py` |
| 4 | Loading LLM model still uses 25% headroom | `tests/test_engine_pool.py` |
| 5a | Format conversion produces valid MP3/FLAC/Opus bytes | `tests/test_audio_engines.py` |
| 5c | Verbose transcription response includes segments with timestamps | `tests/test_audio_engines.py` |
| 5d | `prompt` parameter passed through to `generate_transcription()` | `tests/test_audio_engines.py` |
| 6 | `ModelSettings` round-trips audio fields via `to_dict()`/`from_dict()` | `tests/test_settings.py` |
| 6 | Default voice from settings used when request voice is "default" | `tests/test_audio_engines.py` |
| 7 | Stats counters increment after operations | `tests/test_audio_engines.py` |
| 7 | `total_audio_seconds` and `total_processing_seconds` are positive | `tests/test_audio_engines.py` |
| 8 | `get_languages()` returns list for loaded Whisper model | `tests/test_audio_engines.py` |
| 9 | Invalid audio format raises `InvalidAudioFormatError` | `tests/test_audio_engines.py` |
| 9 | Missing `ref_audio` for base variant raises `VoiceCloningError` | `tests/test_audio_engines.py` |

### Integration Tests

| Phase | Test | File |
|-------|------|------|
| 1 | TTL does not evict TTS engine during active synthesis | `tests/integration/test_phase0_endpoints.py` |
| 2 | LLM inter-token latency stays below threshold during concurrent TTS | `tests/integration/test_phase0_e2e.py` |
| 3 | Streaming TTS endpoint returns chunked data incrementally | `tests/integration/test_phase0_endpoints.py` |
| 5c | `response_format=verbose_json` returns segments in transcription | `tests/integration/test_phase0_endpoints.py` |
| 5a | `response_format=mp3` returns valid MP3 audio | `tests/integration/test_phase0_endpoints.py` |

### Executor Contention Tests

| Test | Description | File |
|------|-------------|------|
| Pre-chunking baseline | Measure LLM token latency while TTS runs (expect high latency) | `tests/test_mlx_thread_safety.py` |
| Post-chunking | Same measurement after Phase 2 (expect latency drops to near-normal) | `tests/test_mlx_thread_safety.py` |
| Concurrent TTS + ASR | Verify both complete without deadlock or corruption | `tests/test_mlx_thread_safety.py` |

### Manual/Smoke Tests

- Admin panel: load a TTS model, verify settings modal shows audio options (Phase 10)
- Admin panel: Active Models card shows audio engine with active request count (Phase 1)
- End-to-end: `curl POST /v1/audio/speech` returns valid WAV while LLM chat is active (Phase 2)
- End-to-end: streaming TTS via `curl --no-buffer` shows incremental data (Phase 3)

---

## Nanobot Impact Assessment

Nanobot calls omlx audio APIs via LiteLLM (`litellm.aspeech()`, `litellm.atranscription()`). The critical integration files are:

- `myemee/knowledge/service.py` -- `transcribe()` and `generate_speech()` methods
- `myemee/gateway/handlers.py` -- `audio.transcribe` and `audio.generate` request handlers
- `myemee/config/schema.py` -- `LocalModelConfig` with TTS/ASR model field definitions
- `myemee/agent/tools/voice.py` -- voice selection, cloning, and preview tools

### Changes that DO NOT require nanobot updates

All phases in this plan are **backward compatible** -- no existing request/response contracts change:

| Phase | Impact | Reason |
|-------|--------|--------|
| 1 (TTL fix) | None | Internal engine pool logic only |
| 2 (Chunked yielding) | None | Internal executor scheduling only |
| 3 (Streaming TTS) | None | Streaming is opt-in via new `stream` param; default behavior unchanged |
| 4 (Memory headroom) | None | Internal memory accounting only |
| 5a (response_format) | None | New formats are additive; `wav` remains default |
| 5b (speed) | None | Existing param now works; no contract change |
| 5c (verbose_json) | None | New `response_format` option; default `json` unchanged |
| 5d (prompt) | None | New optional form field |
| 6 (Model settings) | None | Server-side defaults; API contract unchanged |
| 7 (Stats) | None | Additional fields in stats responses (additive) |
| 8 (get_languages) | None | New endpoint, doesn't affect existing ones |
| 9 (Error handling) | None | More descriptive error messages; HTTP status codes unchanged |
| 10 (Admin panel) | None | UI-only changes |

### Nanobot Changes (Phase 11)

Once the omlx audio fixes are implemented, nanobot should be updated to take advantage of the new capabilities. All changes are additive — existing behavior is preserved as the default.

#### 11a. Verbose Transcription

**`myemee/knowledge/service.py` — `transcribe()`:**
- Add `verbose: bool = False` parameter
- When verbose, pass `response_format="verbose_json"` to `litellm.atranscription()`
- Return structured result with `text`, `language`, `duration`, and `segments` (each with `start`, `end`, `text`)
- Default behavior unchanged (returns plain text string)

**`myemee/gateway/handlers.py` — `_audio_transcribe()`:**
- Add `verbose` parameter to handler
- When verbose, return `{"text": str, "segments": [...], "language": str, "duration": float}` instead of `{"text": str}`

#### 11b. Audio Format Options

**`myemee/knowledge/service.py` — `generate_speech()`:**
- Add `response_format: str = "wav"` parameter
- Pass through to `litellm.aspeech()` extra body
- Return bytes in requested format

**`myemee/gateway/handlers.py` — `_audio_generate()`:**
- Add `format` parameter (default "wav")
- Return `{"audio_base64": str, "format": str}` with actual format used

#### 11c. Default Voice Settings via omlx Model Settings

**`myemee/knowledge/omlx_process.py`:**
- When generating `model_settings.json` for omlx startup, include audio-specific defaults:
  - `default_voice` from `LocalModelConfig.tts_speaker`
  - `default_language` from `LocalModelConfig.asr_default_language`
- This lets omlx apply defaults server-side, reducing per-request boilerplate

#### 11d. Streaming TTS (future consideration)

Streaming TTS via `stream=true` on `/v1/audio/speech` would reduce time-to-first-audio for voice responses. However, this requires changes to the `_tts_and_emit()` pipeline to handle chunked audio events. Deferred until streaming is validated on the omlx side.

### Nanobot Test Updates

Tests need to be updated to cover the new parameters and verify backward compatibility.

#### Unit Tests — `tests/test_voice.py`

| Test | Description |
|------|-------------|
| `test_generate_speech_with_format_mp3` | Verify `generate_speech(format="mp3")` passes `response_format` to litellm |
| `test_generate_speech_default_format_wav` | Verify default format remains "wav" |
| `test_audio_generate_handler_format` | Verify `_audio_generate` handler passes format through and returns it in response |
| `test_transcribe_verbose` | Verify `transcribe(verbose=True)` passes `response_format="verbose_json"` to litellm |
| `test_transcribe_default_not_verbose` | Verify default transcribe still returns plain text |
| `test_audio_transcribe_handler_verbose` | Verify `_audio_transcribe` handler returns segments when verbose |

#### Handler Tests — `tests/gateway/test_knowledge_handlers.py`

| Test | Description |
|------|-------------|
| `test_audio_generate_format_param` | Verify format parameter is extracted and passed through |
| `test_audio_transcribe_verbose_param` | Verify verbose parameter is extracted and passed through |

#### E2E Tests — `tests/e2e/test_voice_e2e.py`

| Test | Description |
|------|-------------|
| `test_tts_mp3_format` | Request MP3 format, verify response has MP3 magic bytes (`0xFF 0xFB` or ID3 header) |
| `test_transcribe_verbose_segments` | Transcribe with verbose=true, verify segments have start/end timestamps |
| `test_omlx_default_voice_settings` | Verify omlx model_settings.json includes default_voice from LocalModelConfig |

---

## Code Review Notes (2026-03-15)

Reviewed all plan phases against current implementation. Most of Phases 1-4, 5c-5d, 6-8, and 11a-11c are already implemented. Findings below.

### Stale Plan

The plan reads as entirely future work, but the following are already implemented in code:

| Phase | Status | Evidence |
|-------|--------|----------|
| 1 (TTL fix) | Done | `_active_operations` counter + `_engine_has_active_work()` in `engine_pool.py:407-425` |
| 2 (Chunked yielding) | Done | `synthesize()` in `tts.py:211-244` uses chunked `next(gen)` pattern |
| 3 (Streaming TTS) | Done | `stream_synthesize()` in `tts.py:253-315`, `stream` query param in `server.py:1674` |
| 4 (Memory headroom) | Done | `engine_pool.py:320-323` sets `kv_headroom=0` for asr/tts/embedding/reranker |
| 5c (Verbose transcription) | Done | `VerboseTranscriptionResponse`, segments in `TranscriptionOutput`, verbose_json in `server.py:1643-1659` |
| 5d (Prompt parameter) | Done | `prompt` param in `asr.py:85`, passed as `initial_prompt` at line 136 |
| 6 (Model settings) | Partially done | `default_voice`/`default_instruct`/`default_language` in `ModelSettings` and applied in `server.py:1697-1707`; admin API can read but not write (see below) |
| 7 (Stats) | Done | Counters in both engines' `get_stats()` |
| 8 (get_languages) | Partially done | `get_languages()` exists in `asr.py:176-193` but no endpoint exposes it |
| 11a-11c (Nanobot) | Done | `verbose` param, `response_format` passthrough, voice cloning all in nanobot handlers |

**Action:** Update the plan with completion status per phase so it accurately reflects current state.

### Bugs

**1. `SpeechRequest.response_format` accepted but silently ignored**
- `audio_models.py:79` accepts any string (`response_format: str = "wav"`)
- `server.py:create_speech()` never reads `request.response_format` — always returns WAV
- A client sending `response_format=mp3` gets WAV back with no error
- [Ola] Fixed: added validation that returns 400 for non-wav formats

**2. `SpeechRequest.speed` accepted but silently ignored**
- `audio_models.py:82-83`: `speed: float = 1.0` with comment "reserved for future use, currently ignored"
- A client sending `speed=2.0` gets normal-speed audio with no feedback
- [Ola] Fixed: added validation that returns 400 for non-1.0 speed values

**3. `SpeakersResponse.languages` always empty**
- `server.py:1785` hardcodes `languages=[]` in the speakers endpoint
- `ASREngine.get_languages()` exists and works but is never called
- This means Phase 8's data is available but not exposed anywhere
- [Ola] Fixed: speakers endpoint now populates languages from a loaded ASR engine if available

### Gaps

**4. Admin API can't set audio defaults**
- `ModelSettingsRequest` (`admin/routes.py:74-94`) is missing `default_voice`, `default_instruct`, `default_language`, `default_response_format` fields
- `update_model_settings()` (`admin/routes.py:1323-1490`) has no logic to apply these fields
- Settings are read in the model list response (`routes.py:1250-1253`) but can't be written through the admin API
- The only way to set audio defaults is by editing `model_settings.json` directly
- [Ola] Fixed: added four fields to `ModelSettingsRequest` and corresponding `if "field" in sent:` blocks in update handler

**5. Admin UI has no audio controls**
- `_modal_model_settings.html` and `dashboard.js` have zero references to audio settings (default voice, default language, etc.)
- Phase 10 is entirely unimplemented
- Not blocking — audio defaults can be set via `model_settings.json` or the REST API
- [Ola] Acknowledged, deferred — low priority, REST API now works for setting defaults

**6. Audio exceptions defined but never raised**
- `exceptions.py:378-399` defines `AudioError`, `InvalidAudioFormatError`, `VoiceCloningError`, `UnsupportedOutputFormatError`
- Neither `tts.py` nor `asr.py` imports or raises any of them
- `server.py` doesn't catch `AudioError`
- Raw mlx-audio exceptions propagate as 500 Internal Server Error
- [Ola] Fixed: `synthesize()`, `stream_synthesize()`, and `transcribe()` now wrap
  mlx-audio exceptions with `AudioError` subclasses. `VoiceCloningError` raised for
  base-variant failures, `InvalidAudioFormatError` for bad audio files, generic
  `AudioError` for other failures. Server catches these and returns 400 (bad input)
  or 422 (processing failure) instead of 500.

**7. No `GET /v1/audio/languages` endpoint**
- Phase 8 says to add this endpoint
- `ASREngine.get_languages()` is implemented but has no server route
- [Ola] Fixed: added `GET /v1/audio/languages` endpoint that loads the ASR model on
  demand and returns supported language codes. Speakers endpoint reverted to TTS-only
  (no longer conflates TTS speakers with ASR languages).

**8. Phase 5a/5b not implemented**
- No `_convert_audio()` function, no ffmpeg subprocess for format conversion
- Speed parameter not implemented
- [Ola] Fixed: full format conversion and speed control implemented.
  - `response_format` supports wav, mp3, opus, flac, pcm via ffmpeg subprocess
  - `speed` supports 0.25-4.0 via ffmpeg atempo filter (chained for extreme values)
  - Unsupported formats and out-of-range speed return 400
  - ffmpeg availability checked at conversion time with descriptive error

### Optimization Opportunities

**9. `stream_synthesize()` generates WAV per segment then strips header**
- Each streamed segment goes through `_audio_to_wav()` (adds 44-byte WAV header) in `tts.py:296`
- Server immediately strips it at `server.py:1719` (`chunk.audio_bytes[44:]`)
- [Ola] Fixed: added `_audio_to_pcm()` that returns raw PCM bytes. `stream_synthesize()` now uses it directly, server yields bytes as-is.

**10. `synthesize()` concatenates even for single segment**
- `_finalize()` (`tts.py:233-242`) always calls `mx.concatenate([s.audio for s in segs])` even when there's only one segment
- [Ola] Fixed: `audio = segs[0].audio if len(segs) == 1 else mx.concatenate([s.audio for s in segs])`

**11. `stream_synthesize()` uses closures inconsistently with `synthesize()`**
- `synthesize()` uses a named function `_next_seg(g)` that takes the generator as parameter (correct approach, avoids late-binding issues)
- `stream_synthesize()` uses `lambda: next(gen, None)` which captures `gen` from enclosing scope and creates a new function object each iteration
- Both work correctly since `gen` doesn't change, but they should be consistent
- [Ola] Fixed: `stream_synthesize()` now uses the `_next_seg` pattern

**12. ASR blocks executor for entire transcription**
- `_do_transcribe()` runs as a single executor submission
- For long audio (5+ minutes), this blocks all LLM token generation
- `generate_transcription()` is not a generator so chunked yielding isn't directly possible
- [Ola] Fixed: ASR now uses per-chunk executor yielding with 60s chunk duration.
  `_do_transcribe()` loads the audio and splits it via Qwen3-ASR's
  `split_audio_into_chunks(chunk_duration=60.0)`, which finds silence boundaries
  within a ±5s search window. Each chunk is processed in a separate
  `run_in_executor()` call, yielding the executor between chunks so other
  engines (LLM, embedding) can interleave.

  E2E validation (`tests/integration/test_asr_long_audio.py`) transcribes a
  7-hour interview while firing concurrent embedding requests:

  | Metric | 20-min chunks (before) | 60s chunks (after) |
  |--------|----------------------|-------------------|
  | Total time | 38 min | 24 min |
  | Segments | 29 | 374 |
  | Transcript length | 136K chars | 141K chars |
  | Worst embedding latency | 125.8s | 40.4s (model loading) |
  | Typical embedding latency | 125s | 2-3s |

  Quality is unaffected — silence-boundary splitting avoids mid-sentence cuts,
  and Qwen3-ASR processes chunks independently regardless of duration.
  For audio ≤ 60s, the single-call fast path is used with no overhead.

### Nanobot Integration Notes

**13. `VoiceConfig.SPEAKERS` hardcoded**
- `schema.py` hardcodes `SPEAKERS = ["Vivian", "Serena", "Ryan", "Aiden"]`
- If speakers change upstream in the TTS model, nanobot's validation list becomes stale
- The speakers endpoint (`/v1/audio/speakers`) returns the actual list from the model
- [Ola] Valid observation but low risk — these are Qwen3-TTS preset speakers which are baked into the model weights and won't change. If a new TTS model with different speakers is added, both the omlx model config and nanobot config would need updating anyway. Not worth the complexity of dynamic fetching.

**14. `_tts_and_emit()` doesn't use streaming**
- `handlers.py:612-628` generates complete audio before emitting the WebSocket event
- Phase 11d acknowledges this as a future improvement
- [Ola] Acknowledged, intentionally deferred. Streaming audio over WebSocket requires chunked events and client-side audio buffering/playback — significant frontend work for marginal benefit given typical TTS latency is 1-3 seconds.
