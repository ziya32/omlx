# SPDX-License-Identifier: Apache-2.0
"""Live integration tests for grammar-constrained decoding.

Tests grammar correctness and measures performance across model families
(Qwen, Gemma, Harmony/OSS) against a self-spawned oMLX server.

By default the test bootstraps its own ``omlx serve`` subprocess on a
free port using ``OMLX_MODEL_DIR`` for model discovery.  One model per
family is auto-picked from the model directory; families without a
local match skip individually.

Override patterns (any combination):
  * ``OMLX_TEST_URL`` / ``OMLX_TEST_API_KEY`` — connect to an existing
    server instead of spawning one.
  * ``OMLX_TEST_GRAMMAR_MODEL_QWEN`` / ``..._GEMMA`` / ``..._OSS`` —
    pin a specific model name for that family (must exist in the model
    directory the server uses).

Run:
  pytest tests/test_grammar_live.py -v -s
  pytest tests/test_grammar_live.py -v -s -k perf   # performance only
"""

import asyncio
import atexit
import json
import os
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import pytest

pytestmark = [pytest.mark.slow, pytest.mark.integration]


def _load_omlx_settings() -> dict:
    """Read ~/.omlx/settings.json if present.  Returns {} on any failure."""
    base = Path(os.environ.get("OMLX_BASE_PATH") or Path.home() / ".omlx")
    settings_path = base / "settings.json"
    try:
        return json.loads(settings_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _resolve_base_url() -> str:
    if env := os.environ.get("OMLX_TEST_URL"):
        return env
    cfg = _load_omlx_settings().get("server", {})
    host = cfg.get("host") or "127.0.0.1"
    # 0.0.0.0 binds all interfaces — connect via loopback.
    if host in ("0.0.0.0", "::"):
        host = "127.0.0.1"
    port = cfg.get("port") or 8899
    return f"http://{host}:{port}"


def _resolve_api_key() -> str:
    if env := os.environ.get("OMLX_TEST_API_KEY"):
        return env
    return _load_omlx_settings().get("auth", {}).get("api_key") or ""


# ---------------------------------------------------------------------------
# Model discovery — pick one local model per family
# ---------------------------------------------------------------------------

# Substrings (lowercased model directory name) that identify each family.
# A model qualifies for the family if its name contains one of these AND
# its model_type indicates a generic chat-capable LLM (i.e. not a TTS,
# STT, embedding, or reranker variant).
_FAMILY_NAME_HINTS = {
    "qwen": ("qwen3", "qwen2"),
    "gemma": ("gemma",),
    "oss": ("gpt-oss", "gpt_oss"),
}

# Reasoning parser to set on each family's loaded model so the
# structural-tag tests exercise the grammar-aware reasoning pathway.
# Gemma family has no reasoning parser by design.
_FAMILY_REASONING_PARSER = {
    "qwen": "qwen",
    "gemma": None,
    "oss": "harmony",
}


def _discover_family_model(model_dir: Path, family: str) -> str | None:
    """Pick the smallest model in ``model_dir`` matching ``family``.

    Honours ``OMLX_TEST_GRAMMAR_MODEL_<FAMILY>`` env var as an override
    (returned as-is, no existence check — server will surface missing
    model errors at request time).
    """
    override = os.environ.get(f"OMLX_TEST_GRAMMAR_MODEL_{family.upper()}")
    if override:
        return override
    if not model_dir.is_dir():
        return None
    hints = _FAMILY_NAME_HINTS.get(family, ())
    candidates: list[tuple[int, str]] = []
    for sub in sorted(model_dir.iterdir()):
        if not sub.is_dir() or not (sub / "config.json").exists():
            continue
        lower = sub.name.lower()
        # Exclude non-chat variants (audio, embedding, reranker).
        if any(t in lower for t in ("embedding", "reranker", "asr", "tts", "sts")):
            continue
        if not any(h in lower for h in hints):
            continue
        try:
            cfg = json.loads((sub / "config.json").read_text())
        except Exception:
            continue
        # True VLMs (vision-conditional architectures) confuse the grammar
        # tests, which assume text-only chat completions.  Require a
        # non-vision architecture.
        archs = " ".join(cfg.get("architectures") or []).lower()
        if "vlforconditional" in archs or "visionforconditional" in archs:
            continue
        try:
            size = sum(f.stat().st_size for f in sub.iterdir() if f.is_file())
        except OSError:
            size = 0
        candidates.append((size, sub.name))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][1]


def _model_dir_for_server() -> Path:
    """Resolve the model directory the spawned server should serve from."""
    return Path(
        os.environ.get("OMLX_MODEL_DIR")
        or Path.home() / ".myemee" / "models"
    )


def _pick_free_port(default: int = 18899) -> int:
    """Pick a free TCP port; fall back to ``default`` if the OS won't bind."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]
    except OSError:
        return default


# ---------------------------------------------------------------------------
# Server bootstrap
# ---------------------------------------------------------------------------

# Resolved at module import: BASE_URL/API_KEY for the test session.  When
# the user supplied OMLX_TEST_URL, we connect to that.  Otherwise the
# session-scoped fixture below spawns a subprocess and overwrites these.
BASE_URL = _resolve_base_url()
API_KEY = _resolve_api_key()
_OMLX_USE_EXISTING = bool(os.environ.get("OMLX_TEST_URL"))

_SERVER_API_KEY = "test-grammar-live"
_SERVER_STARTUP_TIMEOUT = 180


@pytest.fixture(scope="module", autouse=True)
def _grammar_live_server():
    """Spawn an ``omlx serve`` subprocess for this test module.

    **Module-scoped** (not session-scoped): the spawned subprocess
    holds 30+ GB of GPU memory after loading two model families.  A
    session-scoped fixture would keep that memory pinned while later
    test modules — most notably ``test_exclusive_live_server.py`` and
    ``test_full_integration.py`` — spawn their own subprocesses or
    load models in-process, easily blowing past 64 GB system RAM.
    Module-scope guarantees teardown fires the moment the last
    grammar_live test in this file completes, freeing the memory
    before the next module starts.

    Skipped when ``OMLX_TEST_URL`` is set — assumes the user pointed at
    an externally-managed server with the right models pre-loaded.
    """
    global BASE_URL, API_KEY

    if _OMLX_USE_EXISTING:
        # External server — nothing to spawn, no models to configure.
        yield None
        return

    model_dir = _model_dir_for_server()
    if not model_dir.is_dir():
        pytest.skip(f"OMLX_MODEL_DIR not found: {model_dir}")

    port = _pick_free_port()
    base_path = Path(tempfile.mkdtemp(prefix="omlx-grammar-live-"))
    cmd = [
        sys.executable, "-m", "omlx", "serve",
        "--base-path", str(base_path),
        "--model-dir", str(model_dir),
        "--api-key", _SERVER_API_KEY,
        "--port", str(port),
    ]
    log_file = base_path / "server.log"
    log_fh = open(log_file, "w")
    proc = subprocess.Popen(
        cmd, stdout=log_fh, stderr=subprocess.STDOUT, text=True,
    )

    def _kill():
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        try:
            log_fh.close()
        except Exception:
            pass
        # Best-effort tempdir cleanup — leave it on failure for triage,
        # nuke it on clean teardown to keep /tmp tidy across runs.
        try:
            import shutil
            shutil.rmtree(base_path, ignore_errors=True)
        except Exception:
            pass

    # Backup cleanup if pytest crashes / SIGKILL — atexit fires only
    # on clean interpreter shutdown but at least covers normal exit.
    atexit.register(_kill)

    try:
        BASE_URL = f"http://127.0.0.1:{port}"
        API_KEY = _SERVER_API_KEY

        deadline = time.monotonic() + _SERVER_STARTUP_TIMEOUT
        healthy = False
        while time.monotonic() < deadline:
            try:
                r = httpx.get(f"{BASE_URL}/health", timeout=2.0)
                if r.status_code == 200:
                    healthy = True
                    break
            except (httpx.ConnectError, httpx.ReadTimeout):
                pass
            if proc.poll() is not None:
                log_text = log_file.read_text() if log_file.exists() else ""
                _kill()
                pytest.skip(
                    f"omlx server exited with code {proc.returncode}: "
                    f"{log_text[:500]}"
                )
            time.sleep(1)
        if not healthy:
            _kill()
            pytest.skip(f"omlx server not healthy within {_SERVER_STARTUP_TIMEOUT}s")

        # Configure reasoning_parser per family on the loaded models.  Done
        # after health-check so the server is fully initialised.
        for family, parser in _FAMILY_REASONING_PARSER.items():
            if parser is None:
                continue
            model = MODELS.get(family)
            if model is None or model == "<missing>":
                continue
            try:
                httpx.put(
                    f"{BASE_URL}/admin/api/models/{model}/settings",
                    json={"reasoning_parser": parser},
                    headers={"Authorization": f"Bearer {API_KEY}"},
                    timeout=10.0,
                )
            except Exception:
                pass

        yield proc
    finally:
        # Module teardown — kills subprocess + frees GPU memory before
        # the next test module starts loading its own models.
        _kill()


# Auto-discover one model per family at collection time.  Tests for
# families without a local match skip via the per-test ``skipif`` below.
_AUTO_MODEL_DIR = _model_dir_for_server()
MODELS = {
    family: _discover_family_model(_AUTO_MODEL_DIR, family) or "<missing>"
    for family in _FAMILY_NAME_HINTS
}


def _skip_if_family_missing(family: str):
    """pytest mark that skips when the family has no local model."""
    return pytest.mark.skipif(
        MODELS.get(family) in (None, "<missing>"),
        reason=f"No local model for family={family!r} in OMLX_MODEL_DIR — "
               f"set OMLX_TEST_GRAMMAR_MODEL_{family.upper()} to override.",
    )

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "city": {"type": "string"},
    },
    "required": ["name", "age", "city"],
    "additionalProperties": False,
}

REGEX_PATTERN = r"\d{4}-\d{2}-\d{2}"

PROMPT_JSON = "Give me a fictional person with name, age, and city."
PROMPT_REGEX = "What is today's date in YYYY-MM-DD format?"
PROMPT_PLAIN = "Write a short haiku about the ocean."

OSS_MAX_TOKENS = 400  # Harmony needs room for analysis + final channels


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _headers():
    return {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }


def _chat_payload(model, prompt, *, structured_outputs=None,
                  max_tokens=128, temperature=0.1, stream=False,
                  extra_body=None):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }
    if structured_outputs:
        if extra_body is None:
            extra_body = {}
        extra_body["structured_outputs"] = structured_outputs
    if extra_body:
        payload.update(extra_body)
    return payload


async def _complete(client, model, prompt, **kwargs):
    """Send a non-streaming chat completion and return (content, duration_s)."""
    payload = _chat_payload(model, prompt, **kwargs)
    t0 = time.perf_counter()
    resp = await client.post(f"{BASE_URL}/v1/chat/completions",
                             json=payload, headers=_headers(), timeout=120)
    dur = time.perf_counter() - t0
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"].get("content") or ""
    return content, dur


async def _complete_streaming(client, model, prompt, **kwargs):
    """Send a streaming chat completion and return (content, ttft_s, total_s, token_count).

    TTFT is measured to the first delta of any kind (content or
    reasoning_content).  Token count comes from the server-reported
    ``usage.completion_tokens`` when ``stream_options.include_usage``
    is set; falls back to counting content deltas.
    """
    payload = _chat_payload(model, prompt, stream=True, **kwargs)
    payload["stream_options"] = {"include_usage": True}
    t0 = time.perf_counter()
    ttft = None
    chunks = []
    token_count = 0
    server_tokens = None
    async with client.stream("POST", f"{BASE_URL}/v1/chat/completions",
                             json=payload, headers=_headers(), timeout=180) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str.strip() == "[DONE]":
                break
            chunk = json.loads(data_str)
            # Usage-only chunk (final)
            usage = chunk.get("usage")
            if usage and "completion_tokens" in usage:
                server_tokens = usage["completion_tokens"]
            choices = chunk.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            has_content = delta.get("content", "")
            has_reasoning = delta.get("reasoning_content", "")
            if has_content or has_reasoning:
                if ttft is None:
                    ttft = time.perf_counter() - t0
            if has_content:
                chunks.append(has_content)
                token_count += 1
    total = time.perf_counter() - t0
    final_tokens = server_tokens if server_tokens is not None else token_count
    return "".join(chunks), ttft or total, total, final_tokens


# Note: the old top-level "server reachable?" skip has been replaced by
# the autouse ``_grammar_live_server`` fixture which either spawns a
# subprocess or trusts an externally-managed server when
# ``OMLX_TEST_URL`` is set.  Per-family skipif marks (below) handle the
# case where a model for a given family isn't available locally.


# =========================================================================
# Integration Tests: Grammar Correctness
# =========================================================================

def _max_tokens_for(family, default=200):
    """Harmony models need more tokens for analysis + final channels."""
    return OSS_MAX_TOKENS if family == "oss" else default


class TestGrammarJson:
    """JSON schema grammar produces valid JSON for each model family."""

    @pytest.fixture()
    def client(self):
        return httpx.AsyncClient()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("family", [
        pytest.param("qwen", marks=_skip_if_family_missing("qwen")),
        pytest.param("gemma", marks=_skip_if_family_missing("gemma")),
        pytest.param("oss", marks=_skip_if_family_missing("oss")),
    ])
    async def test_json_schema(self, client, family):
        model = MODELS[family]
        content, dur = await _complete(
            client, model, PROMPT_JSON,
            structured_outputs={"json": JSON_SCHEMA},
            max_tokens=_max_tokens_for(family),
        )
        print(f"\n[{family}] JSON output ({dur:.2f}s): {content[:200]}")
        # Harmony may produce multiple final channels with repeated JSON;
        # decode only the first object.
        decoder = json.JSONDecoder()
        parsed, _ = decoder.raw_decode(content.lstrip())
        assert "name" in parsed, f"Missing 'name' in {parsed}"
        assert "age" in parsed, f"Missing 'age' in {parsed}"
        assert "city" in parsed, f"Missing 'city' in {parsed}"
        assert isinstance(parsed["age"], int), f"'age' is not int: {parsed['age']}"


class TestGrammarRegex:
    """Regex grammar produces matching output for each model family."""

    @pytest.fixture()
    def client(self):
        return httpx.AsyncClient()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("family", [
        pytest.param("qwen", marks=_skip_if_family_missing("qwen")),
        pytest.param("gemma", marks=_skip_if_family_missing("gemma")),
        pytest.param("oss", marks=_skip_if_family_missing("oss")),
    ])
    async def test_regex(self, client, family):
        import re
        model = MODELS[family]
        content, dur = await _complete(
            client, model, PROMPT_REGEX,
            structured_outputs={"regex": REGEX_PATTERN},
            max_tokens=_max_tokens_for(family, 50),
        )
        content = content.strip()
        print(f"\n[{family}] Regex output ({dur:.2f}s): {content}")
        # Harmony may produce multiple final channels whose content gets
        # concatenated, so check that the output starts with a valid match.
        assert re.match(REGEX_PATTERN, content), \
            f"Output '{content}' doesn't start with pattern '{REGEX_PATTERN}'"


class TestGrammarChoice:
    """Choice grammar restricts output to one of the given options."""

    @pytest.fixture()
    def client(self):
        return httpx.AsyncClient()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("family", [
        pytest.param("qwen", marks=_skip_if_family_missing("qwen")),
        pytest.param("gemma", marks=_skip_if_family_missing("gemma")),
        pytest.param("oss", marks=_skip_if_family_missing("oss")),
    ])
    async def test_choice(self, client, family):
        model = MODELS[family]
        choices = ["yes", "no", "maybe"]
        content, dur = await _complete(
            client, model, "Is the sky blue? Answer with yes, no, or maybe.",
            structured_outputs={"choice": choices},
            max_tokens=_max_tokens_for(family, 10),
        )
        content = content.strip().strip('"')
        print(f"\n[{family}] Choice output ({dur:.2f}s): {content}")
        # Harmony may produce multiple final channels; check that output
        # starts with a valid choice.
        assert any(content.startswith(c) for c in choices), \
            f"Output '{content}' doesn't start with any of {choices}"


class TestNoGrammar:
    """Baseline: unconstrained generation works for each model."""

    @pytest.fixture()
    def client(self):
        return httpx.AsyncClient()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("family", [
        pytest.param("qwen", marks=_skip_if_family_missing("qwen")),
        pytest.param("gemma", marks=_skip_if_family_missing("gemma")),
    ])
    async def test_plain(self, client, family):
        model = MODELS[family]
        content, dur = await _complete(
            client, model, PROMPT_PLAIN,
            max_tokens=100,
        )
        print(f"\n[{family}] Plain output ({dur:.2f}s): {content[:200]}")
        assert len(content.strip()) > 5, "Expected non-trivial output"

    @pytest.mark.asyncio
    @_skip_if_family_missing("oss")
    async def test_plain_oss(self, client):
        """OSS/Harmony needs more tokens; analysis channel may consume most of them."""
        model = MODELS["oss"]
        content, dur = await _complete(
            client, model, PROMPT_PLAIN,
            max_tokens=OSS_MAX_TOKENS,
        )
        print(f"\n[oss] Plain output ({dur:.2f}s): {content[:200]}")
        # Harmony may return empty content if the model doesn't reach
        # the final channel within max_tokens. This is expected behavior.
        if not content.strip():
            pytest.skip("Harmony model did not produce final channel content")


# =========================================================================
# Performance Benchmarks
# =========================================================================

BENCH_DURATION = int(os.environ.get("OMLX_BENCH_DURATION", "60"))
BENCH_MAX_TOKENS = 128
CONCURRENCY_LEVELS = [1, 2, 4]

BENCH_PROMPT = "Write a detailed description of a fictional city."
BENCH_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "city_name": {"type": "string"},
        "description": {"type": "string"},
        "population": {"type": "integer"},
        "landmarks": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["city_name", "description", "population", "landmarks"],
}


def _mean_std(values):
    if len(values) < 2:
        return (values[0] if values else 0.0), 0.0
    return statistics.mean(values), statistics.stdev(values)


@dataclass
class BenchResult:
    model: str
    grammar: str       # "none" or "json"
    thinking: str      # "on" or "off"
    concurrency: int
    durations: list = field(default_factory=list)
    ttfts: list = field(default_factory=list)
    token_counts: list = field(default_factory=list)

    @property
    def n(self):
        return len(self.durations)

    def ttft_stats(self):
        return _mean_std(self.ttfts)

    def dur_stats(self):
        return _mean_std(self.durations)

    def tps_stats(self):
        tps_list = [t / d for t, d in zip(self.token_counts, self.durations) if d > 0 and t > 0]
        return _mean_std(tps_list)


async def _run_one_bench(client, model, grammar, thinking, family):
    """Run a single streaming request and return (ttft, duration, tokens)."""
    so = {"json": BENCH_JSON_SCHEMA} if grammar == "json" else None
    extra = {}
    if thinking == "off":
        extra["chat_template_kwargs"] = {"enable_thinking": False}
        extra["thinking_budget"] = 0

    max_tok = _max_tokens_for(family, BENCH_MAX_TOKENS)

    _, ttft, total, tokens = await _complete_streaming(
        client, model, BENCH_PROMPT,
        structured_outputs=so,
        max_tokens=max_tok,
        temperature=0.7,
        extra_body=extra if extra else None,
    )
    return ttft, total, tokens


async def _bench_timed(model, grammar, thinking, concurrency, duration, family):
    """Run requests for *duration* seconds at the given concurrency."""
    result = BenchResult(
        model=model, grammar=grammar, thinking=thinking, concurrency=concurrency,
    )
    sem = asyncio.Semaphore(concurrency)
    stop = asyncio.Event()
    pending: set = set()

    async def _worker(client):
        async with sem:
            if stop.is_set():
                return
            try:
                ttft, dur, tokens = await _run_one_bench(
                    client, model, grammar, thinking, family,
                )
                result.durations.append(dur)
                result.ttfts.append(ttft)
                result.token_counts.append(tokens)
            except Exception as e:
                pass  # skip failed requests

    async def _dispatcher(client):
        while not stop.is_set():
            task = asyncio.create_task(_worker(client))
            pending.add(task)
            task.add_done_callback(pending.discard)
            # Small sleep to avoid tight-looping; the semaphore throttles actual concurrency.
            await asyncio.sleep(0.01)

    async with httpx.AsyncClient() as client:
        dispatcher = asyncio.create_task(_dispatcher(client))
        await asyncio.sleep(duration)
        stop.set()
        dispatcher.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    return result


def _fmt(mean, std):
    if std < 0.005:
        return f"{mean:.3f}"
    return f"{mean:.3f}\u00b1{std:.3f}"


def _print_results(results):
    hdr = (f"  {'Model':<25} {'Think':>5} {'Gram':>5} {'Conc':>4} {'Reqs':>5} "
           f"{'TTFT (s)':>14} {'Dur (s)':>14} {'TPS':>14}")
    print(hdr)
    print(f"  {'-'*25} {'-'*5} {'-'*5} {'-'*4} {'-'*5} {'-'*14} {'-'*14} {'-'*14}")
    for r in results:
        tm, ts = r.ttft_stats()
        dm, ds = r.dur_stats()
        pm, ps = r.tps_stats()
        print(f"  {r.model:<25} {r.thinking:>5} {r.grammar:>5} {r.concurrency:>4} {r.n:>5} "
              f"{_fmt(tm, ts):>14} {_fmt(dm, ds):>14} {_fmt(pm, ps):>14}")


class TestPerformance:
    """Time-boxed performance benchmarks.

    For each model, runs requests for OMLX_BENCH_DURATION seconds (default 60)
    at each concurrency level, with and without grammar, with thinking on/off.
    Reports mean +/- stdev for TTFT, duration, and TPS.
    """

    @staticmethod
    async def _warmup(model):
        async with httpx.AsyncClient() as c:
            await _complete(c, model, "Hi", max_tokens=5)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("family", [
        pytest.param("qwen", marks=_skip_if_family_missing("qwen")),
        pytest.param("gemma", marks=_skip_if_family_missing("gemma")),
        pytest.param("oss", marks=_skip_if_family_missing("oss")),
    ])
    async def test_perf(self, family):
        model = MODELS[family]

        # Gemma has no reasoning_parser so thinking on/off is irrelevant
        think_modes = ["off"] if family == "gemma" else ["on", "off"]

        print(f"\n  Warming up {model}...")
        await self._warmup(model)

        results = []
        total_combos = len(think_modes) * 2 * len(CONCURRENCY_LEVELS)
        done = 0
        for thinking in think_modes:
            for grammar in ["none", "json"]:
                for conc in CONCURRENCY_LEVELS:
                    done += 1
                    label = f"think={thinking} grammar={grammar} conc={conc}"
                    print(f"  [{done}/{total_combos}] {label} "
                          f"({BENCH_DURATION}s)...", end="", flush=True)
                    r = await _bench_timed(
                        model, grammar, thinking, conc, BENCH_DURATION, family,
                    )
                    print(f" {r.n} reqs")
                    results.append(r)

        print()
        _print_results(results)

        # Grammar overhead analysis
        print()
        for thinking in think_modes:
            base = [r for r in results
                    if r.grammar == "none" and r.thinking == thinking and r.concurrency == 1]
            gram = [r for r in results
                    if r.grammar == "json" and r.thinking == thinking and r.concurrency == 1]
            if base and gram and base[0].n > 0 and gram[0].n > 0:
                bm, _ = base[0].dur_stats()
                gm, _ = gram[0].dur_stats()
                ratio = gm / max(bm, 0.001)
                print(f"  Grammar overhead (think={thinking}, conc=1): {ratio:.2f}x")
                assert ratio < 5.0, f"Grammar overhead too high: {ratio:.2f}x"
