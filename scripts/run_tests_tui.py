#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""TUI test runner for omlx with a fixed status bar, live failure panel,
and scrolling output.

Runs three suites by default:
  1. unit         — tests/ minus tests/integration, ``-m "not slow and not integration"``
  2. slow         — tests/ minus tests/integration, ``-m "slow and not integration"``
  3. integration  — tests/integration/ (all markers)

Failures are written to ``test-reports/logs_<timestamp>/`` as each test
finishes (see ``_tui_failure_plugin.py``):
  - ``index.md``                     — top-level run summary
  - ``failures.md``                  — rolling markdown report
  - ``failures/<nodeid>.md``         — per-failure traceback
  - ``<suite>.md``                   — per-suite summary
  - ``<suite>.log``                  — full pytest stdout
  - ``<suite>.xml``                  — junit-xml

Usage:
    python scripts/run_tests_tui.py                    # everything
    python scripts/run_tests_tui.py --no-slow          # skip slow phase
    python scripts/run_tests_tui.py --no-integration   # skip integration phase
    python scripts/run_tests_tui.py --only unit        # one phase

Keys:
    q / Ctrl-C   cancel the run
    ↑ / ↓        scroll log
    PgUp/PgDn    page log
    g / Home     scroll to top
    G / End      follow tail
"""

from __future__ import annotations

import argparse
import asyncio
import curses
import os
import re
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent           # scripts/
REPO_ROOT = SCRIPT_DIR.parent                          # repo root
PYTHON = str(REPO_ROOT / ".venv" / "bin" / "python")
PLUGIN_NAME = "_tui_failure_plugin"                    # importable from scripts/


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class SuiteResult:
    name: str
    status: str = "pending"
    duration: float = 0.0
    summary: str = ""
    log_path: str = ""


@dataclass
class State:
    suites: list[SuiteResult] = field(default_factory=list)
    current_suite: str = ""
    current_test: str = ""
    last_test: str = ""
    last_result: str = ""
    overall_pct: int = 0
    suite_index: int = 0
    total_suites: int = 0
    start_time: float = 0.0
    last_output_time: float = 0.0
    failed_tests: list = field(default_factory=list)
    output_lines: deque = field(default_factory=lambda: deque(maxlen=10000))
    scroll_offset: int = 0
    done: bool = False
    cancelled: bool = False
    exit_code: int = 0
    report_dir: Path | None = None
    _proc: object = None

    def log(self, line: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S.%f")[:12]
        self.output_lines.append(f"{ts} {line}")
        self.last_output_time = time.time()


# ---------------------------------------------------------------------------
# Suite definitions
# ---------------------------------------------------------------------------

def build_suites(args, report_dir: Path) -> list[tuple[str, list[str]]]:
    common = [
        PYTHON, "-u", "-m", "pytest",
        "--override-ini=addopts=",
        "-p", PLUGIN_NAME,
        "--tb=long",
        "-ra",
        "-p", "no:cacheprovider",
        "-v",
    ]

    suites = []

    if args.only in (None, "unit"):
        suites.append(("unit", common + [
            "tests/",
            "--ignore=tests/integration",
            "-m", "not slow and not integration",
        ]))

    if args.only in (None, "slow") and not args.no_slow:
        suites.append(("slow", common + [
            "tests/",
            "--ignore=tests/integration",
            "-m", "slow and not integration",
        ]))

    if args.only in (None, "integration") and not args.no_integration:
        suites.append(("integration", common + [
            "tests/integration/",
            "-m", "",  # all markers
        ]))

    return suites


# ---------------------------------------------------------------------------
# Pytest output parser
# ---------------------------------------------------------------------------

# tests/foo.py::TestBar::test_baz PASSED                             [ 42%]
RE_RESULT = re.compile(
    r"^(tests/.+?::.+?)\s+(PASSED|FAILED|ERROR|XFAIL|SKIPPED|XPASS)\b[^[]*\[\s*(\d+)%\]"
)
RE_TEST_START = re.compile(r"^(tests/\S+::\S+)\s*$")
RE_RESULT_ONLY = re.compile(
    r"^(PASSED|FAILED|ERROR|XFAIL|SKIPPED|XPASS)\b[^[]*\[\s*(\d+)%\]"
)
RE_SECTION = re.compile(r"^={3,}\s+(.+?)\s+={3,}\s*$")


def shorten_test_name(full: str) -> str:
    parts = full.split("::")
    if len(parts) >= 3:
        return f"{parts[-2]}::{parts[-1]}"
    if len(parts) == 2:
        return parts[-1]
    return full


# ---------------------------------------------------------------------------
# Curses TUI
# ---------------------------------------------------------------------------

def format_duration(secs: float) -> str:
    s = int(secs)
    if s >= 3600:
        return f"{s // 3600}h{(s % 3600) // 60:02d}m{s % 60:02d}s"
    if s >= 60:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s}s"


def draw(stdscr, state: State):
    try:
        height, width = stdscr.getmaxyx()
    except Exception:
        return
    if height < 4 or width < 20:
        return

    # ── Status bar ──────────────────────────────────────────────────
    elapsed = time.time() - state.start_time if state.start_time else 0
    idle = time.time() - state.last_output_time if state.last_output_time else 0

    def _trunc(s: str, maxlen: int) -> str:
        return s if len(s) <= maxlen else s[: maxlen - 2] + ".."

    max_name = max(15, (width - 60) // 2)

    parts = [f"[{state.overall_pct}%]"]
    if state.current_suite:
        parts.append(state.current_suite)
    if state.current_test:
        parts.append(f"▶ {_trunc(state.current_test, max_name)}")
    if state.last_test:
        parts.append(
            f"{_trunc(state.last_test, max_name)} {state.last_result} "
            f"| {format_duration(idle)} ago"
        )
    parts.append(format_duration(elapsed))
    status_text = " | ".join(parts)
    if len(status_text) > width - 1:
        status_text = status_text[: width - 4] + "..."

    if state.done:
        color = curses.color_pair(1) if state.exit_code == 0 else curses.color_pair(2)
    elif idle >= 300:
        color = curses.color_pair(2)
    elif idle >= 60:
        color = curses.color_pair(3)
    else:
        color = curses.color_pair(4)

    try:
        stdscr.move(0, 0)
        stdscr.clrtoeol()
        stdscr.addnstr(0, 0, status_text, width - 1, color | curses.A_BOLD)
    except curses.error:
        pass

    row = 1

    # ── Failures panel ──────────────────────────────────────────────
    n_failures = len(state.failed_tests)
    if n_failures > 0:
        sep = "─" * (width - 1)
        try:
            stdscr.addnstr(row, 0, sep, width - 1, curses.color_pair(5))
        except curses.error:
            pass
        row += 1

        header = f" {n_failures} FAILED "
        try:
            stdscr.move(row, 0)
            stdscr.clrtoeol()
            stdscr.addnstr(
                row, 0, header, width - 1,
                curses.color_pair(2) | curses.A_BOLD,
            )
        except curses.error:
            pass
        row += 1

        remaining = max(0, height - 2 - row)
        will_truncate = n_failures > min(10, remaining)
        budget = (remaining - 1) if will_truncate else remaining
        n_show = min(10, n_failures, max(0, budget))
        visible = state.failed_tests[-n_show:] if n_show else []
        hidden = n_failures - n_show

        for ft in visible:
            display = f"  ✗ {ft}"
            if len(display) > width - 1:
                display = display[: width - 4] + "..."
            try:
                stdscr.move(row, 0)
                stdscr.clrtoeol()
                stdscr.addnstr(row, 0, display, width - 1, curses.color_pair(2))
            except curses.error:
                pass
            row += 1

        if hidden > 0:
            more = f"  ... and {hidden} older"
            try:
                stdscr.move(row, 0)
                stdscr.clrtoeol()
                stdscr.addnstr(row, 0, more, width - 1, curses.color_pair(5))
            except curses.error:
                pass
            row += 1

    sep = "─" * (width - 1)
    try:
        stdscr.move(row, 0)
        stdscr.clrtoeol()
        stdscr.addnstr(row, 0, sep, width - 1, curses.color_pair(5))
    except curses.error:
        pass
    row += 1

    # ── Scrolling output ────────────────────────────────────────────
    log_height = height - row
    if log_height <= 0:
        return

    total_lines = len(state.output_lines)
    if state.scroll_offset == 0:
        start = max(0, total_lines - log_height)
    else:
        start = max(0, total_lines - log_height - state.scroll_offset)

    for i in range(log_height):
        line_idx = start + i
        try:
            stdscr.move(row + i, 0)
            stdscr.clrtoeol()
            if line_idx < total_lines:
                line = state.output_lines[line_idx]
                if len(line) > width - 1:
                    line = line[: width - 4] + "..."
                line_color = 0
                if " PASSED " in line:
                    line_color = curses.color_pair(1)
                elif " FAILED " in line or " ERROR " in line:
                    line_color = curses.color_pair(2)
                elif line.startswith("══"):
                    line_color = curses.color_pair(4) | curses.A_BOLD
                stdscr.addnstr(row + i, 0, line, width - 1, line_color)
        except curses.error:
            pass

    stdscr.noutrefresh()


# ---------------------------------------------------------------------------
# Async runner
# ---------------------------------------------------------------------------

async def run_suite(
    name: str,
    cmd: list[str],
    state: State,
    log_dir: Path,
) -> int:
    state.suite_index += 1
    state.current_suite = name
    state.current_test = ""
    state.last_test = ""
    state.last_result = ""

    suite_result = next(s for s in state.suites if s.name == name)
    suite_result.status = "running"
    log_path = log_dir / f"{name}.log"
    suite_result.log_path = str(log_path)

    junit_xml = log_dir / f"{name}.xml"
    cmd = cmd + [f"--junitxml={junit_xml}"]

    header = f"\n{'═' * 60}\n  [{state.suite_index}/{state.total_suites}] {name}\n{'═' * 60}\n"
    for line in header.splitlines():
        state.log(line)

    t0 = time.time()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    # Plugin lives in scripts/ — add it to PYTHONPATH so pytest -p can import.
    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{SCRIPT_DIR}{os.pathsep}{pythonpath}" if pythonpath else str(SCRIPT_DIR)
    )
    env["OMLX_TUI_REPORT_DIR"] = str(log_dir)

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(REPO_ROOT),
        env=env,
    )
    state._proc = proc

    in_test_phase = False
    last_started_full: str | None = None

    def _handle_result(test_full: str, result: str, inner_pct: int) -> None:
        short = shorten_test_name(test_full)
        state.last_test = short
        state.last_result = result
        base = (state.suite_index - 1) * 100 // state.total_suites
        state.overall_pct = base + inner_pct // state.total_suites
        if result in ("FAILED", "ERROR"):
            state.failed_tests.append(f"{name}: {short}")

    def _parse_line(line: str):
        nonlocal in_test_phase, last_started_full
        m_sec = RE_SECTION.match(line)
        if m_sec:
            in_test_phase = "test session starts" in m_sec.group(1).lower()
            last_started_full = None
            return
        if not in_test_phase:
            return

        m = RE_RESULT.match(line)
        if m:
            _handle_result(m.group(1), m.group(2), int(m.group(3)))
            last_started_full = None
            return

        m_start = RE_TEST_START.match(line)
        if m_start:
            last_started_full = m_start.group(1)
            state.current_test = shorten_test_name(last_started_full)
            return

        m_only = RE_RESULT_ONLY.match(line)
        if m_only and last_started_full is not None:
            _handle_result(last_started_full, m_only.group(1), int(m_only.group(2)))
            last_started_full = None
            return

    with open(log_path, "w", buffering=1) as log_f:
        def _write_log(line: str) -> None:
            log_f.write(line + "\n")

        while True:
            if state.cancelled:
                proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=5)
                except asyncio.TimeoutError:
                    proc.kill()
                break
            line_bytes = await proc.stdout.readline()
            if not line_bytes:
                break
            line = line_bytes.decode("utf-8", errors="replace").rstrip("\n")
            _parse_line(line)
            if line.strip():
                state.log(line)
                _write_log(line)

        if not state.cancelled:
            remaining = await proc.stdout.read()
            if remaining:
                for line in remaining.decode("utf-8", errors="replace").splitlines():
                    _parse_line(line)
                    if line.strip():
                        state.log(line)
                        _write_log(line)

    await proc.wait()
    if hasattr(proc, "_transport") and proc._transport:
        proc._transport.close()
    state._proc = None
    state.current_test = ""
    rc = proc.returncode

    suite_result.duration = time.time() - t0
    suite_result.status = "pass" if rc == 0 else "fail"

    counts: dict[str, int] = {}
    if junit_xml.exists():
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(junit_xml)
            ts = tree.getroot()
            if ts.tag == "testsuites":
                total = sum(int(s.get("tests", "0")) for s in ts.findall("testsuite"))
                fails = sum(int(s.get("failures", "0")) for s in ts.findall("testsuite"))
                errs = sum(int(s.get("errors", "0")) for s in ts.findall("testsuite"))
                skips = sum(int(s.get("skipped", "0")) for s in ts.findall("testsuite"))
            else:
                total = int(ts.get("tests", "0"))
                fails = int(ts.get("failures", "0"))
                errs = int(ts.get("errors", "0"))
                skips = int(ts.get("skipped", "0"))
            passed = total - fails - errs - skips
            counts = {"total": total, "passed": passed, "failed": fails,
                      "errors": errs, "skipped": skips}
            suite_result.summary = f"{passed}p {fails}f {errs}e {skips}s"
        except Exception:
            suite_result.summary = f"rc={rc}"
    else:
        suite_result.summary = "passed" if rc == 0 else f"failed(rc={rc})"

    # Per-suite markdown summary (live, written when this suite finishes)
    suite_md = log_dir / f"{name}.md"
    suite_md.write_text(_render_suite_md(name, suite_result, counts, log_path))

    state.overall_pct = state.suite_index * 100 // state.total_suites
    return rc


def _render_suite_md(
    name: str,
    suite_result: SuiteResult,
    counts: dict[str, int],
    log_path: Path,
) -> str:
    lines = [f"# Suite: {name}", ""]
    lines.append(f"- Status: **{suite_result.status}**")
    lines.append(f"- Duration: {format_duration(suite_result.duration)}")
    if counts:
        lines.append(
            f"- Counts: total={counts['total']} passed={counts['passed']} "
            f"failed={counts['failed']} errors={counts['errors']} "
            f"skipped={counts['skipped']}"
        )
    else:
        lines.append(f"- Summary: {suite_result.summary}")
    lines.append(f"- Log: [`{log_path.name}`]({log_path.name})")
    lines.append("")
    return "\n".join(lines) + "\n"


def _write_index(state: State, log_dir: Path) -> None:
    """Refresh ``test-reports/.../index.md`` with current suite states."""
    lines = ["# Test report", ""]
    lines.append(f"- Run dir: `{log_dir.name}`")
    lines.append(f"- Started: {datetime.fromtimestamp(state.start_time).isoformat() if state.start_time else 'pending'}")
    lines.append(f"- Failures (live): [`failures.md`](failures.md)")
    lines.append("")
    lines.append("| Suite | Status | Duration | Counts | Log |")
    lines.append("|-------|--------|----------|--------|-----|")
    for s in state.suites:
        log_link = f"[`{Path(s.log_path).name}`]({Path(s.log_path).name})" if s.log_path else "—"
        lines.append(
            f"| [{s.name}]({s.name}.md) | {s.status} | "
            f"{format_duration(s.duration) if s.duration else '—'} | "
            f"{s.summary or '—'} | {log_link} |"
        )
    lines.append("")
    if state.failed_tests:
        lines.append(f"## Failed tests ({len(state.failed_tests)})")
        lines.append("")
        for ft in state.failed_tests:
            lines.append(f"- {ft}")
        lines.append("")
    (log_dir / "index.md").write_text("\n".join(lines) + "\n")


async def run_all(state: State, suites: list[tuple[str, list[str]]], log_dir: Path):
    state.total_suites = len(suites)
    state.start_time = time.time()
    state.last_output_time = time.time()
    state.suites = [SuiteResult(name=name) for name, _ in suites]

    _write_index(state, log_dir)
    n_fail = 0
    for name, cmd in suites:
        if state.cancelled:
            break
        rc = await run_suite(name, cmd, state, log_dir)
        if rc != 0:
            n_fail += 1
        _write_index(state, log_dir)

    total_dur = time.time() - state.start_time
    state.log("")
    state.log("═" * 60)
    label = "CANCELLED" if state.cancelled else "COMPLETE"
    state.log(f"  {label} — {format_duration(total_dur)}")
    state.log("═" * 60)
    state.log("")
    for s in state.suites:
        icon = "✓" if s.status == "pass" else "✗" if s.status == "fail" else "○"
        state.log(f"  {icon} {s.name:<15s} {format_duration(s.duration):>10s}  ({s.summary})")
    state.log("")
    state.log(f"Logs: {log_dir}/")
    if (log_dir / "failures.md").exists() and len(state.failed_tests) > 0:
        state.log(f"Failures: {log_dir / 'failures.md'}")

    state.done = True
    state.exit_code = 1 if n_fail > 0 else 0
    state.overall_pct = 100
    state.current_suite = "DONE"
    if n_fail == 0:
        state.last_test = "all suites passed"
        state.last_result = "PASS"
    else:
        state.last_test = f"{n_fail} suite(s) failed"
        state.last_result = "FAIL"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(stdscr, args, log_dir: Path):
    curses.curs_set(0)
    curses.use_default_colors()
    curses.start_color()
    curses.init_pair(1, curses.COLOR_GREEN, -1)
    curses.init_pair(2, curses.COLOR_RED, -1)
    curses.init_pair(3, curses.COLOR_YELLOW, -1)
    curses.init_pair(4, curses.COLOR_CYAN, -1)
    curses.init_pair(5, 8, -1)
    stdscr.timeout(200)
    stdscr.keypad(True)

    suites = build_suites(args, log_dir)
    if not suites:
        return 0

    state = State(report_dir=log_dir)
    loop = asyncio.new_event_loop()
    task = loop.create_task(run_all(state, suites, log_dir))

    def cancel_run():
        state.cancelled = True
        state.log(">>> Cancelling... (waiting for subprocess to exit)")
        if state._proc and state._proc.returncode is None:
            try:
                state._proc.terminate()
            except ProcessLookupError:
                pass

    exit_code = 0
    try:
        while not state.done:
            loop.run_until_complete(asyncio.sleep(0.05))
            try:
                key = stdscr.getch()
                if key == ord("q") or key == 3:
                    cancel_run()
                elif key == curses.KEY_RESIZE:
                    stdscr.clear()
                elif key == curses.KEY_UP:
                    state.scroll_offset = min(
                        state.scroll_offset + 3,
                        max(0, len(state.output_lines) - 5),
                    )
                elif key == curses.KEY_DOWN:
                    state.scroll_offset = max(0, state.scroll_offset - 3)
                elif key == curses.KEY_END or key == ord("G"):
                    state.scroll_offset = 0
                elif key == curses.KEY_HOME or key == ord("g"):
                    state.scroll_offset = max(0, len(state.output_lines) - 5)
                elif key == curses.KEY_PPAGE:
                    height, _ = stdscr.getmaxyx()
                    state.scroll_offset = min(
                        state.scroll_offset + height - 4,
                        max(0, len(state.output_lines) - 5),
                    )
                elif key == curses.KEY_NPAGE:
                    height, _ = stdscr.getmaxyx()
                    state.scroll_offset = max(0, state.scroll_offset - (height - 4))
            except curses.error:
                pass
            draw(stdscr, state)
            curses.doupdate()

        if not task.done():
            loop.run_until_complete(task)
        exit_code = state.exit_code

        draw(stdscr, state)
        curses.doupdate()
        stdscr.timeout(-1)
        stdscr.getch()

    except KeyboardInterrupt:
        cancel_run()
        try:
            loop.run_until_complete(asyncio.wait_for(task, timeout=10))
        except (asyncio.TimeoutError, asyncio.CancelledError, KeyboardInterrupt):
            pass
        exit_code = 1
    finally:
        loop.close()

    return exit_code


if __name__ == "__main__":
    default_model_dir = REPO_ROOT / "models"

    parser = argparse.ArgumentParser(description="omlx TUI test runner")
    parser.add_argument("--no-slow", action="store_true",
                        help="Skip the slow phase")
    parser.add_argument("--no-integration", action="store_true",
                        help="Skip the integration phase")
    parser.add_argument("--only", choices=["unit", "slow", "integration"],
                        default=None,
                        help="Run only one phase")
    parser.add_argument("--model-dir", type=Path, default=default_model_dir,
                        help=f"OMLX_MODEL_DIR for slow/integration tests "
                             f"(default: {default_model_dir}). "
                             f"Use --model-dir '' to leave unset.")
    args = parser.parse_args()

    os.chdir(REPO_ROOT)

    # Apply --model-dir → OMLX_MODEL_DIR. Pass an empty string to leave it
    # unset; missing dirs are tolerated (slow/integration tests will skip).
    if args.model_dir and str(args.model_dir):
        resolved = args.model_dir.expanduser().resolve()
        os.environ["OMLX_MODEL_DIR"] = str(resolved)
        if not resolved.is_dir():
            print(f"warning: --model-dir {resolved} does not exist; "
                  f"slow/integration tests that need models will skip")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = REPO_ROOT / "test-reports" / f"logs_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"Logs: {log_dir}")
    print(f"Failures (live): {log_dir / 'failures.md'}")
    if "OMLX_MODEL_DIR" in os.environ:
        print(f"OMLX_MODEL_DIR={os.environ['OMLX_MODEL_DIR']}")
    print()

    rc = curses.wrapper(lambda stdscr: main(stdscr, args, log_dir))
    sys.exit(rc)
