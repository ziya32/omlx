# SPDX-License-Identifier: Apache-2.0
"""Pytest plugin that writes failure detail to disk as each test finishes.

Loaded by run_tests_tui.py via ``-p _tui_failure_plugin``.  The TUI runner
sets ``OMLX_TUI_REPORT_DIR`` to the per-run log directory; the plugin writes:

  - ``<report-dir>/failures.md`` — rolling Markdown report, appended on
    every failed/errored test as it happens.  Survives a Ctrl-C kill.
  - ``<report-dir>/failures/<safe-nodeid>.md`` — per-failure traceback in
    Markdown, one file per failed test, also written immediately.

Without this plugin pytest only emits the FAILURES block at end of run, so
killing the suite mid-flight loses every failure detail collected so far.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any


_NODEID_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _report_dir() -> Path:
    raw = os.environ.get("OMLX_TUI_REPORT_DIR")
    if not raw:
        # Fall back to a sibling of cwd so the plugin still works if loaded
        # outside the runner (e.g. the user invokes pytest -p directly).
        raw = "test-reports/current"
    p = Path(raw)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _safe_nodeid(nodeid: str) -> str:
    safe = _NODEID_SAFE_RE.sub("_", nodeid).strip("_")
    return safe[:200] or "unnamed"


def pytest_configure(config: Any) -> None:
    rd = _report_dir()
    md = rd / "failures.md"
    if not md.exists():
        md.write_text("# Test Failures\n\n")


def pytest_collection_modifyitems(items: Any) -> None:
    """Emit execution-ordered test list so the TUI can show the running test.

    Default ``-v`` output only prints the nodeid once the test finishes
    (``tests/foo.py::test_bar PASSED [42%]``), so without this hook the
    runner never knows which test is *currently* executing — only which
    one just finished. The runner consumes these lines, builds a queue,
    and advances ``current_test`` after each completion.
    """
    if not os.environ.get("OMLX_TUI_REPORT_DIR"):
        return
    for item in items:
        print(f">>> QUEUED {item.nodeid}", flush=True)


def pytest_runtest_logreport(report: Any) -> None:  # noqa: D401
    """Hook: called after each setup/call/teardown phase of every test."""
    # Only the call phase carries a real failure for normal tests; setup
    # and teardown failures still surface as "failed" reports — capture
    # them too so fixture errors aren't lost.
    if report.outcome != "failed":
        return
    phase = report.when  # "setup" | "call" | "teardown"
    nodeid = report.nodeid or "(unknown)"

    longrepr = report.longrepr
    if longrepr is None:
        body = "(no traceback captured)"
    else:
        try:
            # ReprExceptionInfo has reprcrash + reprtraceback; str() gives
            # the same text pytest would print at end-of-run.
            body = str(longrepr)
        except Exception:
            body = repr(longrepr)

    # Capture stdout/stderr too — they often contain the smoking gun.
    sections = []
    for sec in getattr(report, "sections", []) or []:
        if not isinstance(sec, tuple) or len(sec) != 2:
            continue
        name, content = sec
        sections.append(f"--- {name} ---\n{content}")
    extra = "\n\n".join(sections)

    rd = _report_dir()
    failures_dir = rd / "failures"
    failures_dir.mkdir(parents=True, exist_ok=True)
    tag = "" if phase == "call" else f" ({phase})"

    detail = failures_dir / f"{_safe_nodeid(nodeid)}.md"
    with detail.open("w") as f:
        f.write(f"# {nodeid}{tag}\n\n")
        f.write(f"- Phase: `{phase}`\n\n")
        f.write("## Traceback\n\n")
        f.write("```\n")
        f.write(body.rstrip())
        f.write("\n```\n")
        if extra:
            f.write("\n## Captured output\n\n")
            f.write("```\n")
            f.write(extra.rstrip())
            f.write("\n```\n")
        f.flush()
        os.fsync(f.fileno())

    md = rd / "failures.md"
    with md.open("a") as f:
        f.write(f"\n## {nodeid}{tag}\n\n")
        f.write(f"- Detail: [`failures/{_safe_nodeid(nodeid)}.md`]"
                f"(failures/{_safe_nodeid(nodeid)}.md)\n\n")
        f.write("```\n")
        f.write(body.rstrip())
        f.write("\n```\n")
        if extra:
            f.write("\n<details><summary>captured output</summary>\n\n")
            f.write("```\n")
            f.write(extra.rstrip())
            f.write("\n```\n\n</details>\n")
        f.flush()
        os.fsync(f.fileno())
