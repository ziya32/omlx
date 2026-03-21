"""Test-only request capture for verifying what omlx receives.

Gated by OMLX_DEBUG_CAPTURE=1 environment variable. When disabled (default),
all functions are no-ops and the /debug/last-request endpoint is not registered.

WARNING: This captures full request payloads including message content.
Never enable in production.
"""

from __future__ import annotations

import os
import threading
from typing import Any

# Kill switch: only active when explicitly enabled
ENABLED = os.environ.get("OMLX_DEBUG_CAPTURE") == "1"

_lock = threading.Lock()
_requests: list[dict[str, Any]] = []
_prompts: list[str] = []
_capture_id: int = 0  # Incremented by reset; groups requests per test


def reset() -> None:
    """Clear all captured data. Call before each test."""
    if not ENABLED:
        return
    global _capture_id
    with _lock:
        _requests.clear()
        _prompts.clear()
        _capture_id += 1


def capture_request(
    model: str,
    messages: list[dict],
    tools_count: int,
    tool_choice: str | None,
) -> None:
    """Append a chat completion request to the capture list (no-op if disabled)."""
    if not ENABLED:
        return
    with _lock:
        _requests.append({
            "model": model,
            "messages": [
                {"role": m.get("role") if isinstance(m, dict) else getattr(m, "role", None),
                 "content": (str(m.get("content") if isinstance(m, dict) else getattr(m, "content", None)) or "")[:50000]}
                for m in messages
            ],
            "message_count": len(messages),
            "tools_count": tools_count,
            "tool_choice": tool_choice,
        })


def capture_prompt(prompt: str) -> None:
    """Append a rendered prompt to the capture list (no-op if disabled)."""
    if not ENABLED:
        return
    with _lock:
        _prompts.append(prompt)


def get_capture() -> dict[str, Any] | None:
    """Return all captured requests and prompts."""
    if not ENABLED:
        return None
    with _lock:
        if not _requests and not _prompts:
            return None
        return {
            "requests": list(_requests),
            "prompts": list(_prompts),
            "request_count": len(_requests),
            "prompt_count": len(_prompts),
        }


def register_debug_endpoint(app) -> None:
    """Register debug endpoints. Only called when ENABLED."""
    if not ENABLED:
        return

    @app.get("/debug/last-request")
    async def debug_last_request():
        """Return all captured requests and prompts.

        Only available when OMLX_DEBUG_CAPTURE=1.
        """
        data = get_capture()
        if data is None:
            return {"error": "No request captured yet"}
        return data

    @app.post("/debug/reset-capture")
    async def debug_reset_capture():
        """Clear captured data. Call before each test."""
        reset()
        return {"ok": True}
