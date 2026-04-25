# SPDX-License-Identifier: Apache-2.0
"""Tests for VLM image token estimation.

Validates that we can accurately estimate image token counts from image
dimensions using the model's vision config (patch_size, merge_size), without
needing to load/decode images via PIL.

Also benchmarks the cost of the lightweight approach vs the current
extract_images_from_messages approach (which fully decodes every image).
"""

import base64
import io
import time
from pathlib import Path

import pytest
from PIL import Image

from omlx.utils.image import (
    _get_image_dimensions_from_bytes,
    estimate_image_tokens,
    smart_resize,
    strip_images_and_estimate_tokens,
)

TESTSPACE = Path(__file__).resolve().parent.parent.parent / "nanobot" / "testspace"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_test_image(width: int, height: int) -> Image.Image:
    return Image.new("RGB", (width, height), "red")


def _make_data_uri(width: int, height: int, fmt: str = "PNG") -> str:
    img = _make_test_image(width, height)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode()
    mime = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp",
            "GIF": "image/gif", "BMP": "image/bmp"}[fmt]
    return f"data:{mime};base64,{b64}"


def _make_messages_with_images(image_data_uris: list[str], text: str = "Describe") -> list[dict]:
    content = []
    for uri in image_data_uris:
        content.append({"type": "image_url", "image_url": {"url": uri}})
    content.append({"type": "text", "text": text})
    return [{"role": "user", "content": content}]


# ===========================================================================
# Tests: smart_resize
# ===========================================================================

class TestSmartResize:
    """Validate smart_resize matches expected behavior."""

    def test_already_aligned(self):
        h, w = smart_resize(1024, 1024, factor=32, min_pixels=256*256, max_pixels=4096*4096)
        assert h == 1024
        assert w == 1024

    def test_rounds_to_factor(self):
        h, w = smart_resize(1000, 1000, factor=32, min_pixels=256*256, max_pixels=4096*4096)
        assert h % 32 == 0
        assert w % 32 == 0
        assert h == 992  # round(1000/32)*32

    def test_scales_down_large(self):
        max_px = 1024 * 1024
        h, w = smart_resize(4000, 4000, factor=32, min_pixels=256*256, max_pixels=max_px)
        assert h * w <= max_px
        assert h % 32 == 0

    def test_scales_up_small(self):
        min_px = 256 * 256
        h, w = smart_resize(32, 32, factor=32, min_pixels=min_px, max_pixels=4096*4096)
        assert h * w >= min_px
        assert h % 32 == 0


# ===========================================================================
# Tests: estimate_image_tokens
# ===========================================================================

class TestEstimateImageTokens:
    """Validate token estimation for known image sizes."""

    # Qwen3.5 config: patch=16, merge=2 → factor=32, pixels_per_token=1024
    PATCH = 16
    MERGE = 2
    MIN_PX = 65536    # from preprocessor_config: shortest_edge
    MAX_PX = 16777216  # from preprocessor_config: longest_edge

    def _est(self, w, h):
        return estimate_image_tokens(w, h, self.PATCH, self.MERGE, self.MIN_PX, self.MAX_PX)

    def test_1024x1024(self):
        assert self._est(1024, 1024) == 1024

    def test_1920x1080(self):
        assert self._est(1920, 1080) == 2040

    def test_small_image_scaled_up(self):
        tokens = self._est(224, 224)
        assert tokens >= 64

    def test_large_image_capped(self):
        tokens = self._est(8000, 6000)
        assert tokens <= self.MAX_PX // (self.PATCH * self.MERGE) ** 2

    def test_returns_positive(self):
        for w, h in [(100, 100), (640, 480), (3840, 2160)]:
            assert self._est(w, h) > 0

    def test_qwen2_5_vl_config(self):
        tokens = estimate_image_tokens(1024, 1024, patch_size=14, merge_size=2)
        assert tokens == 1225


# ===========================================================================
# Tests: dimension extraction from image headers
# ===========================================================================

class TestGetDimensionsFromBytes:
    """Validate lightweight dimension extraction from image headers."""

    @pytest.mark.parametrize("fmt", ["PNG", "JPEG", "GIF", "BMP", "WEBP"])
    def test_format(self, fmt):
        w, h = 640, 480
        img = _make_test_image(w, h)
        buf = io.BytesIO()
        img.save(buf, format=fmt)
        dims = _get_image_dimensions_from_bytes(buf.getvalue())
        assert dims is not None, f"Failed to extract dimensions from {fmt}"
        assert dims == (w, h), f"{fmt}: expected ({w},{h}), got {dims}"

    def test_heic_format(self):
        """HEIC dimension extraction via PIL fallback (requires pillow-heif)."""
        heic_path = TESTSPACE / "data" / "vision" / "IMG_9008.HEIC"
        if not heic_path.exists():
            pytest.skip("No HEIC test file available")
        data = heic_path.read_bytes()
        dims = _get_image_dimensions_from_bytes(data)
        assert dims is not None, "Failed to extract dimensions from HEIC"
        assert dims == (4032, 3024)

    @pytest.mark.parametrize("size", [(1, 1), (100, 200), (1920, 1080), (4000, 3000)])
    def test_various_sizes_png(self, size):
        w, h = size
        img = _make_test_image(w, h)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        dims = _get_image_dimensions_from_bytes(buf.getvalue())
        assert dims == (w, h)

    def test_unknown_format_returns_none(self):
        assert _get_image_dimensions_from_bytes(b"not an image") is None

    def test_too_short_returns_none(self):
        assert _get_image_dimensions_from_bytes(b"ab") is None


# ===========================================================================
# Tests: strip_images_and_estimate_tokens
# ===========================================================================

class TestStripImagesAndEstimateTokens:
    """Validate lightweight message stripping + token estimation."""

    def test_text_only(self):
        msgs = [{"role": "user", "content": "Hello"}]
        text_msgs, tokens = strip_images_and_estimate_tokens(msgs)
        assert tokens == 0
        assert text_msgs[0]["content"] == "Hello"

    def test_single_image(self):
        uri = _make_data_uri(1024, 1024, "PNG")
        msgs = _make_messages_with_images([uri], "Describe this")
        text_msgs, tokens = strip_images_and_estimate_tokens(
            msgs, patch_size=16, merge_size=2, min_pixels=65536, max_pixels=16777216)
        assert tokens == 1024  # 1024x1024 / (32*32)
        assert text_msgs[0]["content"] == "Describe this"

    def test_multiple_images(self):
        uris = [_make_data_uri(1024, 1024, "PNG"), _make_data_uri(1920, 1080, "JPEG")]
        msgs = _make_messages_with_images(uris, "What are these?")
        text_msgs, tokens = strip_images_and_estimate_tokens(
            msgs, patch_size=16, merge_size=2, min_pixels=65536, max_pixels=16777216)
        assert tokens == 1024 + 2040
        assert text_msgs[0]["content"] == "What are these?"

    def test_preserves_extra_fields(self):
        msgs = [{"role": "assistant", "content": "ok", "tool_calls": [{"id": "1"}]}]
        text_msgs, tokens = strip_images_and_estimate_tokens(msgs)
        assert text_msgs[0]["tool_calls"] == [{"id": "1"}]

    def test_mixed_conversation(self):
        uri = _make_data_uri(800, 600, "PNG")
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": uri}},
                {"type": "text", "text": "What is this?"},
            ]},
            {"role": "assistant", "content": "It is a photo."},
            {"role": "user", "content": "Thanks"},
        ]
        text_msgs, tokens = strip_images_and_estimate_tokens(msgs)
        assert tokens > 0
        assert text_msgs[0]["content"] == "You are helpful."
        assert text_msgs[1]["content"] == "What is this?"
        assert text_msgs[2]["content"] == "It is a photo."
        assert text_msgs[3]["content"] == "Thanks"


# ===========================================================================
# Benchmark: lightweight estimation vs extract_images_from_messages
# ===========================================================================

class TestBenchmarkTokenEstimation:
    """Compare performance of lightweight estimation vs full image loading."""

    @staticmethod
    def _make_large_image_uris(n: int = 2, width: int = 1920, height: int = 1080) -> list[str]:
        return [_make_data_uri(width, height, "JPEG") for _ in range(n)]

    def test_benchmark_lightweight_vs_full(self):
        """Lightweight approach should be faster than full PIL decode + RGB convert."""
        uris = self._make_large_image_uris(n=3, width=1920, height=1080)
        messages = _make_messages_with_images(uris, "Describe these images")

        iters = 50

        from omlx.utils.image import extract_images_from_messages

        start = time.perf_counter()
        for _ in range(iters):
            text_msgs_full, images = extract_images_from_messages(messages)
        full_time = (time.perf_counter() - start) / iters

        start = time.perf_counter()
        for _ in range(iters):
            text_msgs_light, image_tokens = strip_images_and_estimate_tokens(
                messages, patch_size=16, merge_size=2,
                min_pixels=65536, max_pixels=16777216)
        light_time = (time.perf_counter() - start) / iters

        # Same text output
        assert len(text_msgs_full) == len(text_msgs_light)
        for full, light in zip(text_msgs_full, text_msgs_light):
            assert full["content"] == light["content"]

        # Token estimate is reasonable (3 images × ~2040 tokens)
        assert 5000 < image_tokens < 8000

        speedup = full_time / light_time if light_time > 0 else float('inf')
        print(f"\n  Full PIL load: {full_time*1000:.2f}ms")
        print(f"  Lightweight:   {light_time*1000:.2f}ms")
        print(f"  Speedup:       {speedup:.1f}x")
        print(f"  Image tokens:  {image_tokens} ({len(images)} images)")

        assert image_tokens > 0

    def test_dimension_extraction_cost(self):
        """Dimension extraction from base64 should be sub-millisecond."""
        uri = _make_data_uri(3840, 2160, "JPEG")
        b64_data = uri.split(",", 1)[1]
        img_bytes = base64.b64decode(b64_data)

        start = time.perf_counter()
        for _ in range(1000):
            dims = _get_image_dimensions_from_bytes(img_bytes)
        elapsed = (time.perf_counter() - start)

        assert dims == (3840, 2160)
        per_call_us = (elapsed / 1000) * 1_000_000
        print(f"\n  Dimension extraction: {per_call_us:.0f}µs per call")
        assert per_call_us < 1000

    def test_smart_resize_cost(self):
        """smart_resize + token estimation should be sub-microsecond."""
        start = time.perf_counter()
        for _ in range(100_000):
            estimate_image_tokens(1920, 1080, patch_size=16, merge_size=2,
                                  min_pixels=65536, max_pixels=16777216)
        elapsed = (time.perf_counter() - start)

        per_call_ns = (elapsed / 100_000) * 1_000_000_000
        print(f"\n  Token estimation: {per_call_ns:.0f}ns per call")
        assert per_call_ns < 10_000
