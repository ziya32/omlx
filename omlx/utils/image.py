# SPDX-License-Identifier: Apache-2.0
"""
Image processing utilities for VLM (Vision-Language Model) support.

This module provides functions for loading images from URLs/base64,
extracting images from OpenAI-format messages, computing image
hashes for prefix cache deduplication, and estimating image token
counts for context window validation.
"""

import base64
import hashlib
import io
import logging
import math
import struct
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageOps

# Register HEIC/HEIF support so PIL can decode Apple image formats.
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

logger = logging.getLogger(__name__)


def load_image(url_or_base64: str) -> Image.Image:
    """
    Load an image from a URL or base64 data URI.

    Supports:
    - HTTP/HTTPS URLs: Downloads the image
    - Data URIs: "data:image/jpeg;base64,..." format

    Args:
        url_or_base64: Image URL or base64 data URI string

    Returns:
        PIL Image object

    Raises:
        ValueError: If the URL format is unsupported
        IOError: If the image cannot be loaded
    """
    if url_or_base64.startswith("data:"):
        # base64 data URI: "data:image/jpeg;base64,<data>"
        try:
            _, data_part = url_or_base64.split(",", 1)
        except ValueError:
            raise ValueError(f"Invalid data URI format: {url_or_base64[:50]}...")
        img_bytes = base64.b64decode(data_part)
        img = Image.open(io.BytesIO(img_bytes))
    elif url_or_base64.startswith(("http://", "https://")):
        import urllib.request

        with urllib.request.urlopen(url_or_base64, timeout=30) as response:
            img_bytes = response.read()
        img = Image.open(io.BytesIO(img_bytes))
    else:
        # Try as local file path
        img = Image.open(url_or_base64)

    # Apply EXIF orientation (phone photos etc.) before processing.
    # Matches mlx-vlm's load_image which calls ImageOps.exif_transpose().
    img = ImageOps.exif_transpose(img)
    # Ensure RGB format (RGBA/P/L etc. cause broadcast errors in vision processors)
    return img.convert("RGB")


def extract_images_from_messages(
    messages: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Image.Image]]:
    """
    Extract images from OpenAI-format messages.

    Processes messages containing content arrays with image_url parts,
    loads the images, and returns cleaned text-only messages alongside
    the loaded images.

    Args:
        messages: List of OpenAI-format chat messages. Each message may have
            content as a string or a list of content parts (text/image_url).

    Returns:
        Tuple of (text_messages, images):
        - text_messages: Messages with image parts removed, text parts joined
        - images: List of loaded PIL Image objects in order of appearance
    """
    text_messages = []
    images = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if not isinstance(content, list):
            # Simple string content — pass through
            text_messages.append({"role": role, "content": content or ""})
            # Preserve extra fields (tool_calls, tool_call_id, etc.)
            for key in msg:
                if key not in ("role", "content"):
                    text_messages[-1][key] = msg[key]
            continue

        # Content array with text and/or image_url parts
        text_parts = []
        for part in content:
            if isinstance(part, dict):
                part_type = part.get("type")
            else:
                # Pydantic model (ContentPart)
                part_type = getattr(part, "type", None)

            if part_type == "text":
                text = part.get("text") if isinstance(part, dict) else getattr(part, "text", None)
                if text:
                    text_parts.append(text)

            elif part_type in ("image_url", "input_image"):
                # OpenAI chat format: {"type":"image_url","image_url":{"url":"..."}}
                # Responses-style format: {"type":"input_image","image_url":"..."}
                image_url_obj = (
                    part.get("image_url") if isinstance(part, dict)
                    else getattr(part, "image_url", None)
                )
                if image_url_obj is None and isinstance(part, dict):
                    image_url_obj = part.get("input_image")

                url = None
                if isinstance(image_url_obj, str):
                    url = image_url_obj
                elif isinstance(image_url_obj, dict):
                    url = image_url_obj.get("url")
                elif image_url_obj is not None:
                    url = getattr(image_url_obj, "url", None)

                if url:
                    try:
                        img = load_image(url)
                        images.append(img)
                    except Exception as e:
                        logger.warning(f"Failed to load image: {e}")

        new_msg = {"role": role, "content": "\n".join(text_parts) if text_parts else ""}
        # Preserve extra fields
        for key in msg:
            if key not in ("role", "content"):
                new_msg[key] = msg[key]
        text_messages.append(new_msg)

    return text_messages, images


# ---------------------------------------------------------------------------
# Lightweight image token estimation (no PIL required)
# ---------------------------------------------------------------------------

def smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 56 * 56,
    max_pixels: int = 14 * 14 * 4 * 1280,
) -> Tuple[int, int]:
    """Rescale dimensions to be divisible by factor and within pixel bounds.

    Pure arithmetic — no image loading. Reimplements the HuggingFace
    transformers ``smart_resize`` used by Qwen-VL image processors.
    """
    if min(height, width) < 1:
        return factor, factor
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, "
            f"got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def estimate_image_tokens(
    width: int,
    height: int,
    patch_size: int = 14,
    merge_size: int = 2,
    min_pixels: int = 56 * 56,
    max_pixels: int = 14 * 14 * 4 * 1280,
) -> int:
    """Estimate VLM image token count from dimensions and vision config.

    Uses ``smart_resize`` to compute the effective resolution, then divides
    by ``(patch_size * merge_size)^2`` to get the token count.  Works for
    Qwen-VL family models; provides a reasonable approximation for other
    architectures with similar patch+merge pipelines.
    """
    factor = patch_size * merge_size
    rh, rw = smart_resize(height, width, factor=factor,
                          min_pixels=min_pixels, max_pixels=max_pixels)
    return (rh * rw) // (factor * factor)


def _get_image_dimensions_from_bytes(data: bytes) -> Optional[Tuple[int, int]]:
    """Extract (width, height) from image header bytes without full PIL decode.

    Reads only the format header (first ~30 bytes for most formats, scans
    further for JPEG SOF markers).  Returns ``None`` for unknown formats.
    """
    if len(data) < 24:
        return None

    # PNG: IHDR at bytes 16-23
    if data[:8] == b'\x89PNG\r\n\x1a\n':
        w = struct.unpack('>I', data[16:20])[0]
        h = struct.unpack('>I', data[20:24])[0]
        return w, h

    # GIF
    if data[:6] in (b'GIF87a', b'GIF89a'):
        w = struct.unpack('<H', data[6:8])[0]
        h = struct.unpack('<H', data[8:10])[0]
        return w, h

    # BMP
    if data[:2] == b'BM' and len(data) >= 26:
        w = struct.unpack('<I', data[18:22])[0]
        h = abs(struct.unpack('<i', data[22:26])[0])
        return w, h

    # WebP
    if data[:4] == b'RIFF' and data[8:12] == b'WEBP':
        if data[12:16] == b'VP8 ' and len(data) >= 30:
            w = struct.unpack('<H', data[26:28])[0] & 0x3FFF
            h = struct.unpack('<H', data[28:30])[0] & 0x3FFF
            return w, h
        if data[12:16] == b'VP8L' and len(data) >= 25:
            bits = struct.unpack('<I', data[21:25])[0]
            w = (bits & 0x3FFF) + 1
            h = ((bits >> 14) & 0x3FFF) + 1
            return w, h
        if data[12:16] == b'VP8X' and len(data) >= 30:
            w = struct.unpack('<I', data[24:27] + b'\x00')[0] + 1
            h = struct.unpack('<I', data[27:30] + b'\x00')[0] + 1
            return w, h
        return None

    # JPEG: scan for SOF markers
    if data[:2] == b'\xff\xd8':
        i = 2
        while i < len(data) - 9:
            if data[i] != 0xFF:
                break
            marker = data[i + 1]
            if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                          0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                h = struct.unpack('>H', data[i + 5:i + 7])[0]
                w = struct.unpack('>H', data[i + 7:i + 9])[0]
                return w, h
            seg_len = struct.unpack('>H', data[i + 2:i + 4])[0]
            i += 2 + seg_len
        return None

    # HEIC/HEIF (ISOBMFF ftyp box) — parsing the ispe box is complex,
    # fall back to PIL which only reads the header (lazy open).
    if len(data) >= 12 and data[4:8] == b'ftyp':
        try:
            img = Image.open(io.BytesIO(data))
            return img.size  # (width, height)
        except Exception:
            return None

    return None


def _get_image_url_from_part(part: Any) -> Optional[str]:
    """Extract the URL string from an image_url or input_image content part."""
    if isinstance(part, dict):
        image_url_obj = part.get("image_url") or part.get("input_image")
    else:
        image_url_obj = getattr(part, "image_url", None)
        if image_url_obj is None:
            image_url_obj = getattr(part, "input_image", None)

    if isinstance(image_url_obj, str):
        return image_url_obj
    if isinstance(image_url_obj, dict):
        return image_url_obj.get("url")
    if image_url_obj is not None:
        return getattr(image_url_obj, "url", None)
    return None


def strip_images_and_estimate_tokens(
    messages: List[Dict[str, Any]],
    patch_size: int = 14,
    merge_size: int = 2,
    min_pixels: int = 56 * 56,
    max_pixels: int = 14 * 14 * 4 * 1280,
) -> Tuple[List[Dict[str, Any]], int]:
    """Strip image parts from messages and estimate total image tokens.

    Unlike ``extract_images_from_messages``, this function never loads
    images via PIL.  It extracts dimensions from base64 data URI headers
    and uses ``smart_resize`` to estimate token counts.

    Args:
        messages: OpenAI-format chat messages.
        patch_size: Vision encoder patch size (from model config).
        merge_size: Spatial merge factor (from model config).
        min_pixels: Minimum total pixels after resize.
        max_pixels: Maximum total pixels after resize.

    Returns:
        Tuple of (text_only_messages, estimated_image_tokens).
    """
    text_messages = []
    total_image_tokens = 0

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if not isinstance(content, list):
            text_messages.append({"role": role, "content": content or ""})
            for key in msg:
                if key not in ("role", "content"):
                    text_messages[-1][key] = msg[key]
            continue

        text_parts = []
        for part in content:
            if isinstance(part, dict):
                part_type = part.get("type")
            else:
                part_type = getattr(part, "type", None)

            if part_type == "text":
                text = (
                    part.get("text") if isinstance(part, dict)
                    else getattr(part, "text", None)
                )
                if text:
                    text_parts.append(text)

            elif part_type in ("image_url", "input_image"):
                url = _get_image_url_from_part(part)
                if not url:
                    continue

                dims = None
                if url.startswith("data:"):
                    try:
                        _, data_part = url.split(",", 1)
                        img_bytes = base64.b64decode(data_part)
                        dims = _get_image_dimensions_from_bytes(img_bytes)
                    except Exception:
                        pass

                if dims:
                    total_image_tokens += estimate_image_tokens(
                        dims[0], dims[1],
                        patch_size=patch_size,
                        merge_size=merge_size,
                        min_pixels=min_pixels,
                        max_pixels=max_pixels,
                    )
                else:
                    # Can't determine dimensions (HTTP URL or unknown format).
                    # Use a conservative estimate based on a typical 1024x768 image.
                    total_image_tokens += estimate_image_tokens(
                        1024, 768,
                        patch_size=patch_size,
                        merge_size=merge_size,
                        min_pixels=min_pixels,
                        max_pixels=max_pixels,
                    )

        new_msg = {"role": role, "content": "\n".join(text_parts) if text_parts else ""}
        for key in msg:
            if key not in ("role", "content"):
                new_msg[key] = msg[key]
        text_messages.append(new_msg)

    return text_messages, total_image_tokens


def compute_image_hash(images: List[Image.Image]) -> Optional[str]:
    """
    Compute a SHA256 hash from a list of images for prefix cache deduplication.

    Uses image size and raw pixel data to produce a deterministic hash.
    Returns None if images list is empty.

    Args:
        images: List of PIL Image objects

    Returns:
        Hex-encoded SHA256 hash string, or None if no images
    """
    if not images:
        return None

    hasher = hashlib.sha256()
    for img in images:
        # Include image dimensions
        hasher.update(f"{img.size[0]}x{img.size[1]}".encode())
        # Include raw pixel data (convert to RGB for consistency)
        rgb_img = img.convert("RGB")
        hasher.update(rgb_img.tobytes())

    return hasher.hexdigest()
