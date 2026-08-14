"""Viewport rendering: composes a scan panel with burned-in corner annotations,
the way a radiology workstation overlays study metadata onto the image itself.

Kept separate from inference.py so the presentation layer stays testable.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

VIEWPORT_SIZE = 448
PAD = 10
INK = (201, 214, 226)
INK_DIM = (107, 127, 146)


def _mono_font(size: int = 13) -> ImageFont.ImageFont:
    """A monospace face. matplotlib bundles DejaVu, so this works anywhere
    the app's own dependencies are installed."""
    try:
        from matplotlib import font_manager

        path = font_manager.findfont("DejaVu Sans Mono", fallback_to_default=False)
        return ImageFont.truetype(path, size)
    except Exception:
        try:
            return ImageFont.load_default(size=size)
        except TypeError:  # very old Pillow
            return ImageFont.load_default()


def to_image(array: np.ndarray) -> Image.Image:
    """Accept float [0,1] or uint8 RGB and return a PIL image."""
    if array.dtype != np.uint8:
        array = (np.clip(array, 0, 1) * 255).astype("uint8")
    return Image.fromarray(array)


def render_viewport(
    array: np.ndarray,
    top_left: str = "",
    top_right: str = "",
    bottom_left: str = "",
    bottom_right: str = "",
    size: int = VIEWPORT_SIZE,
    border: tuple[int, int, int] | None = None,
) -> Image.Image:
    """Upscale the scan and burn annotations into its four corners.

    Multi-line strings are supported; each corner block is aligned outward from
    its own edge, matching how scanner overlays are laid out. `border` draws a
    hairline frame, which keeps the dark scan from bleeding into a light page.
    """
    img = to_image(array).resize((size, size), Image.LANCZOS).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = _mono_font()
    line_h = 15

    def block(text: str, corner: str) -> None:
        if not text:
            return
        lines = text.split("\n")
        for i, line in enumerate(lines):
            box = draw.textbbox((0, 0), line, font=font)
            w, h = box[2] - box[0], box[3] - box[1]

            if corner.startswith("top"):
                y = PAD + i * line_h
            else:
                y = size - PAD - h - (len(lines) - 1 - i) * line_h

            x = PAD if corner.endswith("left") else size - PAD - w

            # A soft shadow keeps the text legible over bright heatmap regions.
            draw.text((x + 1, y + 1), line, font=font, fill=(0, 0, 0))
            draw.text(
                (x, y), line, font=font, fill=INK if i == 0 else INK_DIM
            )

    block(top_left, "top-left")
    block(top_right, "top-right")
    block(bottom_left, "bottom-left")
    block(bottom_right, "bottom-right")

    if border is not None:
        draw.rectangle([0, 0, size - 1, size - 1], outline=border, width=1)
    return img
