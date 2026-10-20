"""Queries and candidates as prompt parts: text, images, or both."""
from typing import Any, List

from .images import DEFAULT_MAX_PIXELS, to_data_url
from .prompts import Part


def to_parts(item: Any, max_pixels: int = DEFAULT_MAX_PIXELS) -> List[Part]:
    """A query or a candidate as prompt parts.

    - a ``str`` is text (runs of whitespace become one space);
    - a ``dict`` with ``"image"`` (one image or a list) and/or ``"text"`` gives its images first, then its text;
    - a ``list`` is a sequence of such items, concatenated;
    - anything else is an image (a PIL image or raw bytes).
    Images become JPEG data URLs of at most `max_pixels` pixels; http(s) and data URLs pass through.
    """
    if isinstance(item, str):
        return [" ".join(item.split())]
    if isinstance(item, list):
        return [p for x in item for p in to_parts(x, max_pixels)]
    if isinstance(item, dict):
        parts: List[Part] = []
        imgs = item.get("image")
        if imgs is not None:
            parts += [{"image": to_data_url(i, max_pixels)} for i in (imgs if isinstance(imgs, list) else [imgs])]
        if item.get("text"):
            parts.append(" ".join(str(item["text"]).split()))
        return parts
    return [{"image": to_data_url(item, max_pixels)}]
