"""Images as data URLs, downscaled to a pixel budget."""
import base64
import io
from typing import Union

DEFAULT_MAX_PIXELS = 1800 * 32 * 32  # the image budget of Qwen3-VL-Reranker; every system in the report gets it

ImageLike = Union[str, bytes, "PIL.Image.Image"]  # noqa: F821


def to_data_url(image: ImageLike, max_pixels: int = DEFAULT_MAX_PIXELS) -> str:
    """A JPEG data URL of `image` (a PIL image, a file path or raw bytes) with at most `max_pixels` pixels.

    An ``http(s)://`` or ``data:`` URL is returned unchanged: the server reads and resizes it.
    """
    if isinstance(image, str) and image.startswith(("http://", "https://", "data:")):
        return image
    from PIL import Image

    if isinstance(image, (bytes, str)):
        image = Image.open(io.BytesIO(image) if isinstance(image, bytes) else image)
    im = image.convert("RGB")
    w, h = im.size
    if w * h > max_pixels:
        s = (max_pixels / (w * h)) ** 0.5
        im = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BICUBIC)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
