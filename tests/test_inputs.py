import base64
import io

from PIL import Image

from glancerank import to_data_url, to_parts


def test_to_parts_images_first_then_text():
    img = Image.new("RGB", (10, 10))
    parts = to_parts({"text": "a  caption\n", "image": [img, "https://x/y.png"]})
    assert parts[0]["image"].startswith("data:image/jpeg;base64,")
    assert parts[1] == {"image": "https://x/y.png"}
    assert parts[2] == "a caption"
    assert to_parts("  two   words ") == ["two words"]
    assert to_parts(["a", "b"]) == ["a", "b"]


def test_images_are_downscaled_to_the_budget():
    url = to_data_url(Image.new("RGB", (4000, 3000)), max_pixels=1800 * 32 * 32)
    w, h = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).size
    assert w * h <= 1800 * 32 * 32 and abs(w / h - 4 / 3) < 0.01
    small = to_data_url(Image.new("RGB", (100, 50)))
    assert Image.open(io.BytesIO(base64.b64decode(small.split(",", 1)[1]))).size == (100, 50)
