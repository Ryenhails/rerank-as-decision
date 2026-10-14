import base64
import io

import pytest
from PIL import Image

from vlm_as_jev import Decider, FakeBackend, to_data_url, to_parts


def by_length(text):
    return float(len(text))


def test_choice_over_text_options_about_an_image():
    d = Decider(FakeBackend(by_length)).choice(Image.new("RGB", (8, 8)), "Which is it?", ["cat", "dog", "giraffe"])
    assert d.answer == "giraffe" and d.best == 2 and d.ranking == [2, 0, 1]
    assert sum(d.probs) == pytest.approx(1)


def test_options_can_be_images_and_text():
    img = Image.new("RGB", (8, 8))
    d = Decider(FakeBackend(by_length)).choice(img, "Which caption fits?", [{"image": img, "text": "a"}, "longer"])
    assert d.answer == "longer" and d.options[0]["image"] is img


def test_ties_go_to_the_first_option():
    d = Decider(FakeBackend(lambda text: 0.0)).choice("s", "q", ["x", "y", "z"])
    assert d.best == 0 and d.ranking == [0, 1, 2]


def test_option_count_limits():
    with pytest.raises(ValueError):
        Decider(FakeBackend(by_length)).choice("s", "q", ["only"])
    with pytest.raises(ValueError):
        Decider(FakeBackend(by_length)).choice("s", "q", ["o"] * 17)


def test_yes_no():
    assert Decider(FakeBackend(lambda text: 100.0)).yes_no("a state", "is it?") == pytest.approx(1)


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
