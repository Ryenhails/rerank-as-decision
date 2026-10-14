"""Typed decisions from a vision-language model: one forward pass, one probability per option, no decoding."""
from dataclasses import dataclass
from typing import Any, List, Sequence

from .images import DEFAULT_MAX_PIXELS, to_data_url
from .prompts import Part, choice_prompt


def to_parts(item: Any, max_pixels: int = DEFAULT_MAX_PIXELS) -> List[Part]:
    """A state or an option as prompt parts.

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


@dataclass
class Decision:
    options: list            # as given
    probs: List[float]       # one per option, in the given order; they sum to 1
    prompt_tokens: int

    @property
    def best(self) -> int:
        """Index of the most probable option (the first one on a tie)."""
        return max(range(len(self.probs)), key=lambda i: (self.probs[i], -i))

    @property
    def answer(self):
        return self.options[self.best]

    @property
    def ranking(self) -> List[int]:
        """Option indices, most probable first; ties keep the given order."""
        return sorted(range(len(self.probs)), key=lambda i: (-self.probs[i], i))


class Decider:
    """`backend`: a `ChatBackend`, `JevDecideBackend` or any object with `choice` and `yes_no`
    (see `vlm_as_jev.backends.base`)."""

    def __init__(self, backend, max_pixels: int = DEFAULT_MAX_PIXELS):
        self.backend, self.max_pixels = backend, max_pixels

    def choice(self, state: Any, question: str, options: Sequence[Any], word: str = "Option") -> Decision:
        """Which of 2 to 16 `options` answers `question` about `state`. States and options are text, images or both
        (see `to_parts`); the options go into the prompt as ``Option A:``, ``Option B:``, ... (`word` renames them)."""
        d = self.choice_parts(to_parts(state, self.max_pixels), question,
                              [to_parts(o, self.max_pixels) for o in options], word=word)
        d.options = list(options)
        return d

    def yes_no(self, state: Any, question: str) -> float:
        """The probability that the answer to `question` about `state` is yes."""
        return self.yes_no_parts(to_parts(state, self.max_pixels), question)[0]

    # ------------------------------------------------------------------ on prompt parts
    def choice_parts(self, state: Sequence[Part], question: str, options: Sequence[Sequence[Part]],
                     word: str = "Option", label_suffix: str = ": ") -> Decision:
        """`choice` on parts that are already prompt parts (no conversion)."""
        prompt = choice_prompt(state, question, options, word=word, label_suffix=label_suffix)
        probs, tokens = self.backend.choice(prompt)
        return Decision(options=list(options), probs=list(probs), prompt_tokens=tokens)

    def yes_no_parts(self, state: Sequence[Part], question: str):
        """(probability of yes, prompt tokens)."""
        return self.backend.yes_no(state, question)
