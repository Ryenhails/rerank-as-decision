"""A deterministic backend for tests and examples: a score per option text, no model."""
import math
from typing import Callable, List, Sequence, Tuple

from ..prompts import ChoicePrompt, Part


def _options_text(prompt: ChoicePrompt) -> List[str]:
    """The text of each labelled option in the prompt's state."""
    out, cur = [], None
    heads = {f"{prompt.word} {x}" for x in prompt.letters}
    for p in prompt.state:
        if isinstance(p, str) and p.strip().rstrip(":").strip() in heads:
            cur = []
            out.append(cur)
        elif cur is not None and isinstance(p, str):
            cur.append(p)
    return [" ".join(x) for x in out]


class FakeBackend:
    """`score(option_text) -> float` gives each option a logit; a choice reads their softmax."""

    name = "fake"

    def __init__(self, score: Callable[[str], float]):
        self.score = score
        self.calls = 0

    def choice(self, prompt: ChoicePrompt) -> Tuple[List[float], int]:
        self.calls += 1
        z = [self.score(t) for t in _options_text(prompt)]
        m = max(z)
        e = [math.exp(x - m) for x in z]
        return [x / sum(e) for x in e], len(prompt.state)

    def yes_no(self, state: Sequence[Part], question: str) -> Tuple[float, int]:
        self.calls += 1
        last = next((p for p in reversed(state) if isinstance(p, str)), "")
        return 1 / (1 + math.exp(-self.score(last))), len(state)
