"""What a backend implements: two reads, each one forward pass of the model, no decoding."""
from typing import List, Protocol, Sequence, Tuple

from ..prompts import ChoicePrompt, Part


class Backend(Protocol):
    name: str

    def choice(self, prompt: ChoicePrompt) -> Tuple[List[float], int]:
        """One probability per option (they sum to 1), and the number of prompt tokens."""

    def yes_no(self, state: Sequence[Part], question: str) -> Tuple[float, int]:
        """The probability of "yes", and the number of prompt tokens."""


class LabelTokenError(RuntimeError):
    """A label is not one token, or the server did not return its log-probability."""
