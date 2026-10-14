"""How a decision is written into a prompt.

A prompt is a list of *parts*: strings, and ``{"image": url}`` dicts (a data URL or an http URL). A choice puts its
options into the state under letter labels (``Option A:``, ``Option B:``, ...); each option can be text, images or
both. The answer is read at the label tokens.
"""
from dataclasses import dataclass
from typing import Dict, List, Sequence, Union

Part = Union[str, Dict[str, str]]

LABELS = "ABCDEFGHIJKLMNOP"
MAX_OPTIONS = len(LABELS)
YESNO_WORDS = ("yes", "Yes", "no", "No")  # both casings: a chat model may answer "Yes"


def letters(k: int) -> List[str]:
    if not 2 <= k <= MAX_OPTIONS:
        raise ValueError(f"a choice has 2 to {MAX_OPTIONS} options, got {k}")
    return list(LABELS[:k])


@dataclass
class ChoicePrompt:
    """A choice ready for a backend: the state with the labelled options in it, and the question."""

    state: List[Part]
    question: str
    letters: List[str]
    word: str = "Option"

    @property
    def option_names(self) -> List[str]:
        return [f"{self.word} {x}" for x in self.letters]

    def chat_instruction(self) -> str:
        """The question as the last line of a chat prompt, read at the first answer token."""
        return (f"\n{self.question} Answer with the letter of the best {self.word.lower()} only "
                f"({self.letters[0]}-{self.letters[-1]}).")


def choice_prompt(state: Sequence[Part], question: str, options: Sequence[Sequence[Part]], word: str = "Option",
                  label_suffix: str = ": ") -> ChoicePrompt:
    """`state`, then ``"\\n<word>s:"`` and each option under ``"\\n<word> <letter><label_suffix>"``."""
    labs = letters(len(options))
    parts = list(state) + [f"\n{word}s:"]
    for lab, opt in zip(labs, options):
        parts += [f"\n{word} {lab}{label_suffix}"] + list(opt)
    return ChoicePrompt(state=parts, question=question, letters=labs, word=word)


def yes_no_chat_instruction(question: str) -> str:
    return f"\n{question} Answer only yes or no."


def chat_content(parts: Sequence[Part]) -> List[dict]:
    """Parts as OpenAI-style chat content."""
    return [{"type": "text", "text": p} if isinstance(p, str)
            else {"type": "image_url", "image_url": {"url": p["image"]}} for p in parts]
