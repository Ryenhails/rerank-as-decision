"""The listwise readout: rank up to 16 candidates from one forward pass of a vision-language model.

The query and the candidates go into one prompt under the labels A..P; the next-token probabilities of the labels are
the ranking. These are the prompts behind every number of the technical report.
"""
import concurrent.futures as cf
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

from .images import DEFAULT_MAX_PIXELS
from .inputs import to_parts
from .prompts import choice_prompt, letters

DEFAULT_INSTRUCTION = "Given a search query, retrieve relevant candidates that answer the query."
CHOICE_QUESTION = "Which candidate best matches the query, following the instruction?"
YESNO_QUESTION = "Does this candidate match the query, following the instruction?"


@dataclass
class RankResult:
    order: List[int]       # candidate indices, best first (ties keep the input order)
    scores: List[float]    # one score per candidate, in input order (listwise: probabilities that sum to 1)
    prompt_tokens: int
    calls: int             # model calls: 1 listwise, K pointwise


class Ranker:
    """`backend`: a `ChatBackend`, `JevDecideBackend` or any object with `choice` and `yes_no` (see
    `rerank_as_decision.backends.base`). `readout`: ``"listwise"`` (one prompt with all candidates under the labels
    A..P, ranked by the probability of each label) or ``"pointwise"`` (one yes/no prompt per candidate, sent `workers`
    at a time)."""

    def __init__(self, backend, readout: str = "listwise", max_pixels: int = DEFAULT_MAX_PIXELS, workers: int = 16):
        if readout not in ("listwise", "pointwise"):
            raise ValueError(f"readout must be 'listwise' or 'pointwise', got {readout!r}")
        self.backend, self.readout, self.max_pixels, self.workers = backend, readout, max_pixels, workers

    def rank(self, query: Any, candidates: Sequence[Any], instruction: Optional[str] = None) -> RankResult:
        """Rank 2 to 16 `candidates` (text, images or both, see `to_parts`) for `query`."""
        k = len(candidates)
        if k < 2:
            return RankResult(order=list(range(k)), scores=[1.0] * k, prompt_tokens=0, calls=0)
        letters(k)  # 2 <= k <= 16
        head = [f"Instruction: {instruction or DEFAULT_INSTRUCTION}\nQuery:"] + to_parts(query, self.max_pixels)
        cands = [to_parts(c, self.max_pixels) for c in candidates]
        if self.readout == "listwise":
            prompt = choice_prompt(head, CHOICE_QUESTION, cands, word="Candidate", label_suffix=":")
            probs, tokens = self.backend.choice(prompt)
            scores, calls = list(probs), 1
        else:
            with cf.ThreadPoolExecutor(min(self.workers, k)) as ex:
                outs = list(ex.map(lambda c: self.backend.yes_no(head + ["\nCandidate:"] + c, YESNO_QUESTION), cands))
            scores, tokens, calls = [s for s, _ in outs], sum(t for _, t in outs), k
        order = sorted(range(k), key=lambda i: (-scores[i], i))
        return RankResult(order=order, scores=scores, prompt_tokens=tokens, calls=calls)
