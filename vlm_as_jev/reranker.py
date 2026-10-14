"""Reranking as a decision: the options are the retrieved candidates. This is the setting of the technical report,
and its prompts are the ones behind every number there."""
import concurrent.futures as cf
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

from .decider import Decider, to_parts
from .images import DEFAULT_MAX_PIXELS
from .prompts import letters

DEFAULT_INSTRUCTION = "Given a search query, retrieve relevant candidates that answer the query."
CHOICE_QUESTION = "Which candidate best matches the query, following the instruction?"
YESNO_QUESTION = "Does this candidate match the query, following the instruction?"


@dataclass
class RerankResult:
    order: List[int]       # candidate indices, best first (ties keep the input order)
    scores: List[float]    # one score per candidate, in input order
    prompt_tokens: int
    calls: int             # model calls: 1 listwise, K pointwise


class Reranker:
    """`backend`: a backend or a `Decider`. `readout`: ``"listwise"`` (one prompt with all candidates under the
    labels A..P, ranked by the probability of each label) or ``"pointwise"`` (one yes/no prompt per candidate, sent
    `workers` at a time)."""

    def __init__(self, backend, readout: str = "listwise", max_pixels: int = DEFAULT_MAX_PIXELS, workers: int = 16):
        if readout not in ("listwise", "pointwise"):
            raise ValueError(f"readout must be 'listwise' or 'pointwise', got {readout!r}")
        self.decider = backend if isinstance(backend, Decider) else Decider(backend, max_pixels=max_pixels)
        self.backend = self.decider.backend
        self.readout, self.max_pixels, self.workers = readout, max_pixels, workers

    def rerank(self, query: Any, candidates: Sequence[Any], instruction: Optional[str] = None) -> RerankResult:
        """Rerank 2 to 16 `candidates` (text, images or both, see `to_parts`) for `query`."""
        k = len(candidates)
        if k < 2:
            return RerankResult(order=list(range(k)), scores=[1.0] * k, prompt_tokens=0, calls=0)
        letters(k)  # 2 <= k <= 16
        head = [f"Instruction: {instruction or DEFAULT_INSTRUCTION}\nQuery:"] + to_parts(query, self.max_pixels)
        cands = [to_parts(c, self.max_pixels) for c in candidates]
        if self.readout == "listwise":
            d = self.decider.choice_parts(head, CHOICE_QUESTION, cands, word="Candidate", label_suffix=":")
            scores, tokens, calls = d.probs, d.prompt_tokens, 1
        else:
            with cf.ThreadPoolExecutor(min(self.workers, k)) as ex:
                outs = list(ex.map(lambda c: self.decider.yes_no_parts(head + ["\nCandidate:"] + c, YESNO_QUESTION),
                                   cands))
            scores, tokens, calls = [s for s, _ in outs], sum(t for _, t in outs), k
        order = sorted(range(k), key=lambda i: (-scores[i], i))
        return RerankResult(order=order, scores=list(scores), prompt_tokens=tokens, calls=calls)
