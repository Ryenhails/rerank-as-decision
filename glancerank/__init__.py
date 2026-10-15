"""GlanceRank: zero-shot System One ranking with a vision-language model.

The query and up to 16 candidates (images, document pages, text, or a mix) go into one prompt under the labels A..P;
the next-token probabilities of the labels rank the candidates, from one forward pass, with no decoding and no training.
"""
from .backends import ChatBackend, FakeBackend, JevDecideBackend, LabelTokenError
from .images import DEFAULT_MAX_PIXELS, to_data_url
from .inputs import to_parts
from .prompts import LABELS, MAX_OPTIONS
from .ranker import Ranker, RankResult

MAX_CANDIDATES = MAX_OPTIONS
__version__ = "0.1.0"

__all__ = ["ChatBackend", "DEFAULT_MAX_PIXELS", "FakeBackend", "JevDecideBackend", "LABELS", "LabelTokenError",
           "MAX_CANDIDATES", "RankResult", "Ranker", "to_data_url", "to_parts", "__version__"]
