"""VLM-as-Jev: typed decisions from a vision-language model, in one forward pass.

Give the model a state (images, text or both) and a question with 2 to 16 options; the options go into the prompt
under the labels A..P, and the next-token probabilities of the labels are the decision. Nothing is decoded and no
weights are trained. `Reranker` applies this to multimodal retrieval, the setting of the technical report.
"""
from .backends import ChatBackend, FakeBackend, JevDecideBackend, LabelTokenError
from .decider import Decider, Decision, to_parts
from .images import DEFAULT_MAX_PIXELS, to_data_url
from .prompts import LABELS, MAX_OPTIONS
from .reranker import Reranker, RerankResult

__version__ = "0.1.0"

__all__ = ["ChatBackend", "DEFAULT_MAX_PIXELS", "Decider", "Decision", "FakeBackend", "JevDecideBackend", "LABELS",
           "LabelTokenError", "MAX_OPTIONS", "RerankResult", "Reranker", "to_data_url", "to_parts", "__version__"]
