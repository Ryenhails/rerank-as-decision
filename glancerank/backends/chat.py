"""Any vision-language model behind an OpenAI-compatible chat endpoint that returns log-probabilities (vLLM).

The readout asks for one answer token with ``allowed_token_ids`` set to the label tokens and
``top_logprobs = len(labels)``, and normalizes the label log-probabilities. Nothing is sampled that the readout uses.

Start the server with processed log-probabilities, so that the returned top log-probabilities are exactly the labels:

    vllm serve <model> --logprobs-mode processed_logprobs --limit-mm-per-prompt '{"image": 20}' --max-model-len 65536

With vLLM's default (raw) mode the server returns the model's own top tokens, and a label can be missing; this
backend then raises `LabelTokenError` instead of guessing.
"""
import json
import math
import os
import urllib.request
from typing import Dict, List, Optional, Sequence, Tuple

from ..prompts import YESNO_WORDS, ChoicePrompt, Part, chat_content, yes_no_chat_instruction
from .base import LabelTokenError


class ChatBackend:
    """A chat model served by vLLM, read at its first answer token with thinking off.

    `tokenizer` (a name, a path or an object with ``encode(text, add_special_tokens=False)``) maps the labels to
    token ids; it defaults to `model`. `api_key` defaults to ``$VLLM_API_KEY``.
    """

    def __init__(self, base_url: str, model: str, tokenizer=None, api_key: Optional[str] = None,
                 enable_thinking: bool = False, timeout: float = 1800):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.name = f"chat:{model}"
        self.api_key = os.environ.get("VLLM_API_KEY", "") if api_key is None else api_key
        self.enable_thinking = enable_thinking
        self.timeout = timeout
        self._tokenizer = tokenizer if tokenizer is not None else model
        self._ids: Dict[str, int] = {}

    # ------------------------------------------------------------------ tokens
    def token_ids(self, words: Sequence[str]) -> List[int]:
        """The single-token id of each word."""
        missing = [w for w in words if w not in self._ids]
        if missing:
            tok = self._tokenizer
            if isinstance(tok, str):
                from transformers import AutoTokenizer
                tok = self._tokenizer = AutoTokenizer.from_pretrained(tok)
            for w in missing:
                ids = tok.encode(w, add_special_tokens=False)
                if len(ids) != 1:
                    raise LabelTokenError(f"label {w!r} is {len(ids)} tokens in this tokenizer, the readout needs one")
                self._ids[w] = ids[0]
        return [self._ids[w] for w in words]

    # ------------------------------------------------------------------ the read
    def first_token_probs(self, parts: Sequence[Part], words: Sequence[str]) -> Tuple[List[float], int]:
        """Softmax over `words` of the first answer token; returns (probabilities, prompt tokens)."""
        ids = self.token_ids(words)
        body = {"model": self.model, "messages": [{"role": "user", "content": chat_content(parts)}], "max_tokens": 1,
                "temperature": 1.0, "top_p": 1.0, "top_k": 0, "logprobs": True, "top_logprobs": len(ids),
                "allowed_token_ids": ids, "return_tokens_as_token_ids": True,
                "chat_template_kwargs": {"enable_thinking": self.enable_thinking}}
        r = self._post("/v1/chat/completions", body)
        top = r["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        lp = {int(t["token"].split(":")[1]): t["logprob"] for t in top}
        lost = [w for w, i in zip(words, ids) if i not in lp]
        if lost:
            raise LabelTokenError(f"the server returned no log-probability for {lost}; start vLLM with "
                                  "--logprobs-mode processed_logprobs")
        z = [lp[i] for i in ids]
        m = max(z)
        e = [math.exp(x - m) for x in z]
        s = sum(e)
        return [x / s for x in e], r["usage"]["prompt_tokens"]

    def choice(self, prompt: ChoicePrompt) -> Tuple[List[float], int]:
        return self.first_token_probs(list(prompt.state) + [prompt.chat_instruction()], prompt.letters)

    def yes_no(self, state: Sequence[Part], question: str) -> Tuple[float, int]:
        p, tok = self.first_token_probs(list(state) + [yes_no_chat_instruction(question)], YESNO_WORDS)
        return p[0] + p[1], tok

    def _post(self, path: str, body: dict) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.base_url + path, method="POST", data=json.dumps(body).encode(),
                                     headers=headers)
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.load(r)
