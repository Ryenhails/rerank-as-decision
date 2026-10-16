"""JEV-27B-VL's System One endpoint (``POST /v1/decide``), as served by the model's own server.

A choice is one ``choice`` decision over the option names ("Option A", ...), whose contents are in the state; a
yes/no question is one ``noul`` decision. Both are read by the model's decision head, with thinking off. See the
model card of autotrust/JEV-27B-VL for the server command; the report's runs used vLLM 0.30.0.
"""
import json
import os
import urllib.request
from typing import List, Optional, Sequence, Tuple

from ..prompts import ChoicePrompt, Part


class JevDecideBackend:
    def __init__(self, base_url: str, api_key: Optional[str] = None, timeout: float = 1800):
        self.base_url = base_url.rstrip("/")
        self.name = "jev:/v1/decide"
        self.api_key = os.environ.get("VLLM_API_KEY", "") if api_key is None else api_key
        self.timeout = timeout

    def choice(self, prompt: ChoicePrompt) -> Tuple[List[float], int]:
        r = self._post({"kind": "choice", "state": list(prompt.state), "question": prompt.question,
                        "options": prompt.option_names, "strategy": "single", "thinking": "off"})
        return r["probabilities"], r["usage"]["prompt_tokens"]

    def yes_no(self, state: Sequence[Part], question: str) -> Tuple[float, int]:
        r = self._post({"kind": "noul", "state": list(state), "question": question, "thinking": "off"})
        return r["probabilities"][1], r["usage"]["prompt_tokens"]

    def _post(self, body: dict) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.base_url + "/v1/decide", method="POST", data=json.dumps(body).encode(),
                                     headers=headers)
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.load(r)
