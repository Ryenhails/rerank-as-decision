import math

import pytest

from listwise_rerank import ChatBackend, JevDecideBackend, LabelTokenError
from listwise_rerank.prompts import choice_prompt

from .conftest import CharTokenizer

STATE = ["Look at this:", {"image": "data:image/jpeg;base64,S"}]
OPTIONS = [[{"image": f"data:image/jpeg;base64,{i}"}] for i in range(16)]


def test_chat_choice_request_and_readout(server):
    backend = ChatBackend(server.url, model="m", tokenizer=CharTokenizer(), api_key="secret")
    probs, tokens = backend.choice(choice_prompt(STATE, "Which one?", OPTIONS))
    (path, headers, body), = server.requests
    assert path == "/v1/chat/completions"
    assert headers["Authorization"] == "Bearer secret"
    ids = [ord(c) for c in "ABCDEFGHIJKLMNOP"]
    assert body["allowed_token_ids"] == ids and body["top_logprobs"] == 16
    assert body["max_tokens"] == 1 and body["logprobs"] and body["return_tokens_as_token_ids"]
    assert (body["temperature"], body["top_p"], body["top_k"]) == (1.0, 1.0, 0)
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    texts = [c["text"] for c in body["messages"][0]["content"] if c["type"] == "text"]
    assert texts[:3] == ["Look at this:", "\nOptions:", "\nOption A: "]
    assert texts[-1] == "\nWhich one? Answer with the letter of the best option only (A-P)."
    z = [server.logprob(i) for i in ids]
    assert probs == pytest.approx([math.exp(x) / sum(map(math.exp, z)) for x in z]) and tokens == 13


def test_chat_yes_no_reads_both_casings(server):
    backend = ChatBackend(server.url, model="m", tokenizer=CharTokenizer(), api_key="")
    p, _ = backend.yes_no(STATE, "Is it a cat?")
    (_, headers, body), = server.requests
    assert "Authorization" not in headers
    words = ["yes", "Yes", "no", "No"]
    assert body["allowed_token_ids"] == [sum(map(ord, w)) for w in words]
    assert body["messages"][0]["content"][-1]["text"] == "\nIs it a cat? Answer only yes or no."
    e = [math.exp(server.logprob(sum(map(ord, w)))) for w in words]
    assert p == pytest.approx((e[0] + e[1]) / sum(e))


def test_chat_missing_label_raises(server):
    backend = ChatBackend(server.url, model="m", tokenizer=CharTokenizer())
    server.drop = {ord("C")}
    with pytest.raises(LabelTokenError, match="processed_logprobs"):
        backend.choice(choice_prompt(STATE, "Which one?", OPTIONS[:4]))


def test_chat_multi_token_label_raises():
    class TwoTokens:
        def encode(self, text, add_special_tokens=False):
            return [1, 2]

    with pytest.raises(LabelTokenError, match="2 tokens"):
        ChatBackend("http://unused", model="m", tokenizer=TwoTokens()).token_ids(["A"])


def test_jev_choice_and_yes_no(server):
    backend = JevDecideBackend(server.url, api_key="k")
    probs, tokens = backend.choice(choice_prompt(STATE, "Which one?", OPTIONS[:3], word="Candidate", label_suffix=":"))
    p_yes, _ = backend.yes_no(STATE, "Is it a cat?")
    (p1, _, choice), (p2, _, noul) = server.requests
    assert p1 == p2 == "/v1/decide"
    assert choice["kind"] == "choice" and choice["options"] == ["Candidate A", "Candidate B", "Candidate C"]
    assert choice["question"] == "Which one?" and choice["strategy"] == "single" and choice["thinking"] == "off"
    assert choice["state"][:4] == STATE + ["\nCandidates:", "\nCandidate A:"]
    assert noul == {"kind": "noul", "state": STATE, "question": "Is it a cat?", "thinking": "off"}
    assert probs == pytest.approx([1 / 6, 2 / 6, 3 / 6]) and tokens == 11
    assert p_yes == pytest.approx(2 / 3)
