"""A local HTTP server that records requests and answers like vLLM's chat endpoint or JEV's /v1/decide."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest


class FakeServer:
    def __init__(self):
        self.requests = []          # (path, headers, body)
        self.drop = set()           # token ids left out of top_logprobs, as vLLM's raw mode can
        self.logprob = lambda token_id: -float(token_id % 7)

    def answer(self, path, body):
        if path == "/v1/decide":
            n = len(body.get("options", ["no", "yes"]))
            return {"probabilities": [(i + 1) / (n * (n + 1) / 2) for i in range(n)], "usage": {"prompt_tokens": 11}}
        ids = [i for i in body["allowed_token_ids"] if i not in self.drop]
        top = [{"token": f"token_id:{i}", "logprob": self.logprob(i)} for i in ids]
        return {"choices": [{"logprobs": {"content": [{"top_logprobs": top}]}}], "usage": {"prompt_tokens": 13}}


@pytest.fixture
def server():
    fake = FakeServer()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            fake.requests.append((self.path, dict(self.headers), body))
            data = json.dumps(fake.answer(self.path, body)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    fake.url = f"http://127.0.0.1:{httpd.server_port}"
    yield fake
    httpd.shutdown()


class CharTokenizer:
    """Every label is one token: its code point."""

    def encode(self, text, add_special_tokens=False):
        return [sum(map(ord, text))]
