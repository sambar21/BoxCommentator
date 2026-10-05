import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from src.config.backends import Backend, load_backends, load_routing
from src.generation.llm_interface.llm_factory import LLMFactory
from src.generation.llm_interface.openai_compat_client import OpenAICompatClient
from src.generation.timing import collect_timed

WORDS = ["Down", " he", " goes", "!"]


class FakeOpenAI(BaseHTTPRequestHandler):
    """Minimal /v1/chat/completions: SSE streaming with a final usage chunk."""
    seen = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeOpenAI.seen.append(body)
        if body["messages"][0]["content"] == "FAIL":
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b'{"error":"boom"}')
            return
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for w in WORDS:
                chunk = {"choices": [{"delta": {"content": w}}]}
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            usage = {"choices": [], "usage": {
                "prompt_tokens": 120, "completion_tokens": 4,
                "prompt_tokens_details": {"cached_tokens": 96}}}
            self.wfile.write(f"data: {json.dumps(usage)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
        else:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "choices": [{"message": {"content": " Down he goes! "}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4},
            }).encode())


@pytest.fixture
def server():
    FakeOpenAI.seen = []
    srv = HTTPServer(("127.0.0.1", 0), FakeOpenAI)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}/v1"
    srv.shutdown()


def _client(url, **kw):
    return OpenAICompatClient(Backend(name="fake", base_url=url, model="m", api_key="k"), **kw)


def test_streaming_ttft_and_usage(server):
    c = _client(server)
    text, ttft, total, n = collect_timed(c.generate_streaming("hi"))
    assert text == "Down he goes!"
    assert n == 4 and ttft is not None and 0 <= ttft <= total
    assert c.last_completion_tokens == 4
    assert c.last_prompt_tokens == 120
    assert c.last_cached_tokens == 96
    assert FakeOpenAI.seen[0]["stream_options"] == {"include_usage": True}


def test_non_streaming(server):
    c = _client(server)
    assert c.generate("hi") == "Down he goes!"
    assert c.last_completion_tokens == 4


def test_errors_are_counted_not_hidden(server):
    c = _client(server)
    assert c.generate("FAIL") == ""
    assert list(c.generate_streaming("FAIL")) == []
    assert c.error_count == 2
    assert "500" in c.last_error or "InternalServerError" in c.last_error


def test_usage_is_thread_local(server):
    c = _client(server)
    results = {}

    def run(name):
        collect_timed(c.generate_streaming("hi"))
        results[name] = c.last_completion_tokens

    c._tl.completion_tokens = 999        # main thread value must not leak into workers
    ts = [threading.Thread(target=run, args=(i,)) for i in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert set(results.values()) == {4}
    assert c.last_completion_tokens == 999


def test_yaml_loads_and_expands_env(monkeypatch):
    monkeypatch.setenv("VLLM_BASE_URL", "https://tunnel.example/v1")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("NEBIUS_MODEL", raising=False)
    b = load_backends()
    assert b["vllm-3b"].base_url == "https://tunnel.example/v1"
    assert b["vllm-3b"].self_hosted
    assert b["groq-70b"].api_key == ""
    assert not b["nebius"].configured          # unset until the user fills it in
    assert load_routing()["default"] in b


def test_factory_builds_backend_by_name(monkeypatch):
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)
    client = LLMFactory.create_client("vllm-3b")
    assert isinstance(client, OpenAICompatClient)
    assert client.get_provider_name() == "vllm-3b"
    assert client.model == "Qwen/Qwen2.5-3B-Instruct"


def test_factory_rejects_unconfigured_backend(monkeypatch):
    monkeypatch.delenv("NEBIUS_BASE_URL", raising=False)
    monkeypatch.delenv("NEBIUS_MODEL", raising=False)
    with pytest.raises(ValueError, match="missing base_url/model"):
        LLMFactory.create_client("nebius")
    with pytest.raises(ValueError, match="Unknown provider"):
        LLMFactory.create_client("nope")
