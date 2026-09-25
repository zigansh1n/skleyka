"""Fallback model behaviour of VisionAPI (primary 5xx -> configured fallback)."""

import json
from contextlib import contextmanager

import pytest

from openscenesense.montage import provider


class _FakeResponse:
    def __init__(self, status_code: int, body: dict | None = None):
        self.status_code = status_code
        self.headers = {"content-type": "application/json"}
        self._body = json.dumps(body or {}).encode()

    def iter_bytes(self):
        yield self._body


class _FakeClient:
    """Stands in for httpx.Client; answers by model name."""

    responses: dict[str, list[int]] = {}
    seen: list[str] = []

    def __init__(self, *a, **kw):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    @contextmanager
    def stream(self, method, url, headers=None, content=None):
        model = json.loads(content)["model"]
        _FakeClient.seen.append(model)
        codes = _FakeClient.responses[model]
        code = codes.pop(0) if len(codes) > 1 else codes[0]
        if code == 200:
            body = {
                "status": "completed",
                "output": [
                    {
                        "content": [
                            {
                                "type": "output_text",
                                "text": json.dumps({"ok": True, "model": model}),
                            }
                        ]
                    }
                ],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            }
            yield _FakeResponse(200, body)
        else:
            yield _FakeResponse(code)


def _config(**extra):
    cfg = {
        "MONTAGE_API_KEY": "k",
        "MONTAGE_BASE_URL": "https://example.invalid/v1",
        "MONTAGE_MODEL": "deepseek-flash",
        "MONTAGE_API_MODE": "responses",
        "MONTAGE_REASONING_EFFORT": "low",
        "MONTAGE_ALLOWED_MODELS": "deepseek-flash,gpt-5.6-luna",
    }
    cfg.update(extra)
    return cfg


@pytest.fixture(autouse=True)
def _fake_httpx(monkeypatch):
    _FakeClient.seen = []
    monkeypatch.setattr(provider.httpx, "Client", _FakeClient)
    monkeypatch.setattr(provider.time, "sleep", lambda *_: None)


def test_falls_back_to_luna_after_primary_5xx():
    _FakeClient.responses = {"deepseek-flash": [503, 503], "gpt-5.6-luna": [200]}
    api = provider.VisionAPI(_config(MONTAGE_FALLBACK_MODEL="gpt-5.6-luna"))
    data = api.request("p", [])
    assert data["model"] == "gpt-5.6-luna"
    assert api.fallback_used is True
    assert _FakeClient.seen == ["deepseek-flash", "deepseek-flash", "gpt-5.6-luna"]


def test_no_fallback_configured_raises():
    _FakeClient.responses = {"deepseek-flash": [503, 503]}
    api = provider.VisionAPI(_config())
    with pytest.raises(RuntimeError, match="HTTP 503|retry exhausted"):
        api.request("p", [])
    assert api.fallback_used is False


def test_primary_success_never_touches_fallback():
    _FakeClient.responses = {"deepseek-flash": [200], "gpt-5.6-luna": [200]}
    api = provider.VisionAPI(_config(MONTAGE_FALLBACK_MODEL="gpt-5.6-luna"))
    assert api.request("p", [])["model"] == "deepseek-flash"
    assert _FakeClient.seen == ["deepseek-flash"]


def test_fallback_outside_allow_list_is_rejected():
    with pytest.raises(ValueError, match="Fallback model"):
        provider.VisionAPI(_config(MONTAGE_FALLBACK_MODEL="claude-opus-5"))


def test_non_provider_error_does_not_fall_back():
    _FakeClient.responses = {"deepseek-flash": [401], "gpt-5.6-luna": [200]}
    api = provider.VisionAPI(_config(MONTAGE_FALLBACK_MODEL="gpt-5.6-luna"))
    with pytest.raises(RuntimeError, match="HTTP 401"):
        api.request("p", [])
    assert _FakeClient.seen == ["deepseek-flash"]
