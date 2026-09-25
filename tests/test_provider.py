import json
from types import SimpleNamespace

from openscenesense.models import Frame, SceneType
from openscenesense.providers.openai_provider import OpenAIProvider


class Recorder:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            output_text=self.outputs.pop(0),
            usage=SimpleNamespace(input_tokens=10, output_tokens=5, total_tokens=15),
        )


def test_responses_provider_uses_image_schema_and_usage():
    recorder = Recorder(
        [
            json.dumps(
                {
                    "description": "A test frame",
                    "objects": ["square"],
                    "actions": [],
                    "visible_text": [],
                    "tags": ["test"],
                }
            )
        ]
    )
    client = SimpleNamespace(responses=recorder)
    provider = OpenAIProvider(client, "vision", "summary")
    frame = Frame(None, 1.0, SceneType.STATIC, selection_reason="opening")

    analysis, usage = provider.analyze_frame(frame, "encoded", "describe")

    assert analysis.description == "A test frame"
    assert usage.total_tokens == 15
    call = recorder.calls[0]
    assert call["store"] is False
    assert call["text"]["format"]["type"] == "json_schema"
    assert call["input"][0]["content"][1]["type"] == "input_image"


def test_summary_is_generated_in_one_structured_call():
    recorder = Recorder([json.dumps({"detailed": "long", "brief": "short", "events": []})])
    provider = OpenAIProvider(SimpleNamespace(responses=recorder), "vision", "summary")

    summary, events, usage = provider.generate_summary("prompt", "timeline", "transcript")

    assert summary.detailed == "long"
    assert summary.brief == "short"
    assert events == []
    assert usage.total_tokens == 15
    assert len(recorder.calls) == 1


def test_frames_are_spliced_as_bytes_without_touching_prompt_text(tmp_path, monkeypatch):
    import base64
    import json

    import httpx

    from openscenesense.montage import provider

    seen = {}

    def handle(request):
        seen["body"] = json.loads(request.content)
        message = {"message": {"content": '{"ok": 1}'}, "finish_reason": "stop"}
        usage = {"prompt_tokens": 1, "completion_tokens": 1}
        return httpx.Response(200, json={"choices": [message], "usage": usage})

    client = httpx.Client(transport=httpx.MockTransport(handle))
    monkeypatch.setitem(provider._CLIENTS, httpx.Client, client)
    frames = []
    for name, data in (("a.jpg", b"\xff\xd8one"), ("b.jpg", b"\xff\xd8two\x00")):
        (tmp_path / name).write_bytes(data)
        frames.append((tmp_path / name, len(frames) + 1.0))
    api = provider.VisionAPI(
        {
            "MONTAGE_API_KEY": "k",
            "MONTAGE_BASE_URL": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "MONTAGE_MODEL": "kimi-k3",
            "MONTAGE_API_MODE": "chat_completions",
            "MONTAGE_REASONING_EFFORT": "none",
        }
    )
    prompt = 'quoted "text" with @@IMAGE0@@ and @@0123456789abcdef:1@@'
    assert api.request(prompt, frames) == {"ok": 1}
    body = seen["body"]
    content = body["messages"][0]["content"]
    assert content[0]["text"] == prompt
    urls = [content[i]["image_url"]["url"] for i in (2, 4)]
    assert [base64.b64decode(u.split(",", 1)[1]) for u in urls] == [
        p.read_bytes() for p, _ in frames
    ]
    assert body["stream"] is False and body["enable_thinking"] is False
