from __future__ import annotations

import base64
import json
import os
import re
import secrets
import stat
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from dotenv import dotenv_values

DEFAULT_CONFIG = Path.home() / ".config/video-montage/config.env"


def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    config = {}
    if path.exists():
        if stat.S_IMODE(path.stat().st_mode) & 0o077:
            raise ValueError("API configuration must have permissions 0600")
        config.update({k: v for k, v in dotenv_values(path).items() if v})
    # Deliberately do not read ANTHROPIC_*, OPENAI_* or Hermes/Codex auth files.
    config.update({k: v for k, v in os.environ.items() if k.startswith("MONTAGE_")})
    for field in ("MONTAGE_API_KEY", "MONTAGE_BASE_URL", "MONTAGE_MODEL"):
        if not config.get(field):
            raise ValueError(f"Missing {field} in dedicated video-montage configuration")
    config.setdefault("MONTAGE_API_MODE", "responses")
    config.setdefault("MONTAGE_REASONING_EFFORT", "low")
    if config["MONTAGE_REASONING_EFFORT"] not in ("none", "low", "high", "max"):
        raise ValueError("MONTAGE_REASONING_EFFORT must be none, low, high or max")
    if config["MONTAGE_API_MODE"] not in ("responses", "chat_completions"):
        raise ValueError("API mode must be responses or chat_completions")
    url = urlsplit(config["MONTAGE_BASE_URL"])
    if url.username or url.password or url.query or url.fragment:
        raise ValueError("API base URL must not contain credentials, query or fragment")
    if url.scheme != "https" and not (
        url.scheme == "http" and url.hostname in ("127.0.0.1", "localhost", "::1")
    ):
        raise ValueError("Remote API endpoints must use HTTPS")
    return config


def parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Model returned a non-object JSON response")
    return data


def validate_analysis(data: dict, frame_count: int) -> dict:
    if not isinstance(data.get("summary"), str) or not data["summary"].strip():
        raise ValueError("Missing analysis summary")
    observations = data.get("observations")
    if not isinstance(observations, list) or not observations:
        raise ValueError("Missing evidence-backed observations")
    for obs in observations:
        if not isinstance(obs, dict) or not isinstance(obs.get("description"), str):
            raise ValueError("Invalid observation")
        ids = obs.get("frames")
        if (
            not isinstance(ids, list)
            or not ids
            or any(type(i) is not int or i < 1 or i > frame_count for i in ids)
        ):
            raise ValueError("Observation cites a missing evidence frame")
        if obs.get("confidence") not in ("high", "medium", "low"):
            raise ValueError("Invalid confidence")
    for field in ("recreation_steps", "uncertainties"):
        if not isinstance(data.get(field), list) or any(
            not isinstance(x, str) for x in data[field]
        ):
            raise ValueError(f"Missing or invalid {field}")
    return data


_CLIENTS: dict = {}
_CLIENTS_GUARD = threading.Lock()


def shared_client(factory) -> httpx.Client:
    """One pooled client per process: a fresh client costs ~30 ms of CPU loading CA
    certificates, and a fresh TLS connection one more round trip per request."""
    with _CLIENTS_GUARD:
        if factory not in _CLIENTS:
            _CLIENTS[factory] = factory(
                timeout=httpx.Timeout(300, connect=30),
                follow_redirects=False,
                limits=httpx.Limits(max_connections=None, max_keepalive_connections=2000),
            )
        return _CLIENTS[factory]


class TransientAPIError(RuntimeError):
    """Overload or gateway failure (429/5xx): the same request may succeed later, and
    it is the only failure that may switch to the configured fallback model."""


ProviderUnavailable = TransientAPIError


class VisionAPI:
    def __init__(self, config: dict, max_attempts: int = 2):
        allowed = {
            x.strip() for x in config.get("MONTAGE_ALLOWED_MODELS", "").split(",") if x.strip()
        }
        if allowed and config["MONTAGE_MODEL"] not in allowed:
            raise ValueError(
                "Model is outside the dedicated allowed-model list; no API request sent"
            )
        fallback = (config.get("MONTAGE_FALLBACK_MODEL") or "").strip()
        if fallback == config["MONTAGE_MODEL"]:
            fallback = ""
        if fallback and allowed and fallback not in allowed:
            raise ValueError(
                "Fallback model is outside the dedicated allowed-model list; no API request sent"
            )
        self.fallback_model = fallback or None
        self.fallback_used = False
        self.config = config
        self.calls = 0
        self.usage = []
        self.max_attempts = max_attempts

    @property
    def identity(self) -> dict:
        identity = {
            k: self.config[k] for k in ("MONTAGE_BASE_URL", "MONTAGE_MODEL", "MONTAGE_API_MODE")
        }
        identity["MONTAGE_REASONING_EFFORT"] = self.config.get("MONTAGE_REASONING_EFFORT", "low")
        identity["MONTAGE_FALLBACK_MODEL"] = self.fallback_model or ""
        return identity

    def request(self, prompt: str, frames: list[tuple[Path, float]]) -> dict:
        mode = self.config["MONTAGE_API_MODE"]
        content = [{"type": "input_text" if mode == "responses" else "text", "text": prompt}]
        # A per-request random slot cannot collide with prompt or transcript text.
        slot, images = f"@@{secrets.token_hex(8)}:%d@@", []
        for i, (path, timestamp) in enumerate(frames, 1):
            content.append(
                {
                    "type": "input_text" if mode == "responses" else "text",
                    "text": f"Frame {i}; absolute source time {timestamp:.6f} seconds",
                }
            )
            images.append(base64.b64encode(path.read_bytes()))
            url = slot % (len(images) - 1)
            if mode == "responses":
                content.append({"type": "input_image", "image_url": url, "detail": "high"})
            else:
                content.append({"type": "image_url", "image_url": {"url": url, "detail": "high"}})
        models = [self.config["MONTAGE_MODEL"]] + [self.fallback_model] * bool(self.fallback_model)
        for model_index, model in enumerate(models):
            try:
                return self._request_model(model, mode, content, slot, images)
            except TransientAPIError as exc:
                if model_index + 1 == len(models):
                    raise
                self.fallback_used = True
                print(
                    f"[video-montage] provider error with {model} ({exc}); "
                    f"falling back to {models[model_index + 1]}",
                    file=sys.stderr,
                )

    def _request_model(self, model, mode, content, slot, images) -> dict:
        payload = {"model": model, "stream": True}
        output_limit = int(self.config.get("MONTAGE_MAX_OUTPUT_TOKENS", "8192"))
        if not 1024 <= output_limit <= 32768:
            raise ValueError("MONTAGE_MAX_OUTPUT_TOKENS must be between 1024 and 32768")
        if mode == "responses":
            payload.update(
                {
                    "input": [{"role": "user", "content": content}],
                    "max_output_tokens": output_limit,
                    "store": False,
                    "reasoning": {"effort": self.config.get("MONTAGE_REASONING_EFFORT", "low")},
                }
            )
            endpoint = "/responses"
        else:
            payload.update(
                {"messages": [{"role": "user", "content": content}], "max_tokens": output_limit}
            )
            endpoint = "/chat/completions"
            effort = self.config.get("MONTAGE_REASONING_EFFORT", "low")
            if "deepseek" in model.lower():
                payload["thinking"] = {"type": "disabled" if effort == "none" else "enabled"}
                if effort != "none":
                    payload["reasoning_effort"] = effort
            if urlsplit(self.config["MONTAGE_BASE_URL"]).hostname.endswith(".aliyuncs.com"):
                # DashScope: thinking is a separate switch. One JSON body instead of SSE:
                # hundreds of parallel token streams saturate the GIL parsing events.
                payload["enable_thinking"] = effort != "none"
                if effort != "none":
                    payload["reasoning_effort"] = effort
                payload["stream"] = False
            # Provider-specific extras, merged last, e.g. {"enable_thinking": false}.
            if self.config.get("MONTAGE_EXTRA_BODY"):
                payload.update(json.loads(self.config["MONTAGE_EXTRA_BODY"]))
        # Base64 is JSON-safe, so frames are spliced in as bytes: running megabytes of image
        # data through json.dumps held the GIL for most of a many-worker run.
        marker = re.escape(slot.split(":")[0].encode()) + rb":(\d+)@@"
        skeleton = re.split(marker, json.dumps(payload, ensure_ascii=False).encode())
        body = b"".join(
            part if n % 2 == 0 else b"data:image/jpeg;base64," + images[int(part)]
            for n, part in enumerate(skeleton)
        )
        headers = {
            "Authorization": "Bearer " + self.config["MONTAGE_API_KEY"],
            "Content-Type": "application/json",
        }
        for attempt in range(self.max_attempts):
            self.calls += 1
            client = shared_client(httpx.Client)
            with client.stream(
                "POST",
                self.config["MONTAGE_BASE_URL"].rstrip("/") + endpoint,
                headers=headers,
                content=body,
            ) as response:
                if (
                    response.status_code in (429, 500, 502, 503, 504)
                    and attempt + 1 < self.max_attempts
                ):
                    time.sleep(2)
                    continue
                if response.status_code in (429, 500, 502, 503, 504):
                    raise TransientAPIError(f"Vision API HTTP {response.status_code} from {model}")
                if response.status_code != 200:
                    raise RuntimeError(
                        f"Vision API HTTP {response.status_code} from {model}; not retried"
                    )
                text, usage = self._read(response)
            # Keep portable accounting, not gateway-specific per-item attribution payloads.
            self.usage.append(
                {
                    k: usage[k]
                    for k in (
                        "input_tokens",
                        "output_tokens",
                        "total_tokens",
                        "prompt_tokens",
                        "completion_tokens",
                        "input_tokens_details",
                        "output_tokens_details",
                    )
                    if k in usage
                }
            )
            return parse_json(text)
        raise TransientAPIError(f"Vision API retry exhausted for {model}")

    @staticmethod
    def _read(response: httpx.Response) -> tuple[str, dict]:
        if "text/event-stream" not in response.headers.get("content-type", ""):
            raw = b""
            for chunk in response.iter_bytes():
                raw += chunk
                if len(raw) > 4 * 1024 * 1024:
                    raise ValueError("API response exceeds limit")
            data = json.loads(raw)
            if data.get("choices"):
                choice = data["choices"][0]
                if choice.get("finish_reason") not in (None, "stop"):
                    raise ValueError("API output incomplete")
                return choice["message"]["content"], data.get("usage", {})
            if data.get("status") not in (None, "completed"):
                raise ValueError("API output incomplete")
            return "".join(
                p.get("text", "")
                for o in data.get("output", [])
                for p in o.get("content", [])
                if p.get("type") == "output_text"
            ), data.get("usage", {})
        chunks, usage, complete, size = [], {}, False, 0
        for line in response.iter_lines():
            size += len(line)
            if size > 4 * 1024 * 1024:
                raise ValueError("API stream exceeds limit")
            if not line.startswith("data:"):
                continue
            raw = line[5:].strip()
            if raw == "[DONE]":
                # Chat completion still requires an explicit finish_reason=stop.
                continue
            if not raw:
                continue
            event = json.loads(raw)
            kind = event.get("type")
            if kind == "response.output_text.delta":
                chunks.append(event.get("delta", ""))
            elif kind == "response.completed":
                final = event.get("response", {})
                complete = final.get("status") == "completed"
                usage = final.get("usage", {})
                if not chunks:
                    chunks = [
                        p.get("text", "")
                        for o in final.get("output", [])
                        for p in o.get("content", [])
                        if p.get("type") == "output_text"
                    ]
            elif kind in ("error", "response.failed", "response.incomplete"):
                raise ValueError("API generation failed or was truncated")
            for choice in event.get("choices", []):
                chunks.append(choice.get("delta", {}).get("content") or "")
                if choice.get("finish_reason") == "stop":
                    complete = True
                elif choice.get("finish_reason"):
                    raise ValueError("API generation was truncated")
            if event.get("usage"):
                usage = event["usage"]
        if not complete:
            raise ValueError("API stream ended without confirmed completion")
        return "".join(chunks), usage
