import json
import os
import sys
from types import SimpleNamespace

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from plugins.agent_detection_plugin import GeminiAgentPlugin  # noqa: E402


def fake_response(content: str):
    """Mimic the shape of an OpenAI chat completion response."""
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class FakeClient:
    """Stand-in for the OpenAI client that returns a canned reply."""

    def __init__(self, reply: str):
        self.reply = reply
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return fake_response(self.reply)


def detections_json(items):
    return json.dumps(
        [
            {"id": f"obj{i}", "label": "thing", "confidence": conf, "box_2d": box}
            for i, (conf, box) in enumerate(items)
        ]
    )


@pytest.fixture
def make_plugin(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.chdir(tmp_path)  # plugin creates debug_output/ in cwd

    def _make(reply: str = "[]"):
        plugin = GeminiAgentPlugin()
        plugin.client = FakeClient(reply)
        return plugin

    return _make
