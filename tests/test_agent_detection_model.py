"""End-to-end test of the Agent Detector plugin against a real (small) vision-language model.

Run with:  pytest -m model
Requires torch + transformers (see requirements-test-model.txt). The model is downloaded
from the Hugging Face Hub on first use (~2 GB) and runs on CPU.
"""
import base64
import io
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import fake_response

pytestmark = pytest.mark.model

MODEL_ID = "Qwen/Qwen3.5-0.8B"


class LocalVLMClient:
    """Adapter exposing the OpenAI chat-completions interface on top of a local HF model."""

    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, messages, **_):
        from PIL import Image

        content = []
        for part in messages[0]["content"]:
            if part["type"] == "text":
                content.append({"type": "text", "text": part["text"]})
            else:
                b64 = part["image_url"]["url"].split(",", 1)[1]
                image = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
                content.append({"type": "image", "image": image})

        inputs = self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        out = self.model.generate(**inputs, max_new_tokens=256, do_sample=False)
        text = self.processor.batch_decode(
            out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )[0]
        return fake_response(text)


@pytest.fixture(scope="module")
def vlm_client():
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    processor = transformers.AutoProcessor.from_pretrained(MODEL_ID)
    model = transformers.AutoModelForImageTextToText.from_pretrained(MODEL_ID)
    model.eval()
    return LocalVLMClient(model, processor)


@pytest.fixture
def red_square_image():
    img = np.full((448, 448, 3), 255, dtype=np.uint8)
    img[100:300, 120:320] = (0, 0, 255)  # BGR red square
    return img


def test_plugin_runs_end_to_end_with_real_model(make_plugin, vlm_client, red_square_image):
    plugin = make_plugin()
    plugin.client = vlm_client

    out = plugin.run(red_square_image, prompt="red square", confidence_threshold=0.3)

    assert out.shape == red_square_image.shape
    assert out.dtype == red_square_image.dtype


def test_real_model_reply_is_parseable_and_in_bounds(make_plugin, vlm_client, red_square_image):
    plugin = make_plugin()
    plugin.client = vlm_client

    detections = plugin.extract_bounding_boxes(red_square_image, "", "red square", 0.0)

    h, w = red_square_image.shape[:2]
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        assert 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h
        assert 0.0 <= det["confidence"] <= 1.0
