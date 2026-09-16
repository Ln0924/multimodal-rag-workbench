import base64

import httpx
import numpy as np

from rag_workbench.config import Settings
from rag_workbench.providers import Generator, SemanticEmbedding, VisionProvider


def test_vlm_transmits_actual_image_bytes(monkeypatch):
    captured = {}

    def fake_post(url, **kwargs):
        captured.update(kwargs["json"])
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={"choices": [{"message": {"content": "流程包含三步"}}]},
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    settings = Settings(api_key="test-only", vlm_model="test-vision")
    assert VisionProvider(settings).describe(b"actual-image-data") == "流程包含三步"
    uri = captured["messages"][0]["content"][1]["image_url"]["url"]
    assert base64.b64decode(uri.split(",", 1)[1]) == b"actual-image-data"
    assert captured["model"] == "test-vision"


def test_remote_embedding_orders_and_normalizes_vectors(monkeypatch):
    def fake_post(url, **kwargs):
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={
                "data": [
                    {"index": 1, "embedding": [0, 3]},
                    {"index": 0, "embedding": [4, 0]},
                ]
            },
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    settings = Settings(embedding="remote", api_key="test-only")
    vectors = SemanticEmbedding(settings).encode(["first", "second"])
    np.testing.assert_allclose(vectors, [[1, 0], [0, 1]])


def test_langchain_generation_chain_passes_evidence(monkeypatch):
    import rag_workbench.providers as module

    captured = []

    def fake_chat(settings, messages, model=""):
        captured.extend(messages)
        return "保修期为 24 个月 [E1]"

    monkeypatch.setattr(module, "chat", fake_chat)
    result = Generator(Settings()).generate("保修期？", "[E1] 保修期为 24 个月。")
    assert "[E1]" in result
    assert "24 个月" in captured[-1]["content"]
    assert captured[0]["role"] == "system"
