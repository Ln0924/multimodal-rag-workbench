from fastapi.testclient import TestClient

from rag_workbench.api import create_app


def test_upload_query_stream_and_delete(settings):
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/api/documents",
            files={
                "file": (
                    "atlas.md",
                    "# Atlas\nAtlas A1 保修期为 24 个月。".encode(),
                    "text/markdown",
                )
            },
        )
        assert response.status_code == 200
        doc_id = response.json()["document"]["id"]
        answer = client.post("/api/ask", json={"question": "Atlas A1 保修期"})
        assert "24" in answer.json()["answer"]
        stream = client.post("/api/ask/stream", json={"question": "Atlas A1 保修期"})
        assert "event: result" in stream.text
        assert client.get(f"/api/documents/{doc_id}/original").status_code == 200
        assert client.delete(f"/api/documents/{doc_id}").status_code == 200


def test_invalid_kb_and_file_type_are_rejected(settings):
    with TestClient(create_app(settings)) as client:
        assert client.get("/api/documents", params={"kb": "../private"}).status_code == 400
        result = client.post("/api/documents", files={"file": ("x.exe", b"data")})
        assert result.status_code == 400
        assert client.post("/api/ask", json={"question": ""}).status_code == 400
