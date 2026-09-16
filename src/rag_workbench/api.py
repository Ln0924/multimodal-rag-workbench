from __future__ import annotations

import json
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel

from rag_workbench.config import Settings
from rag_workbench.demo import generate_demo, project_root
from rag_workbench.evaluation import evaluate_ragas, evaluate_retrieval
from rag_workbench.service import Workbench


class AskRequest(BaseModel):
    question: str
    kb: str = "default"
    session: str = "default"


def create_app(settings: Settings | None = None) -> FastAPI:
    workbench = Workbench(settings)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        workbench.close()

    app = FastAPI(title="多模态 RAG 工作台", lifespan=lifespan)
    app.state.workbench = workbench

    @app.exception_handler(ValueError)
    async def invalid_request(_request, exc):
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "embedding": workbench.settings.embedding,
            "generator": workbench.settings.generator,
            "vlm": "configured" if workbench.settings.vlm_model else "not_configured",
        }

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (Path(__file__).parent / "web" / "index.html").read_text(encoding="utf-8")

    @app.get("/api/documents")
    def documents(kb: str = "default"):
        return {"documents": workbench.documents(kb), "epoch": workbench.store.epoch(kb)}

    @app.post("/api/demo")
    def demo():
        paths = generate_demo(project_root(), workbench.settings.data_dir / "demo")
        results = [workbench.ingest(path, "demo") for path in paths]
        return {"kb": "demo", "results": results}

    @app.post("/api/evaluate")
    def evaluate(kb: str = "demo", ragas: bool = False):
        dataset = project_root() / "examples" / "evaluation.jsonl"
        with workbench.lock:
            result = evaluate_retrieval(workbench, dataset, kb)
            if ragas:
                result["ragas"] = evaluate_ragas(workbench, dataset, kb)
            return result

    @app.post("/api/documents")
    def upload(file: Annotated[UploadFile, File()], kb: str = "default"):
        workbench.validate_kb(kb)
        source = re.split(r"[/\\]", file.filename or "document")[-1]
        suffix = Path(source).suffix.lower()
        if suffix not in {".pdf", ".md", ".png", ".jpg", ".jpeg"}:
            raise HTTPException(400, "不支持该文件类型")
        data = file.file.read(workbench.settings.max_file_bytes + 1)
        if len(data) > workbench.settings.max_file_bytes:
            raise HTTPException(413, "文件超过大小限制")
        temp = workbench.settings.data_dir / "uploads" / (uuid4().hex + suffix)
        temp.parent.mkdir(parents=True, exist_ok=True)
        temp.write_bytes(data)
        try:
            return workbench.ingest(temp, kb, source_name=source)
        finally:
            temp.unlink(missing_ok=True)

    @app.delete("/api/documents/{document_id}")
    def delete(document_id: str, kb: str = "default"):
        try:
            return workbench.delete(document_id, kb)
        except KeyError as exc:
            raise HTTPException(404, "文档不存在") from exc

    @app.get("/api/documents/{document_id}/original")
    def original(document_id: str, kb: str = "default"):
        workbench.validate_kb(kb)
        with workbench.lock:
            document = next(
                (doc for doc in workbench.store.documents(kb) if doc["id"] == document_id), None
            )
            if not document:
                raise HTTPException(404, "文档不存在")
            return FileResponse(
                document["path"], filename=document["source"], content_disposition_type="inline"
            )

    @app.get("/api/assets/{image_id}")
    def asset(image_id: str):
        if not re.fullmatch(r"[a-f0-9]{24}", image_id):
            raise HTTPException(400, "无效图片 ID")
        path = workbench.settings.data_dir / "assets" / f"{image_id}.png"
        if not path.is_file():
            raise HTTPException(404, "图片不存在")
        return FileResponse(path)

    @app.post("/api/ask")
    def ask(request: AskRequest):
        return workbench.ask(request.question, request.kb, request.session)

    @app.post("/api/ask/stream")
    def ask_stream(request: AskRequest):
        def events():
            yield 'event: status\ndata: {"stage":"running"}\n\n'
            try:
                answer = workbench.ask(request.question, request.kb, request.session)
                # 引用核查结束后才释放答案，避免先展示未经核查的模型结论。
                for start in range(0, len(answer.answer), 80):
                    data = json.dumps(
                        {"text": answer.answer[start : start + 80]}, ensure_ascii=False
                    )
                    yield f"event: text\ndata: {data}\n\n"
                yield f"event: result\ndata: {answer.model_dump_json()}\n\n"
            except Exception as exc:
                yield (
                    "event: error\ndata: "
                    + json.dumps({"detail": str(exc)}, ensure_ascii=False)
                    + "\n\n"
                )

        return StreamingResponse(events(), media_type="text/event-stream")

    return app


app = create_app()
