from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from rag_workbench.demo import generate_demo, project_root
from rag_workbench.evaluation import evaluate_ragas, evaluate_retrieval
from rag_workbench.service import Workbench

app = typer.Typer(help="多模态个人知识库 RAG 工作台")
console = Console()


@app.command()
def ingest(path: Path, kb: str = "default"):
    workbench = Workbench()
    try:
        console.print_json(json.dumps(workbench.ingest(path, kb), ensure_ascii=False))
    finally:
        workbench.close()


@app.command()
def ask(question: str, kb: str = "default", session: str = "default"):
    workbench = Workbench()
    try:
        console.print_json(workbench.ask(question, kb, session).model_dump_json())
    finally:
        workbench.close()


@app.command()
def demo():
    workbench = Workbench()
    try:
        root = project_root()
        paths = generate_demo(root, workbench.settings.data_dir / "demo")
        for path in paths:
            console.print(workbench.ingest(path, kb="demo"))
        result = workbench.ask("Atlas A1 的保修期是多久？", kb="demo")
        console.print(result.answer)
        report = evaluate_retrieval(workbench, root / "examples" / "evaluation.jsonl", "demo")
        output = workbench.settings.data_dir / "evaluation.json"
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        console.print(f"评测结果：{output}")
    finally:
        workbench.close()


@app.command("evaluate")
def evaluate_command(dataset: Path, kb: str = "demo", ragas: bool = False):
    workbench = Workbench()
    try:
        result = evaluate_retrieval(workbench, dataset, kb)
        if ragas:
            result["ragas"] = evaluate_ragas(workbench, dataset, kb)
        console.print_json(json.dumps(result, ensure_ascii=False))
    finally:
        workbench.close()


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000):
    import uvicorn

    uvicorn.run("rag_workbench.api:app", host=host, port=port)


if __name__ == "__main__":
    app()
