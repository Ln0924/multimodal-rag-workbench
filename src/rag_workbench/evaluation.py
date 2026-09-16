from __future__ import annotations

import json
import time
from pathlib import Path

from rag_workbench.service import Workbench


def evaluate_retrieval(
    workbench: Workbench, dataset: Path, kb: str = "default", top_k: int = 5
) -> dict:
    samples = [
        json.loads(line)
        for line in dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not samples:
        raise ValueError("评测集不能为空")
    experiments = []
    with workbench.lock:
        workbench._snapshot(kb)
    for mode, rerank in (("bm25", False), ("vector", False), ("hybrid", False), ("hybrid", True)):
        details = []
        for sample in samples:
            start = time.perf_counter()
            evidence = workbench.search(
                sample["question"], kb, mode, top_k, rerank, use_cache=False
            )
            expected = set(sample["relevant_sources"])
            sources = [item.chunk.source for item in evidence]
            rank = next((i for i, source in enumerate(sources, 1) if source in expected), 0)
            details.append(
                {
                    "question": sample["question"],
                    "sources": sources,
                    "hit": int(rank > 0),
                    "rr": 1 / rank if rank else 0,
                    "recall": len(set(sources) & expected) / max(len(expected), 1),
                    "elapsed_ms": round((time.perf_counter() - start) * 1000, 2),
                }
            )
        count = max(len(details), 1)
        experiments.append(
            {
                "mode": mode,
                "rerank": rerank,
                "top_k": top_k,
                "hit_rate": sum(item["hit"] for item in details) / count,
                "mrr": sum(item["rr"] for item in details) / count,
                "recall": sum(item["recall"] for item in details) / count,
                "details": details,
            }
        )
    return {
        "epoch": workbench.store.epoch(kb),
        "embedding": workbench.settings.embedding,
        "reranker": workbench.settings.reranker,
        "sample_count": len(samples),
        "label_unit": "source_document",
        "cache": "bypassed",
        "experiments": experiments,
        "ragas": {"status": "not_run", "reason": "需要单独配置评测模型"},
    }


def evaluate_ragas(workbench: Workbench, dataset: Path, kb: str = "default") -> dict:
    settings = workbench.settings
    if not settings.api_key:
        return {"status": "not_run", "reason": "未配置评测模型凭据"}
    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas import EvaluationDataset, evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import AnswerRelevancy, Faithfulness, LLMContextPrecisionWithReference

        samples = [
            json.loads(line)
            for line in dataset.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        rows = []
        for sample in samples:
            answer = workbench.ask(sample["question"], kb, session="evaluation")
            rows.append(
                {
                    "user_input": sample["question"],
                    "response": answer.answer,
                    "retrieved_contexts": [item.chunk.text for item in answer.evidence],
                    "reference": sample["reference"],
                }
            )
        llm = LangchainLLMWrapper(
            ChatOpenAI(
                model=settings.chat_model, api_key=settings.api_key, base_url=settings.base_url
            )
        )
        embeddings = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(
                model=settings.embedding_model, api_key=settings.api_key, base_url=settings.base_url
            )
        )
        results = evaluate(
            EvaluationDataset.from_list(rows),
            metrics=[Faithfulness(), AnswerRelevancy(), LLMContextPrecisionWithReference()],
            llm=llm,
            embeddings=embeddings,
        )
        return {
            "status": "executed",
            "rows": json.loads(results.to_pandas().to_json(orient="records", force_ascii=False)),
        }
    except Exception as exc:
        return {"status": "failed", "error_type": type(exc).__name__, "message": str(exc)}
