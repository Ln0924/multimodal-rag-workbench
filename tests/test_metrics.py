import json

from rag_workbench.evaluation import evaluate_ragas, evaluate_retrieval
from rag_workbench.retrieval import rrf


def test_rrf_counts_each_item_once_per_ranking():
    result = rrf([["a", "a", "b"], ["b", "a"]])
    assert result["a"] == 1 / 61 + 1 / 62
    assert result["a"] == result["b"]


def test_evaluation_reports_actual_rank_metrics(workbench, note, tmp_path):
    workbench.ingest(note)
    dataset = tmp_path / "eval.jsonl"
    dataset.write_text(
        json.dumps(
            {"question": "Atlas A1 保修期", "relevant_sources": ["note.md"], "reference": "24 个月"}
        ),
        encoding="utf-8",
    )
    report = evaluate_retrieval(workbench, dataset)
    assert report["sample_count"] == 1
    assert all(item["mrr"] == 1 for item in report["experiments"])
    assert report["ragas"]["status"] == "not_run"


def test_ragas_without_model_is_not_fabricated(workbench, tmp_path):
    result = evaluate_ragas(workbench, tmp_path / "missing.jsonl")
    assert result["status"] == "not_run"
