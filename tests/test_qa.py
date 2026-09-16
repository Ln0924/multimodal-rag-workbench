from rag_workbench.service import validate_citations


def test_hybrid_retrieval_uses_both_channels(workbench, note):
    workbench.ingest(note)
    result = workbench.search("Atlas A1 保修期")
    assert result
    assert set(result[0].channels) == {"bm25", "vector"}


def test_model_heading_outranks_other_model_cross_reference(workbench, tmp_path):
    path = tmp_path / "models.md"
    path.write_text(
        "## Atlas A1\nAtlas A1 保修期为 24 个月。\n\n"
        "## Atlas A2\nAtlas A2 保修期为 12 个月，不要套用 Atlas A1 的保修期。",
        encoding="utf-8",
    )
    workbench.ingest(path)
    evidence = workbench.search("Atlas A1 的保修期是多久？")
    assert evidence[0].chunk.heading == "Atlas A1"


def test_nonexistent_citation_is_invalid(workbench, note):
    workbench.ingest(note)
    evidence = workbench.search("Atlas A1")
    assert validate_citations("保修期为 24 个月 [E1]", evidence)
    assert not validate_citations("保修期为 24 个月 [E999]", evidence)
    assert not validate_citations("没有引用编号", evidence)


def test_unrelated_question_is_not_matched_only_by_common_words(workbench, note):
    workbench.ingest(note)
    answer = workbench.ask("火星大气的成分是什么？")
    assert answer.insufficient
    assert not answer.evidence


def test_ambiguous_first_question_requests_clarification(workbench):
    answer = workbench.ask("它的保修期呢？")
    assert answer.route == "clarify"
    assert answer.evidence == []


def test_sessions_do_not_share_rewrite_history(workbench, note):
    workbench.ingest(note)
    workbench.ask("Atlas A1 保修期", session="alice")
    assert workbench.ask("它的功率呢？", session="bob").route == "clarify"
    answer = workbench.ask("它的功率呢？", session="alice")
    assert "Atlas A1" in answer.rewritten_query


def test_unsupported_model_answer_falls_back_to_evidence(workbench, note, monkeypatch):
    import rag_workbench.service as module

    class FakeGenerator:
        def __init__(self, settings):
            pass

        def generate(self, question, context):
            return "保修期为 999 年 [E999]"

    workbench.settings.generator = "remote"
    monkeypatch.setattr(module, "Generator", FakeGenerator)
    workbench.ingest(note)
    answer = workbench.ask("Atlas A1 保修期")
    assert answer.mode == "extractive"
    assert "999" not in answer.answer
    assert any(item["stage"] == "fallback" for item in answer.trace)


def test_valid_number_but_unsupported_fact_is_rejected(workbench, note, monkeypatch):
    import rag_workbench.service as module

    class UnsupportedGenerator:
        def __init__(self, settings):
            pass

        def generate(self, question, context):
            return "保修期为 999 年 [E1]"

    workbench.settings.generator = "remote"
    monkeypatch.setattr(module, "Generator", UnsupportedGenerator)
    monkeypatch.setattr(module, "chat", lambda *args, **kwargs: '{"supported":false}')
    workbench.ingest(note)
    answer = workbench.ask("Atlas A1 保修期")
    assert answer.mode == "extractive"
    assert "999" not in answer.answer
    assert any(item.get("model_support_check") is False for item in answer.trace)
