import pytest

from rag_workbench.service import Workbench


def test_duplicate_import_skips_reindex(workbench, note):
    first = workbench.ingest(note)
    second = workbench.ingest(note)
    assert first["status"] == "created"
    assert second["status"] == "unchanged"
    assert first["epoch"] == second["epoch"]


def test_modification_replaces_old_version(workbench, note):
    workbench.ingest(note)
    old_version = workbench.documents()[0]["version"]
    note.write_text("# Atlas A1\n\nAtlas A1 保修期为 36 个月。", encoding="utf-8")
    result = workbench.ingest(note)
    evidence = workbench.search("Atlas A1 保修期")
    assert result["status"] == "updated"
    assert all(item.chunk.version != old_version for item in evidence)
    assert any("36" in item.chunk.text for item in evidence)
    assert not any("24" in item.chunk.text for item in evidence)


def test_deletion_removes_evidence_and_invalidates_answer(workbench, note):
    result = workbench.ingest(note)
    workbench.ask("Atlas A1 保修期")
    workbench.delete(result["document"]["id"])
    answer = workbench.ask("Atlas A1 保修期")
    assert answer.insufficient
    assert answer.evidence == []
    assert answer.cache_hit is False


def test_failed_parse_keeps_previous_version(workbench, note):
    first = workbench.ingest(note)
    note.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        workbench.ingest(note)
    assert workbench.store.epoch("default") == first["epoch"]
    assert "24" in workbench.search("Atlas A1 保修期")[0].chunk.text


def test_knowledge_bases_are_isolated(workbench, note, tmp_path):
    other = tmp_path / "private.md"
    other.write_text("# Boreal\nBoreal 设备保修期为 6 个月。", encoding="utf-8")
    workbench.ingest(note, "public")
    workbench.ingest(other, "private")
    public = workbench.search("保修期", "public")
    assert all("Boreal" not in item.chunk.text for item in public)


def test_restart_recovers_active_documents(settings, note):
    first = Workbench(settings)
    first.ingest(note)
    collection_name = first.snapshots["default"].collection.name
    first.close()
    second = Workbench(settings)
    try:
        assert "24" in second.search("Atlas A1 保修期")[0].chunk.text
        assert second.snapshots["default"].collection.name == collection_name
    finally:
        second.close()


def test_new_version_invalidates_answer_cache(workbench, note):
    workbench.ingest(note)
    assert workbench.ask("Atlas A1 保修期").cache_hit is False
    assert workbench.ask("Atlas A1 保修期").cache_hit is True
    note.write_text("# Atlas A1\nAtlas A1 保修期为 36 个月。", encoding="utf-8")
    workbench.ingest(note)
    answer = workbench.ask("Atlas A1 保修期")
    assert answer.cache_hit is False
    assert "36" in answer.answer


def test_index_build_failure_does_not_activate_new_content(workbench, note, monkeypatch):
    import rag_workbench.service as module

    workbench.ingest(note)
    epoch = workbench.store.epoch("default")
    note.write_text("# Atlas A1\nAtlas A1 保修期为 36 个月。", encoding="utf-8")

    def failed_snapshot(*args, **kwargs):
        raise RuntimeError("simulated index build failure")

    monkeypatch.setattr(module, "Snapshot", failed_snapshot)
    with pytest.raises(RuntimeError):
        workbench.ingest(note)
    assert workbench.store.epoch("default") == epoch
    assert "24" in workbench.search("Atlas A1 保修期")[0].chunk.text
