from pathlib import Path
from uuid import uuid4

import pytest

from rag_workbench.config import Settings
from rag_workbench.service import Workbench


@pytest.fixture
def tmp_path() -> Path:
    path = Path.cwd() / "runtime" / "tests" / uuid4().hex
    path.mkdir(parents=True)
    return path


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path / "data", ocr_enabled=False)


@pytest.fixture
def workbench(settings):
    instance = Workbench(settings)
    yield instance
    instance.close()


@pytest.fixture
def note(tmp_path):
    path = tmp_path / "note.md"
    path.write_text("# Atlas A1\n\nAtlas A1 保修期为 24 个月。额定功率为 65W。", encoding="utf-8")
    return path
