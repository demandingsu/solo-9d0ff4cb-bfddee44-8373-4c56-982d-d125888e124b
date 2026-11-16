"""跨测试模块共享的夹具。"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as c:
        c.data_dir = tmp_path / "data"
        yield c
