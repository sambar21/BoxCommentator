import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from bench.stub_llm import StubLLM
from src.api import routes
from src.core.orchestrator import CommentaryOrchestrator


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(
        routes, "CommentaryOrchestrator",
        lambda: CommentaryOrchestrator(llm_client=StubLLM(), use_rag=False),
    )
    routes.reset_sessions()
    app = FastAPI()
    app.include_router(routes.router)
    yield TestClient(app)
    routes.reset_sessions()


def _punch(fight_id, **over):
    body = dict(attacker=1, punch_type="jab", target="head", outcome="landed",
                damage=10, fight_id=fight_id, fighter1_name="Alvarez",
                fighter2_name="Garcia", round_num=1)
    body.update(over)
    return body


def test_fights_have_isolated_state(client):
    for _ in range(5):
        assert client.post("/internal/punch", json=_punch("a")).status_code == 200
    client.post("/internal/punch", json=_punch("b"))

    a = routes._sessions["a"].orchestrator
    b = routes._sessions["b"].orchestrator
    assert a is not b
    assert a.stats["total_punches"] == 5
    assert b.stats["total_punches"] == 1
    assert client.get("/health").json()["active_fights"] == 2


def test_knockdown_over_http_gets_called(client):
    for _ in range(3):
        client.post("/internal/punch", json=_punch("kd"))
    r = client.post("/internal/punch", json=_punch("kd", punch_type="hook", damage=55, knockdown=True))
    assert r.status_code == 200
    assert "KNOCKDOWN" in r.json()["commentary"]
    assert r.json()["fight_id"] == "kd"


def test_concurrent_fights_do_not_corrupt_each_other(client):
    def run(fight_id):
        for _ in range(20):
            assert client.post("/internal/punch", json=_punch(fight_id)).status_code == 200
        return fight_id

    ids = [f"f{i}" for i in range(8)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(run, ids))

    for fid in ids:
        assert routes._sessions[fid].orchestrator.stats["total_punches"] == 20


def test_missing_fight_id_still_works(client):
    body = _punch("x")
    del body["fight_id"]
    assert client.post("/internal/punch", json=body).json()["fight_id"] == "default"
