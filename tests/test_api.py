import json

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from triage.api import create_app
from triage.data import load_alerts
from triage.embeddings import HashEmbedder
from triage.llm import Completion, ToolCall
from triage.tools import Toolbox


class SearchThenClear:
    """Searches change records once, then clears the alert."""

    model = "scripted"

    def complete(self, messages, tools):
        if any(m.get("role") == "tool" for m in messages):
            args = {"severity": "low", "escalate": False, "reason": "Covered by a change record",
                    "evidence": [], "needs_human": False}
            call = ToolCall(id="c2", name="submit_verdict", arguments=json.dumps(args))
        else:
            call = ToolCall(id="c1", name="search_change_records", arguments=json.dumps({"query": "handover"}))
        message = {"role": "assistant", "tool_calls": [
            {"id": call.id, "type": "function", "function": {"name": call.name, "arguments": call.arguments}}
        ]}
        return Completion(message=message, text=None, tool_calls=[call], input_tokens=10, output_tokens=2)


@pytest.fixture(scope="module")
def client():
    toolbox = Toolbox.load(QdrantClient(":memory:"), HashEmbedder())
    with TestClient(create_app(SearchThenClear(), toolbox)) as c:
        yield c


@pytest.fixture(scope="module")
def alert():
    return next(a for a in load_alerts() if "named_case" in a.label.tags).alert


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model": "scripted"}


def test_triage_returns_verdict_and_trail(client, alert):
    response = client.post("/triage", json=alert.model_dump(mode="json"))
    assert response.status_code == 200
    body = response.json()
    assert body["alert_id"] == alert.id
    assert body["verdict"]["escalate"] is False
    assert [c["name"] for c in body["tool_calls"]] == ["search_change_records", "submit_verdict"]


def test_baseline_setup_skips_lookups(client, alert):
    body = client.post("/triage?setup=baseline", json=alert.model_dump(mode="json")).json()
    assert body["setup"] == "baseline"
    assert body["tool_calls"][0]["ok"] is False  # the scripted model tried a lookup it doesn't have


def test_invalid_alert_is_rejected(client, alert):
    bad = alert.model_dump(mode="json") | {"type": "not_a_type"}
    assert client.post("/triage", json=bad).status_code == 422
    assert client.post("/triage?setup=nope", json=alert.model_dump(mode="json")).status_code == 422
