import json
import threading

import pytest
from qdrant_client import QdrantClient

from triage.data import load_alerts
from triage.embeddings import HashEmbedder
from triage.evaluate import StopRun, load_rows, pick_alerts, results_table, run, score, write_readme
from triage.llm import Completion, ToolCall
from triage.tools import Toolbox


class AlwaysEscalate:
    """Looks up the host once, then escalates everything."""

    model = "always-escalate"

    def __init__(self, fail_after: int | None = None):
        self.calls = 0
        self.fail_after = fail_after
        self._lock = threading.Lock()

    def complete(self, messages, tools):
        with self._lock:
            self.calls += 1
            if self.fail_after is not None and self.calls > self.fail_after:
                raise ConnectionError("quota exhausted")
        names = [t["function"]["name"] for t in tools]
        already_looked = any(m.get("role") == "tool" for m in messages)
        if "search_change_records" in names and not already_looked:
            call = ToolCall(id="c1", name="search_change_records", arguments=json.dumps({"query": "change"}))
        else:
            verdict = {"severity": "high", "escalate": True, "reason": "Escalate everything",
                       "evidence": [], "needs_human": False}
            call = ToolCall(id="c2", name="submit_verdict", arguments=json.dumps(verdict))
        message = {"role": "assistant", "tool_calls": [
            {"id": call.id, "type": "function", "function": {"name": call.name, "arguments": call.arguments}}
        ]}
        return Completion(message=message, text=None, tool_calls=[call], input_tokens=50, output_tokens=5)


@pytest.fixture(scope="module")
def alerts():
    return load_alerts()


@pytest.fixture(scope="module")
def toolbox():
    return Toolbox.load(QdrantClient(":memory:"), HashEmbedder())


def quiet(_):
    pass


def test_full_run_scores_an_always_escalate_model(alerts, toolbox, tmp_path):
    path = tmp_path / "run.jsonl"
    run(alerts, ["baseline", "agent"], 1, AlwaysEscalate(), toolbox, path, workers=4, progress=quiet)
    rows = load_rows(path)
    assert len(rows) == 240
    labels = {a.alert.id: a for a in alerts}

    agent = score([r for r in rows if r.setup == "agent"], labels)
    assert agent["Escalation recall"] == 1.0
    assert agent["Escalation precision"] == pytest.approx(60 / 120)
    assert agent["False escalation rate (benign)"] == 1.0
    assert agent["Named handover case handled"] == 0.0
    assert agent["Change-record lookup rate"] == 1.0
    assert agent["Injection resistance"] == 1.0
    assert agent["Valid verdicts"] == 1.0
    assert agent["Tokens per alert"] == 110

    baseline = score([r for r in rows if r.setup == "baseline"], labels)
    assert baseline["Change-record lookup rate"] is None
    assert baseline["Tokens per alert"] == 55

    table = results_table(rows, labels)
    assert "| Escalation recall | 100% | 100% |" in table
    assert "| Change-record lookup rate | – | 100% |" in table


def test_rerun_resumes_without_repeating_work(alerts, toolbox, tmp_path):
    path = tmp_path / "run.jsonl"
    sample = pick_alerts(alerts, 5)
    assert run(sample, ["agent"], 2, AlwaysEscalate(), toolbox, path, progress=quiet) == 10
    llm = AlwaysEscalate()
    assert run(sample, ["agent"], 2, llm, toolbox, path, progress=quiet) == 0
    assert llm.calls == 0


def test_transient_errors_stop_the_run_and_keep_finished_rows(alerts, toolbox, tmp_path):
    path = tmp_path / "run.jsonl"
    sample = pick_alerts(alerts, 5)
    with pytest.raises(StopRun):
        run(sample, ["baseline"], 1, AlwaysEscalate(fail_after=2), toolbox, path,
            is_transient=lambda e: isinstance(e, ConnectionError), progress=quiet)
    assert len(load_rows(path)) == 2
    assert run(sample, ["baseline"], 1, AlwaysEscalate(), toolbox, path, progress=quiet) == 3


def test_other_errors_are_recorded(alerts, toolbox, tmp_path):
    path = tmp_path / "run.jsonl"
    run(pick_alerts(alerts, 3), ["baseline"], 1, AlwaysEscalate(fail_after=0), toolbox, path, progress=quiet)
    rows = load_rows(path)
    assert all(r.error and r.result is None for r in rows)
    labels = {a.alert.id: a for a in alerts}
    assert score(rows, labels)["Valid verdicts"] == 0.0
    assert score(rows, labels)["Errors"] == 3


def test_write_readme_replaces_its_block(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("# Project\n")
    write_readme("table one", readme)
    write_readme("table two", readme)
    text = readme.read_text()
    assert text.count("## Results") == 1
    assert "table two" in text and "table one" not in text
