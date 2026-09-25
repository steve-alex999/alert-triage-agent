import json
from itertools import count

import pytest
from qdrant_client import QdrantClient

from triage.agent import MAX_LOOKUPS, triage, user_prompt
from triage.data import load_alerts
from triage.embeddings import HashEmbedder
from triage.llm import Completion, ToolCall
from triage.tools import Toolbox

VALID = {"severity": "low", "escalate": False, "reason": "Explained by the handover",
         "evidence": ["CHG-1"], "needs_human": False}
_ids = count()


def turn(*calls: tuple[str, dict]) -> Completion:
    tool_calls = [ToolCall(id=f"call-{next(_ids)}", name=n, arguments=json.dumps(a)) for n, a in calls]
    message = {"role": "assistant", "tool_calls": [
        {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}} for c in tool_calls
    ]}
    return Completion(message=message, text=None, tool_calls=tool_calls, input_tokens=100, output_tokens=10)


def text_turn(text: str) -> Completion:
    return Completion(message={"role": "assistant", "content": text}, text=text, tool_calls=[],
                      input_tokens=100, output_tokens=10)


class ScriptedLLM:
    model = "scripted"

    def __init__(self, *turns: Completion):
        self.turns = list(turns)
        self.requests: list[tuple[list, list]] = []

    def complete(self, messages, tools):
        self.requests.append(([dict(m) for m in messages], tools))
        return self.turns.pop(0)


@pytest.fixture(scope="module")
def toolbox():
    return Toolbox.load(QdrantClient(":memory:"), HashEmbedder())


@pytest.fixture(scope="module")
def named():
    return next(a for a in load_alerts() if "named_case" in a.label.tags)


def test_lookups_then_verdict(toolbox, named):
    alert = named.alert
    llm = ScriptedLLM(
        turn(("search_change_records", {"query": "handover", "host": alert.host,
                                        "account": alert.principal, "around": alert.timestamp.isoformat()})),
        turn(("submit_verdict", VALID)),
    )
    run = triage(alert, llm, toolbox)

    assert run.fallback is None
    assert run.verdict.escalate is False
    assert [c.name for c in run.tool_calls] == ["search_change_records", "submit_verdict"]
    assert (run.llm_calls, run.input_tokens, run.output_tokens) == (2, 200, 20)
    # The search result reached the model on its second call.
    tool_message = llm.requests[1][0][-1]
    assert tool_message["role"] == "tool"
    assert named.label.change_record_id in tool_message["content"]


def test_one_retry_on_invalid_verdict(toolbox, named):
    llm = ScriptedLLM(turn(("submit_verdict", {**VALID, "severity": "critical"})), turn(("submit_verdict", VALID)))
    run = triage(named.alert, llm, toolbox)
    assert run.fallback is None
    assert run.verdict.severity == "low"
    assert [c.ok for c in run.tool_calls] == [False, True]


def test_second_invalid_verdict_falls_back_to_human(toolbox, named):
    bad = {k: v for k, v in VALID.items() if k != "escalate"}
    llm = ScriptedLLM(turn(("submit_verdict", bad)), turn(("submit_verdict", bad)))
    run = triage(named.alert, llm, toolbox)
    assert run.fallback == "invalid verdict twice"
    assert run.verdict.needs_human and run.verdict.escalate


def test_text_without_verdict_gets_one_nudge(toolbox, named):
    run = triage(named.alert, ScriptedLLM(text_turn("Looks fine."), turn(("submit_verdict", VALID))), toolbox)
    assert run.fallback is None
    run = triage(named.alert, ScriptedLLM(text_turn("Looks fine."), text_turn("Still fine.")), toolbox)
    assert run.fallback == "no verdict submitted"


def test_lookup_budget(toolbox, named):
    lookups = [turn(("get_asset", {"hostname": named.alert.host})) for _ in range(MAX_LOOKUPS + 1)]
    run = triage(named.alert, ScriptedLLM(*lookups, turn(("submit_verdict", VALID))), toolbox)
    oks = [c.ok for c in run.tool_calls if c.name == "get_asset"]
    assert oks == [True] * MAX_LOOKUPS + [False]
    assert run.fallback is None


def test_baseline_offers_only_submit_verdict(named):
    llm = ScriptedLLM(turn(("submit_verdict", VALID)))
    triage(named.alert, llm, None, setup="baseline")
    tools = llm.requests[0][1]
    assert [t["function"]["name"] for t in tools] == ["submit_verdict"]


def test_guard_escapes_tags_inside_the_alert():
    item = next(a for a in load_alerts() if "</alert>" in a.alert.raw_message)
    prompt = user_prompt(item.alert, "guard")
    assert prompt.startswith("<untrusted_alert>") and prompt.endswith("</untrusted_alert>")
    assert prompt.count("<") == 2  # only the wrapper's own tags


def test_toolbox_answers(toolbox):
    assert toolbox.call("get_asset", '{"hostname": "nope"}') == {"error": "No asset named 'nope'"}
    assert "error" in toolbox.call("get_asset", '{"host": "db-prod-01"}')
    assert "error" in toolbox.call("delete_everything", "{}")
    bad_ip = next(iter(toolbox.indicators))
    assert toolbox.call("check_indicator", {"value": bad_ip.upper()})["listed"] is True
    assert toolbox.call("check_indicator", {"value": "10.0.0.1"})["listed"] is False
