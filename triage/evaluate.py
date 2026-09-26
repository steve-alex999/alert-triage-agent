"""Evaluation harness: run the setups over the labeled alerts and score them.

    python eval.py --limit 6                                  # smoke test
    python eval.py --model gemini-3.8-flash --repeats 1       # full run
    python eval.py --report results/a.jsonl results/b.jsonl --readme   # score finished runs

Every triage is appended to results/<run>.jsonl as it finishes, so rerunning the same
command resumes an interrupted run. Rate limits and outages stop the run cleanly.

Scoring: a verdict "flags" an alert when it escalates or asks for a human. Threats and
ambiguous alerts should be flagged; benign alerts should not.
"""

from __future__ import annotations

import argparse
import random
import re
import statistics
import threading
from collections import defaultdict
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from triage.agent import Setup, TriageResult, triage
from triage.llm import LLM
from triage.models import LabeledAlert, Verdict
from triage.tools import Toolbox

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
SETUPS: tuple[Setup, ...] = ("baseline", "agent", "guard")


class Row(BaseModel):
    setup: Setup
    repeat: int
    alert_id: str
    result: TriageResult | None = None
    error: str | None = None


class StopRun(Exception):
    """A rate limit or outage: stop now and resume later."""


def load_rows(path: Path) -> list[Row]:
    if not path.exists():
        return []
    with path.open() as f:
        return [Row.model_validate_json(line) for line in f if line.strip()]


def pick_alerts(alerts: list[LabeledAlert], limit: int | None) -> list[LabeledAlert]:
    """All alerts, or a fixed pseudo-random sample for smoke tests."""
    if not limit:
        return alerts
    return sorted(random.Random(0).sample(alerts, limit), key=lambda a: a.alert.timestamp)


class LockedToolbox:
    """Serialises tool calls: embedded Qdrant and the embedder are shared across threads."""

    def __init__(self, toolbox: Toolbox):
        self._toolbox = toolbox
        self._lock = threading.Lock()

    def call(self, name: str, arguments: Any) -> dict[str, Any]:
        with self._lock:
            return self._toolbox.call(name, arguments)


def run(
    alerts: list[LabeledAlert],
    setups: Iterable[Setup],
    repeats: int,
    llm: LLM,
    toolbox: Toolbox | None,
    path: Path,
    *,
    workers: int = 1,
    is_transient: Callable[[Exception], bool] = lambda e: False,
    progress: Callable[[str], None] = print,
) -> int:
    """Triage every (setup, repeat, alert) not already in `path`. Returns how many ran."""
    path.parent.mkdir(parents=True, exist_ok=True)
    done = {(r.setup, r.repeat, r.alert_id) for r in load_rows(path)}
    jobs = [
        (setup, k, item) for k in range(repeats) for setup in setups for item in alerts
        if (setup, k, item.alert.id) not in done
    ]
    if not jobs:
        return 0
    tools = LockedToolbox(toolbox) if toolbox else None
    write_lock, stop = threading.Lock(), threading.Event()
    finished = 0

    def work(setup: Setup, k: int, item: LabeledAlert) -> Row | None:
        if stop.is_set():
            return None
        try:
            result = triage(item.alert, llm, None if setup == "baseline" else tools, setup)
        except Exception as e:
            if is_transient(e):
                stop.set()
                raise StopRun(f"{type(e).__name__}: {e}") from e
            return Row(setup=setup, repeat=k, alert_id=item.alert.id, error=f"{type(e).__name__}: {e}")
        return Row(setup=setup, repeat=k, alert_id=item.alert.id, result=result)

    labels = {a.alert.id: a.label for a in alerts}
    stopped: StopRun | None = None
    with ThreadPoolExecutor(max_workers=workers) as pool, path.open("a") as out:
        futures = [pool.submit(work, *job) for job in jobs]
        for future in as_completed(futures):
            try:
                row = future.result()
            except StopRun as e:
                stopped = stopped or e
                continue
            if row is None:
                continue
            with write_lock:
                out.write(row.model_dump_json() + "\n")
                out.flush()
                finished += 1
            if row.result:
                label = labels[row.alert_id]
                hit = flags(row.result.verdict) == label.escalate
                progress(f"[{len(done) + finished}/{len(done) + len(jobs)}] {row.setup:<8} {row.alert_id} "
                         f"{'ok   ' if hit else 'MISS '} {label.category:<9} {row.result.latency_s:>6.1f}s")
            else:
                progress(f"[{len(done) + finished}/{len(done) + len(jobs)}] {row.setup:<8} {row.alert_id} "
                         f"ERROR {row.error}")
    if stopped:
        raise stopped
    return finished


# ---- scoring -------------------------------------------------------------------------

def flags(verdict: Verdict) -> bool:
    return verdict.escalate or verdict.needs_human


def _rate(hits: list[bool]) -> float | None:
    return sum(hits) / len(hits) if hits else None


METRICS: list[tuple[str, str]] = [
    # key, format: "pct" (rate) or "num"
    ("Escalation recall", "pct"),
    ("Escalation precision", "pct"),
    ("False escalation rate (benign)", "pct"),
    ("Named handover case handled", "pct"),
    ("Severity accuracy", "pct"),
    ("Change-record lookup rate", "pct"),
    ("Injection resistance", "pct"),
    ("Valid verdicts", "pct"),
    ("Median latency (s)", "num"),
    ("Tokens per alert", "num"),
    ("Errors", "int"),
]


def score(rows: list[Row], labels: dict[str, LabeledAlert]) -> dict[str, float | None]:
    """Metrics for one setup and one repeat."""
    ok = [(r.result, labels[r.alert_id].label) for r in rows if r.result]
    flagged = [(flags(res.verdict), label) for res, label in ok]
    with_record = [(res, label) for res, label in ok if label.change_record_id]
    looked_up = (
        None if rows and rows[0].setup == "baseline" else
        _rate([any(c.name == "search_change_records" for c in res.tool_calls) for res, _ in with_record])
    )
    return {
        "Escalation recall": _rate([f for f, label in flagged if label.escalate]),
        "Escalation precision": _rate([label.escalate for f, label in flagged if f]),
        "False escalation rate (benign)": _rate([f for f, label in flagged if label.category == "benign"]),
        "Named handover case handled": _rate([not f for f, label in flagged if "named_case" in label.tags]),
        "Severity accuracy": _rate([res.verdict.severity == label.severity for res, label in ok]),
        "Change-record lookup rate": looked_up,
        "Injection resistance": _rate([f for f, label in flagged if "injection" in label.tags]),
        "Valid verdicts": _rate([bool(r.result) and r.result.fallback is None for r in rows]),
        "Median latency (s)": statistics.median(res.latency_s for res, _ in ok) if ok else None,
        "Tokens per alert": statistics.mean(res.input_tokens + res.output_tokens for res, _ in ok) if ok else None,
        "Errors": float(sum(r.error is not None for r in rows)),
    }


def summarise(rows: list[Row], labels: dict[str, LabeledAlert]) -> dict[str, dict[str, list[float]]]:
    """setup -> metric -> one value per repeat (repeats with no value are skipped)."""
    groups: dict[tuple[str, int], list[Row]] = defaultdict(list)
    for row in rows:
        groups[(row.setup, row.repeat)].append(row)
    summary: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for (setup, _), group in sorted(groups.items()):
        for metric, value in score(group, labels).items():
            if value is not None:
                summary[setup][metric].append(value)
    return summary


def _fmt(values: list[float], kind: str) -> str:
    if not values:
        return "–"
    if kind == "pct":
        def show(v: float) -> str:
            return f"{v * 100:.0f}%"
    elif kind == "int":
        def show(v: float) -> str:
            return f"{v:.0f}"
    else:
        def show(v: float) -> str:
            return f"{v:,.0f}" if v >= 100 else f"{v:.1f}"
    mean = show(statistics.mean(values))
    if len(values) == 1 or min(values) == max(values):
        return mean
    return f"{mean} ({show(min(values))}–{show(max(values))})"


def results_table(rows: list[Row], labels: dict[str, LabeledAlert]) -> str:
    summary = summarise(rows, labels)
    setups = [s for s in SETUPS if s in summary]
    repeats = len({r.repeat for r in rows})
    models = sorted({r.result.model for r in rows if r.result})
    alerts = len({r.alert_id for r in rows})
    lines = [
        f"Model: {', '.join(models)} · {alerts} alerts · {repeats} repeat{'s' if repeats != 1 else ''}"
        + (" · mean (range) across repeats" if repeats > 1 else ""),
        "",
        "| Metric | " + " | ".join(s.capitalize() for s in setups) + " |",
        "| --- | " + " | ".join("---" for _ in setups) + " |",
    ]
    for metric, kind in METRICS:
        lines.append(f"| {metric} | " + " | ".join(_fmt(summary[s].get(metric, []), kind) for s in setups) + " |")
    return "\n".join(lines)


def write_readme(table: str, readme: Path = ROOT / "README.md") -> None:
    start, end = "<!-- results:start -->", "<!-- results:end -->"
    text = readme.read_text()
    block = f"{start}\n{table}\n{end}"
    if start in text:
        text = re.sub(re.escape(start) + r".*?" + re.escape(end), lambda _: block, text, flags=re.S)
    else:
        text = text.rstrip() + f"\n\n## Results\n\n{block}\n"
    readme.write_text(text)


# ---- CLI -----------------------------------------------------------------------------

def _is_transient(e: Exception) -> bool:
    import openai

    return isinstance(e, (openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError))


def main() -> None:
    from triage.data import load_alerts
    from triage.embeddings import get_embedder
    from triage.llm import get_llm
    from triage.store import get_client

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", help="Overrides TRIAGE_MODEL")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--setups", nargs="+", choices=SETUPS, default=list(SETUPS))
    parser.add_argument("--limit", type=int, help="Triage a fixed sample of this many alerts")
    parser.add_argument("--workers", type=int, default=1, help="Parallel triages (mind free-tier rate limits)")
    parser.add_argument("--out", type=Path, help="Results file (default: results/<model>-r<repeats>.jsonl)")
    parser.add_argument("--report", type=Path, nargs="+", help="Only score existing results files")
    parser.add_argument("--readme", action="store_true", help="Write the results table into README.md")
    args = parser.parse_args()

    alerts = load_alerts()
    labels = {a.alert.id: a for a in alerts}

    if args.report:
        paths = args.report
    else:
        llm = get_llm(model=args.model)
        sample = pick_alerts(alerts, args.limit)
        name = f"{llm.model}-r{args.repeats}" + (f"-n{args.limit}" if args.limit else "")
        path = args.out or RESULTS_DIR / f"{name}.jsonl"
        paths = [path]
        toolbox = Toolbox.load(get_client(), get_embedder()) if set(args.setups) - {"baseline"} else None
        print(f"Running {len(args.setups)} setups × {len(sample)} alerts × {args.repeats} repeats → {path}")
        try:
            run(sample, args.setups, args.repeats, llm, toolbox, path, workers=args.workers,
                is_transient=_is_transient)
        except StopRun as e:
            print(f"\nStopped: {e}\nResults so far are saved; rerun the same command to resume.")
            return
        except KeyboardInterrupt:
            print("\nInterrupted; rerun the same command to resume.")
            return

    table = "\n\n".join(results_table(load_rows(p), labels) for p in paths)
    print("\n" + table)
    if args.readme:
        write_readme(table)
        print("\nWrote the table to README.md")


if __name__ == "__main__":
    main()
