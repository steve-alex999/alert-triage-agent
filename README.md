# Alert Triage Agent

A tool-calling LLM agent that triages security alerts, with an evaluation harness that
measures it against a no-tools baseline. All data is synthetic.

**Status:** data, change-record search and the agent loop are done. `eval.py`, the API
and Docker Compose come next.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
python -m triage.synthetic   # regenerate data/ (the committed files are identical)
python -m triage.store       # index change records in Qdrant, then search for the named case
pytest                       # offline; the agent tests use a scripted model

export GEMINI_API_KEY=...    # free key from https://aistudio.google.com/app/apikey
python -m triage.agent                        # triage the named case
python -m triage.agent --setup baseline       # same alert, no lookups
python -m triage.agent --alert ALR-53006 --setup guard
```

The model is reached through an OpenAI-compatible adapter. `TRIAGE_PROVIDER` picks
`gemini` (default) or `ollama`, and `TRIAGE_MODEL` picks the model (default
`gemini-3.5-flash-lite`).

## How the agent works

The model gets the alert and five tools: `search_change_records`, `get_asset`,
`get_identity`, `check_indicator` and `submit_verdict`. It may make up to 5 lookups, then
must call `submit_verdict`, whose arguments are validated against the `Verdict` model. An
invalid verdict gets one retry; a second failure, or no verdict at all, returns a fallback
verdict that escalates with `needs_human` set. The `baseline` setup offers only
`submit_verdict`, and the `guard` setup wraps the alert in `<untrusted_alert>` tags and
tells the model to treat its contents as data.

`triage.store` uses Qdrant's embedded mode (`.qdrant/`) unless `QDRANT_URL` points at a
server. Embeddings come from `BAAI/bge-small-en-v1.5` via fastembed, run locally; set
`EMBEDDING_PROVIDER=hash` to skip the model download (the tests do this).

## Data

`python -m triage.synthetic` writes these files from a fixed seed:

| File | Rows | Contents |
| --- | --- | --- |
| `alerts.jsonl` | 120 | Alerts with ground-truth labels: 40 threat, 60 benign, 20 ambiguous |
| `change_records.jsonl` | 300 | Change tickets: deploys, rotations, handovers, firewall changes, maintenance, access grants |
| `assets.jsonl` | 50 | Hosts with owner, criticality and environment |
| `identities.jsonl` | 80 | People, service accounts and scanners, with credential history |
| `threat_intel.jsonl` | 200 | Known-bad IPs and domains |

Many benign and threat alerts share a message template, so the alert text alone can't
separate them. Only the lookups can. Ten threat alerts carry a prompt-injection string in
an attacker-controlled field. One benign alert is tagged `named_case`: a service-account
credential handover that looks like credential misuse unless you find its change record.

Labeling policy: threats escalate, with severity from the asset's criticality. Benign
alerts are low severity and don't escalate. Ambiguous alerts escalate at medium, and a
verdict that asks for a human also counts as correct.

External IPs come from the RFC 5737 documentation ranges, and domains use the reserved
`.test` and `.example` TLDs.
