"""The four lookup tools the agent can call, and their JSON schemas."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from qdrant_client import QdrantClient

from triage.data import load_assets, load_change_records, load_identities, load_threat_intel
from triage.embeddings import Embedder
from triage.models import Asset, ChangeRecord, Identity, Indicator
from triage.store import COLLECTION, index_change_records, search_change_records


class Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SearchArgs(Args):
    query: str = Field(min_length=1)
    host: str | None = None
    account: str | None = None
    around: datetime | None = None
    window_hours: float = Field(default=24, gt=0, le=168)


class AssetArgs(Args):
    hostname: str


class IdentityArgs(Args):
    name: str


class IndicatorArgs(Args):
    value: str


def _function(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


LOOKUP_TOOLS = [
    _function(
        "search_change_records",
        "Semantic search over approved change tickets (deploys, credential rotations, service-account "
        "handovers, firewall changes, maintenance, access grants). Filter by host and/or account "
        "(records touching either match) and by time: pass the alert timestamp as `around` to find "
        "changes whose window is within `window_hours` of it.",
        {
            "query": {"type": "string", "description": "What the change would be about, in plain words"},
            "host": {"type": "string", "description": "Hostname the change touches"},
            "account": {"type": "string", "description": "User or service account the change names"},
            "around": {"type": "string", "description": "ISO 8601 time, usually the alert timestamp"},
            "window_hours": {"type": "number", "description": "Half-width of the time window (default 24)"},
        },
        ["query"],
    ),
    _function(
        "get_asset",
        "Owner, criticality, environment, role and internal IP for a host.",
        {"hostname": {"type": "string"}},
        ["hostname"],
    ),
    _function(
        "get_identity",
        "Account type (human, service or scanner), team, owner and credential-change history for an account.",
        {"name": {"type": "string"}},
        ["name"],
    ),
    _function(
        "check_indicator",
        "Look up an IP address or domain in the threat-intel list of known-bad indicators.",
        {"value": {"type": "string"}},
        ["value"],
    ),
]


class Toolbox:
    """Answers tool calls from the synthetic dataset and the change-record index."""

    def __init__(self, client: QdrantClient, embedder: Embedder, assets: list[Asset],
                 identities: list[Identity], indicators: list[Indicator]):
        self.client = client
        self.embedder = embedder
        self.assets = {a.hostname: a for a in assets}
        self.identities = {i.name: i for i in identities}
        self.indicators = {i.value.lower(): i for i in indicators}

    @classmethod
    def load(cls, client: QdrantClient, embedder: Embedder) -> Toolbox:
        ensure_index(client, embedder, load_change_records())
        return cls(client, embedder, load_assets(), load_identities(), load_threat_intel())

    def call(self, name: str, arguments: str | dict[str, Any]) -> dict[str, Any]:
        """Run one tool call. Bad arguments come back as an error the model can read."""
        handlers = {
            "search_change_records": (SearchArgs, self.search_change_records),
            "get_asset": (AssetArgs, self.get_asset),
            "get_identity": (IdentityArgs, self.get_identity),
            "check_indicator": (IndicatorArgs, self.check_indicator),
        }
        if name not in handlers:
            return {"error": f"Unknown tool {name!r}"}
        model, handler = handlers[name]
        try:
            raw = json.loads(arguments) if isinstance(arguments, str) else arguments
            args = model.model_validate(raw)
        except (json.JSONDecodeError, ValidationError) as e:
            return {"error": f"Invalid arguments for {name}: {e}"}
        return handler(args)

    def search_change_records(self, args: SearchArgs) -> dict[str, Any]:
        hits = search_change_records(
            self.client, self.embedder, args.query, host=args.host, principal=args.account,
            at=args.around, window=timedelta(hours=args.window_hours),
        )
        return {"results": [
            {**hit.record.model_dump(mode="json"), "score": round(hit.score, 3)} for hit in hits
        ]}

    def get_asset(self, args: AssetArgs) -> dict[str, Any]:
        asset = self.assets.get(args.hostname)
        return asset.model_dump(mode="json") if asset else {"error": f"No asset named {args.hostname!r}"}

    def get_identity(self, args: IdentityArgs) -> dict[str, Any]:
        identity = self.identities.get(args.name)
        return identity.model_dump(mode="json") if identity else {"error": f"No account named {args.name!r}"}

    def check_indicator(self, args: IndicatorArgs) -> dict[str, Any]:
        hit = self.indicators.get(args.value.strip().lower())
        if hit is None:
            return {"value": args.value, "listed": False}
        return {"value": args.value, "listed": True, "threat_type": hit.threat_type, "confidence": hit.confidence}


def ensure_index(client: QdrantClient, embedder: Embedder, records: list[ChangeRecord]) -> None:
    """Build the change-record index unless one of the right size and dimension already exists."""
    if client.collection_exists(COLLECTION):
        info = client.get_collection(COLLECTION)
        if info.points_count == len(records) and info.config.params.vectors.size == embedder.dim:
            return
    index_change_records(client, embedder, records)
