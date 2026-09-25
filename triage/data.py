"""Load the synthetic dataset from data/."""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from triage.models import Asset, ChangeRecord, Identity, Indicator, LabeledAlert

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

M = TypeVar("M", bound=BaseModel)

FILES = {
    "alerts": "alerts.jsonl",
    "change_records": "change_records.jsonl",
    "assets": "assets.jsonl",
    "identities": "identities.jsonl",
    "threat_intel": "threat_intel.jsonl",
}


def _load(name: str, model: type[M], data_dir: Path) -> list[M]:
    path = data_dir / FILES[name]
    with path.open() as f:
        return [model.model_validate_json(line) for line in f if line.strip()]


def load_alerts(data_dir: Path = DATA_DIR) -> list[LabeledAlert]:
    return _load("alerts", LabeledAlert, data_dir)


def load_change_records(data_dir: Path = DATA_DIR) -> list[ChangeRecord]:
    return _load("change_records", ChangeRecord, data_dir)


def load_assets(data_dir: Path = DATA_DIR) -> list[Asset]:
    return _load("assets", Asset, data_dir)


def load_identities(data_dir: Path = DATA_DIR) -> list[Identity]:
    return _load("identities", Identity, data_dir)


def load_threat_intel(data_dir: Path = DATA_DIR) -> list[Indicator]:
    return _load("threat_intel", Indicator, data_dir)
