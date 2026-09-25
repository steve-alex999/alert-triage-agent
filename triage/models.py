"""Pydantic models for alerts, lookup records, tool results and verdicts."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Severity = Literal["low", "medium", "high"]
Criticality = Literal["low", "medium", "high"]
Category = Literal["threat", "benign", "ambiguous"]

AlertType = Literal[
    "suspicious_login",
    "new_credential_use",
    "auth_failures",
    "unusual_process",
    "config_change",
    "outbound_connection",
    "port_opened",
    "port_scan",
    "privilege_change",
    "mass_file_modification",
    "service_stopped",
]

ChangeKind = Literal[
    "credential_rotation",
    "service_account_handover",
    "deploy",
    "firewall_change",
    "maintenance",
    "access_grant",
]


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Alert(Record):
    """One security alert, as the agent receives it."""

    id: str
    type: AlertType
    principal: str = Field(description="User or service account the alert is about")
    host: str
    timestamp: datetime
    source_ip: str | None = None
    destination: str | None = None
    raw_message: str


class Label(Record):
    """Ground truth for one alert. The agent never sees this."""

    severity: Severity
    escalate: bool
    category: Category
    scenario: str
    tags: tuple[str, ...] = ()
    change_record_id: str | None = Field(
        default=None, description="Change record that explains or relates to the alert, if any"
    )


class LabeledAlert(Record):
    alert: Alert
    label: Label


class ChangeRecord(Record):
    id: str
    kind: ChangeKind
    title: str
    description: str
    hosts: tuple[str, ...]
    principals: tuple[str, ...]
    window_start: datetime
    window_end: datetime
    requested_by: str
    approved_by: str

    def search_text(self) -> str:
        """The text that gets embedded for semantic search."""
        return (
            f"{self.title}\n{self.description}\n"
            f"Kind: {self.kind}\nHosts: {', '.join(self.hosts)}\n"
            f"Accounts: {', '.join(self.principals)}"
        )


class Asset(Record):
    hostname: str
    ip: str
    role: str
    environment: Literal["prod", "staging", "dev"]
    criticality: Criticality
    owner: str = Field(description="Owning team, or the person for a workstation")
    os: str


class CredentialChange(Record):
    at: datetime
    kind: Literal["created", "rotated", "handover", "reset"]
    note: str


class Identity(Record):
    name: str
    account_type: Literal["human", "service", "scanner"]
    team: str
    owner: str | None = Field(default=None, description="Responsible person for non-human accounts")
    credential_changes: tuple[CredentialChange, ...] = ()


class Indicator(Record):
    value: str
    kind: Literal["ip", "domain"]
    threat_type: str
    confidence: Literal["low", "medium", "high"]


class ChangeRecordHit(Record):
    """One result from search_change_records."""

    record: ChangeRecord
    score: float


class Verdict(Record):
    """The agent's schema-validated output."""

    severity: Severity
    escalate: bool
    reason: str = Field(min_length=1)
    evidence: tuple[str, ...] = Field(default=(), description="IDs of the records used")
    needs_human: bool = False
