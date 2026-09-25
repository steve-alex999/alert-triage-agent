from collections import Counter
from datetime import timedelta

import pytest

from triage import data
from triage.synthetic import generate, write


@pytest.fixture(scope="module")
def dataset():
    return generate()


def test_counts(dataset):
    assert len(dataset["alerts"]) == 120
    assert len(dataset["change_records"]) == 300
    assert len(dataset["assets"]) == 50
    assert len(dataset["identities"]) == 80
    assert len(dataset["threat_intel"]) == 200

    labels = [a.label for a in dataset["alerts"]]
    assert Counter(l.category for l in labels) == {"threat": 40, "benign": 60, "ambiguous": 20}
    assert sum("injection" in l.tags for l in labels) == 10
    assert sum("named_case" in l.tags for l in labels) == 1


def test_ids_are_unique(dataset):
    for name, key in [("alerts", lambda a: a.alert.id), ("change_records", lambda c: c.id)]:
        ids = [key(row) for row in dataset[name]]
        assert len(ids) == len(set(ids))


def test_references_resolve(dataset):
    hosts = {a.hostname for a in dataset["assets"]}
    accounts = {i.name for i in dataset["identities"]}
    changes = {c.id for c in dataset["change_records"]}
    for item in dataset["alerts"]:
        assert item.alert.host in hosts
        assert item.alert.principal in accounts
        assert item.label.change_record_id is None or item.label.change_record_id in changes
    for change in dataset["change_records"]:
        assert set(change.hosts) <= hosts
        assert set(change.principals) <= accounts


def test_labels_match_categories(dataset):
    for item in dataset["alerts"]:
        label = item.label
        assert label.escalate == (label.category != "benign")
        if label.category == "benign":
            assert label.severity == "low"


def covering(changes, alert, slack=timedelta(0)):
    """Change records on the alert's host or account whose window covers the alert time."""
    return [
        c for c in changes
        if (alert.host in c.hosts or alert.principal in c.principals)
        and c.window_start - slack <= alert.timestamp <= c.window_end + slack
    ]


def test_threats_have_no_explaining_change(dataset):
    """No change record touches a threat's host or account within a day of the alert."""
    for item in dataset["alerts"]:
        if item.label.category == "threat":
            assert covering(dataset["change_records"], item.alert, slack=timedelta(hours=24)) == []


def test_benign_changes_cover_their_alert(dataset):
    by_id = {c.id: c for c in dataset["change_records"]}
    for item in dataset["alerts"]:
        if item.label.category == "benign" and item.label.change_record_id:
            change = by_id[item.label.change_record_id]
            assert change.window_start <= item.alert.timestamp <= change.window_end
            assert item.alert.host in change.hosts


def test_named_case(dataset):
    item = next(a for a in dataset["alerts"] if "named_case" in a.label.tags)
    assert item.alert.type == "new_credential_use"
    assert item.label.escalate is False
    identity = next(i for i in dataset["identities"] if i.name == item.alert.principal)
    assert any(c.kind == "handover" and item.label.change_record_id in c.note
               for c in identity.credential_changes)


def test_threat_indicators_are_in_the_intel_list(dataset):
    intel = {i.value for i in dataset["threat_intel"]}
    for item in dataset["alerts"]:
        if item.label.scenario == "bad_ip_login":
            assert item.alert.source_ip in intel
        if item.label.scenario == "bad_destination":
            assert item.alert.destination in intel
        if item.label.category != "threat" and item.alert.source_ip:
            assert item.alert.source_ip not in intel


def test_generation_is_deterministic_and_matches_committed_data(dataset, tmp_path):
    write(dataset, tmp_path)
    for filename in data.FILES.values():
        committed = (data.DATA_DIR / filename).read_text()
        assert (tmp_path / filename).read_text() == committed, (
            f"data/{filename} is stale; run `python -m triage.synthetic`"
        )
