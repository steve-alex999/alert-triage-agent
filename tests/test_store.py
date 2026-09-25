import pytest
from qdrant_client import QdrantClient

from triage.data import load_alerts, load_change_records
from triage.embeddings import HashEmbedder
from triage.store import index_change_records, search_change_records


@pytest.fixture(scope="module")
def index():
    client = QdrantClient(":memory:")
    embedder = HashEmbedder()
    index_change_records(client, embedder, load_change_records())
    return client, embedder


def search_for(index, item, **overrides):
    client, embedder = index
    alert = item.alert
    kwargs = dict(host=alert.host, principal=alert.principal, at=alert.timestamp) | overrides
    return search_change_records(client, embedder, alert.raw_message, **kwargs)


def test_named_case_finds_its_handover_record(index):
    item = next(a for a in load_alerts() if "named_case" in a.label.tags)
    hits = search_for(index, item)
    assert hits[0].record.id == item.label.change_record_id
    assert hits[0].record.kind == "service_account_handover"


def test_every_benign_change_is_retrievable(index):
    for item in load_alerts():
        if item.label.category == "benign" and item.label.change_record_id:
            ids = [h.record.id for h in search_for(index, item)]
            assert item.label.change_record_id in ids, item.alert.id


def test_threats_retrieve_nothing_in_their_window(index):
    for item in load_alerts():
        if item.label.category == "threat":
            assert search_for(index, item) == [], item.alert.id


def test_filters_exclude_other_hosts(index):
    client, embedder = index
    hits = search_change_records(client, embedder, "deploy", host="db-prod-01", limit=50)
    assert hits
    assert all("db-prod-01" in h.record.hosts for h in hits)
