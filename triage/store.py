"""Qdrant index of change records, and the semantic search behind search_change_records.

With QDRANT_URL set this talks to a Qdrant server; otherwise it uses Qdrant's embedded
mode, stored in .qdrant/ at the repo root, so nothing else needs to be running.

    python -m triage.store                      # rebuild the index, then search for the named case
    python -m triage.store --query "rotate svc-billing password" --host db-prod-02
"""

from __future__ import annotations

import argparse
import os
import uuid
from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    Range,
    VectorParams,
)

from triage.embeddings import Embedder, get_embedder
from triage.models import ChangeRecord, ChangeRecordHit

COLLECTION = "change_records"
LOCAL_PATH = Path(__file__).resolve().parent.parent / ".qdrant"
DEFAULT_WINDOW = timedelta(hours=24)


def get_client() -> QdrantClient:
    url = os.environ.get("QDRANT_URL")
    return QdrantClient(url=url) if url else QdrantClient(path=str(LOCAL_PATH))


def _point_id(record_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"change-record/{record_id}"))


def index_change_records(
    client: QdrantClient, embedder: Embedder, records: Sequence[ChangeRecord], batch_size: int = 64
) -> None:
    """Recreate the collection and load every record."""
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(
        COLLECTION, vectors_config=VectorParams(size=embedder.dim, distance=Distance.COSINE)
    )
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        vectors = embedder.embed_documents([r.search_text() for r in batch])
        client.upsert(
            COLLECTION,
            points=[
                PointStruct(
                    id=_point_id(r.id),
                    vector=v,
                    payload={
                        **r.model_dump(mode="json"),
                        "window_start_ts": r.window_start.timestamp(),
                        "window_end_ts": r.window_end.timestamp(),
                    },
                )
                for r, v in zip(batch, vectors, strict=True)
            ],
        )


def search_change_records(
    client: QdrantClient,
    embedder: Embedder,
    query: str,
    *,
    host: str | None = None,
    principal: str | None = None,
    at: datetime | None = None,
    window: timedelta = DEFAULT_WINDOW,
    limit: int = 5,
) -> list[ChangeRecordHit]:
    """Semantic search over change records.

    `host` and `principal` keep records that touch that host OR that account.
    `at` keeps records whose change window overlaps [at - window, at + window].
    """
    must: list[FieldCondition] = []
    should: list[FieldCondition] = []
    if host:
        should.append(FieldCondition(key="hosts", match=MatchValue(value=host)))
    if principal:
        should.append(FieldCondition(key="principals", match=MatchValue(value=principal)))
    if at:
        must.append(FieldCondition(key="window_start_ts", range=Range(lte=(at + window).timestamp())))
        must.append(FieldCondition(key="window_end_ts", range=Range(gte=(at - window).timestamp())))

    response = client.query_points(
        COLLECTION,
        query=embedder.embed_query(query),
        query_filter=Filter(must=must or None, should=should or None),
        limit=limit,
        with_payload=True,
    )
    hits = []
    for point in response.points:
        payload = {k: v for k, v in point.payload.items() if not k.endswith("_ts")}
        hits.append(ChangeRecordHit(record=ChangeRecord.model_validate(payload), score=point.score))
    return hits


def main() -> None:
    from triage.data import load_alerts, load_change_records

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--query", help="Free-text query; defaults to the named credential-handover alert")
    parser.add_argument("--host")
    parser.add_argument("--principal")
    parser.add_argument("--skip-index", action="store_true", help="Search the existing index")
    args = parser.parse_args()

    embedder = get_embedder()
    client = get_client()
    if not args.skip_index:
        records = load_change_records()
        index_change_records(client, embedder, records)
        print(f"Indexed {len(records)} change records with {embedder.name}")

    at = None
    query, host, principal = args.query, args.host, args.principal
    if query is None:
        named = next(a for a in load_alerts() if "named_case" in a.label.tags)
        query, host, principal, at = (
            named.alert.raw_message, named.alert.host, named.alert.principal, named.alert.timestamp,
        )
        print(f"\nNamed case {named.alert.id}: {query}\nExpected record: {named.label.change_record_id}")

    print()
    for hit in search_change_records(client, embedder, query, host=host, principal=principal, at=at):
        r = hit.record
        print(f"{hit.score:.3f}  {r.id}  {r.kind:<25} {r.title}")


if __name__ == "__main__":
    main()
