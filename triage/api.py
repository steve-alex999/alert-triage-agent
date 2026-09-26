"""HTTP API: POST an alert, get back a verdict and the evidence trail.

    uvicorn triage.api:app --port 8000
    curl -s localhost:8000/triage -H 'content-type: application/json' -d @alert.json
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query
from fastapi.concurrency import run_in_threadpool

from triage.agent import Setup, TriageResult, triage
from triage.evaluate import LockedToolbox
from triage.llm import LLM
from triage.models import Alert
from triage.tools import Toolbox


def create_app(llm: LLM | None = None, toolbox: Toolbox | None = None) -> FastAPI:
    """Build the app. Tests pass a model and toolbox; otherwise both load from the environment at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if llm is None or toolbox is None:
            from triage.embeddings import get_embedder
            from triage.llm import get_llm
            from triage.store import get_client

        app.state.llm = llm or get_llm()
        # Toolbox.load builds the change-record index on first start.
        app.state.toolbox = LockedToolbox(toolbox or Toolbox.load(get_client(), get_embedder()))
        yield

    app = FastAPI(title="Alert Triage Agent", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "model": app.state.llm.model}

    @app.post("/triage")
    async def triage_alert(alert: Alert, setup: Setup = Query("agent")) -> TriageResult:
        toolbox = None if setup == "baseline" else app.state.toolbox
        # The loop makes blocking HTTP calls to the model; keep it off the event loop.
        return await run_in_threadpool(triage, alert, app.state.llm, toolbox, setup)

    return app


app = create_app()
