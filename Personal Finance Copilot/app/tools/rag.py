"""Retrieval tool over the curated document corpus.

Unlike the expense tools, this one stays local: it closes over a FAISS index
held in this process's memory. Moving it behind MCP would mean the server owned
the index — a redesign, not a decorator swap.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import BaseTool, tool

from app.config import Settings, get_settings
from app.tools.embeddings import get_embeddings
from app.tools.vectorstore import get_faiss_class

log = logging.getLogger("copilot.rag")

TOP_K = 4


def load_retriever(settings: Settings | None = None):
    """Load the prebuilt index from disk. Fast — no PDFs, no re-embedding."""
    settings = settings or get_settings()

    if not (settings.index_dir / "index.faiss").exists():
        raise FileNotFoundError(
            f"No FAISS index at {settings.index_dir}. "
            "Build it first:  python scripts/ingest.py"
        )

    FAISS = get_faiss_class()

    store = FAISS.load_local(
        str(settings.index_dir),
        get_embeddings(settings),
        # Safe: we wrote this file ourselves in scripts/ingest.py.
        allow_dangerous_deserialization=True,
    )
    return store.as_retriever(search_type="similarity", search_kwargs={"k": TOP_K})


def make_rag_tool(retriever) -> BaseTool:
    @tool
    def search_finance_docs(query: str) -> dict[str, Any]:
        """Search the curated library of Indian financial regulator publications.

        Covers budgeting, saving, investing, banking, insurance, tax and
        retirement material from SEBI, RBI, NCFE, AMFI, the Income Tax
        Department, IRDAI and PFRDA.

        Use this for any question about how something works, what a rule says,
        what a term means, or what an official limit or threshold is. Do NOT
        use it for the user's own spending — that lives in the expense tools.

        Returns passages with their source document. Cite the source in your
        answer. If nothing relevant comes back, say so rather than answering
        from memory.

        Args:
            query: What to look up, phrased as a natural-language question.
        """
        docs = retriever.invoke(query)
        if not docs:
            return {"query": query, "found": 0, "passages": []}

        return {
            "query": query,
            "found": len(docs),
            "passages": [
                {
                    "source": d.metadata.get("source", "unknown"),
                    "topic": d.metadata.get("topic", ""),
                    "section": d.metadata.get("h2") or d.metadata.get("h1") or "",
                    "text": d.page_content,
                }
                for d in docs
            ],
        }

    return search_finance_docs


def load_rag_tool(settings: Settings | None = None) -> list[BaseTool]:
    """Startup entry point.

    Non-fatal on failure, matching the MCP loader: a missing index degrades the
    agent to expense-only rather than refusing to boot. /health shows which
    tools actually loaded.
    """
    try:
        retriever = load_retriever(settings)
        log.info("RAG index loaded from disk")
        return [make_rag_tool(retriever)]
    except Exception as exc:
        log.error("RAG tool unavailable — continuing without it: %s", exc)
        return []
