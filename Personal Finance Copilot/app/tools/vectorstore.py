"""FAISS import shim.

LangChain 1.x moved the legacy integrations around, and langchain-community
0.3.x pins langchain-core<1.0 — which conflicts with the core 1.x that langgraph
1.x requires. Depending on which combination resolves in your venv, FAISS lives
in one of a few places.

Rather than hard-code one and break on the others, try them in order and give a
clear instruction if none work. Both ingest.py and rag.py import from here so
there is exactly one place to fix if the packaging changes again.
"""

from __future__ import annotations


def get_faiss_class():
    attempts = (
        ("langchain_community.vectorstores", "langchain-community"),
        ("langchain_classic.vectorstores", "langchain-classic"),
        ("langchain.vectorstores", "langchain"),
    )

    errors = []
    for module_path, package in attempts:
        try:
            module = __import__(module_path, fromlist=["FAISS"])
            return getattr(module, "FAISS")
        except Exception as exc:  # ImportError, or a version-conflict error
            errors.append(f"  {package}: {type(exc).__name__}: {exc}")

    raise ImportError(
        "Could not import FAISS from any known location.\n"
        + "\n".join(errors)
        + "\n\nTry one of:\n"
        "  pip install langchain-community\n"
        "  pip install langchain-classic\n"
        "If pip reports a langchain-core version conflict, prefer "
        "langchain-classic — it is the LangChain 1.x home for these."
    )
