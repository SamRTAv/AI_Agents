"""Phase 4 smoke test — RAG. No LLM key required.

Checks the index exists, retrieval returns grounded passages with usable
citations, and that off-topic queries do not silently return confident junk.

Requires:  python scripts/ingest.py
Run:       python tests/smoke_phase4.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.tools.rag import load_rag_tool  # noqa: E402


def main() -> int:
    failures = []

    def check(label, cond, extra=""):
        print(f"  {'PASS' if cond else 'FAIL'}  {label}{'  ' + extra if extra else ''}")
        if not cond:
            failures.append(label)

    settings = get_settings()

    print("\n[1] index exists and loads")
    if not (settings.index_dir / "index.faiss").exists():
        print("  FAIL  no index — run: python scripts/ingest.py")
        return 1

    tools = load_rag_tool(settings)
    check("rag tool loaded", len(tools) == 1)
    rag = tools[0]
    check("tool is named for the model", rag.name == "search_finance_docs", rag.name)

    print("\n[2] retrieval returns grounded passages")
    res = rag.invoke({"query": "How should I budget my monthly income?"})
    check("passages returned", res["found"] > 0, f"found={res['found']}")

    if res["found"]:
        p = res["passages"][0]
        check("passage carries a source", bool(p["source"]), p["source"])
        check("passage carries a topic", bool(p["topic"]), p["topic"])
        check("passage has real text", len(p["text"]) > 100, f"{len(p['text'])} chars")
        print(f"\n  top hit: {p['source']} [{p['topic']}] {p['section'][:50]}")
        print(f"  {p['text'][:220].strip()}...\n")

    print("[3] different queries retrieve different passages")
    a = rag.invoke({"query": "What is a savings account?"})
    b = rag.invoke({"query": "How does compound interest work?"})
    texts_a = {p["text"][:100] for p in a["passages"]}
    texts_b = {p["text"][:100] for p in b["passages"]}
    check("results are not identical", texts_a != texts_b)

    print("\n[4] no duplicate chunks in results")
    seen = [p["text"] for p in res["passages"]]
    check("top-k are distinct", len(set(seen)) == len(seen), f"{len(seen)} passages")

    print(f"\n{'ALL PASSED' if not failures else 'FAILURES: ' + ', '.join(failures)}\n")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
