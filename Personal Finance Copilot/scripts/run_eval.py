"""Run the eval set and report tool-selection accuracy.

    python scripts/run_eval.py                 # everything
    python scripts/run_eval.py --kind composed # one category
    python scripts/run_eval.py --limit 5       # quick smoke

What it measures, per case:
  tool_recall     did the run call every tool in expected_tools?
  no_forbidden    did it avoid everything in forbidden_tools?
  approval        did write cases actually pause for approval?
  latency_s       wall clock

These are cheap, objective, reproducible metrics — no LLM judge needed, so the
numbers are stable enough to compare across prompt and chunking changes. That
comparison is the point: run it, change one thing, run it again.

Every run also lands in LangSmith (if tracing is on) under the configured
project, so you can open any failing case and see exactly what happened.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from app.config import get_llm, get_settings  # noqa: E402
from app.graph import build_graph  # noqa: E402
from app.tools.mcp_client import load_mcp_tools  # noqa: E402
from app.tools.rag import load_rag_tool  # noqa: E402

DATASET = Path(__file__).resolve().parent.parent / "evals" / "dataset.json"

REFUSAL_MARKERS = (
    "cannot", "can't", "not able", "do not", "don't", "unable",
    "outside", "not something", "no data", "could not find", "not found",
)


def called_tools(messages) -> list[str]:
    names = []
    for m in messages:
        for call in getattr(m, "tool_calls", None) or []:
            names.append(call["name"])
    return names


async def run_case(graph, case: dict) -> dict:
    # A fresh thread per case, so one case cannot contaminate the next.
    config = {"configurable": {"thread_id": f"eval-{case['id']}-{uuid.uuid4().hex[:6]}"}}
    started = time.perf_counter()

    try:
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=case["query"])]}, config=config
        )
        error = None
    except Exception as exc:
        return {
            "id": case["id"], "kind": case["kind"], "error": str(exc),
            "tool_recall": False, "no_forbidden": False, "approval_ok": False,
            "latency_s": time.perf_counter() - started, "tools": [],
        }

    latency = time.perf_counter() - started
    messages = result.get("messages", []) if isinstance(result, dict) else []
    tools = called_tools(messages)
    paused = bool(isinstance(result, dict) and result.get("__interrupt__"))

    expected = set(case.get("expected_tools", []))
    forbidden = set(case.get("forbidden_tools", []))
    answer = str(messages[-1].content).lower() if messages else ""

    approval_ok = True
    if case.get("expects_approval"):
        approval_ok = paused
    if case.get("must_refuse"):
        # A refusal should not silently invent an answer.
        approval_ok = any(m in answer for m in REFUSAL_MARKERS)

    return {
        "id": case["id"],
        "kind": case["kind"],
        "error": error,
        "query": case["query"],
        # The answer text is what you need to judge a "behaviour" failure:
        # a refusal the marker list missed looks identical to no refusal at all
        # until you read it.
        "answer": str(messages[-1].content) if messages else "",
        "expected_tools": sorted(expected),
        "tools": tools,
        "paused": paused,
        "tool_recall": expected.issubset(set(tools)),
        "no_forbidden": not (forbidden & set(tools)),
        "approval_ok": approval_ok,
        "latency_s": latency,
    }


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", help="expense | rag | composed | write | adversarial")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    data = json.loads(DATASET.read_text(encoding="utf-8"))
    cases = data["cases"]
    if args.kind:
        cases = [c for c in cases if c["kind"] == args.kind]
    if args.limit:
        cases = cases[: args.limit]

    settings = get_settings()
    print(f"Model: {settings.llm_provider}/{settings.active_model_name}")
    print(f"Cases: {len(cases)}\n")

    tools = list(await load_mcp_tools()) + list(load_rag_tool(settings))
    if not tools:
        print("No tools loaded — run scripts/ingest.py and check the MCP server.")
        return 1

    # InMemorySaver, not the sqlite one: eval threads are throwaway and should
    # not pollute the real checkpoint database.
    graph = build_graph(get_llm(settings), InMemorySaver(), tools)

    results = []
    for i, case in enumerate(cases, 1):
        r = await run_case(graph, case)
        results.append(r)
        ok = r["tool_recall"] and r["no_forbidden"] and r["approval_ok"]
        print(
            f"  [{i:>2}/{len(cases)}] {'ok  ' if ok else 'FAIL'} "
            f"{r['id']:<8} {r['latency_s']:>5.1f}s  tools={r['tools']}"
        )

    print("\n" + "=" * 62)
    by_kind = defaultdict(list)
    for r in results:
        by_kind[r["kind"]].append(r)

    def pct(rows, key):
        return 100 * sum(1 for r in rows if r[key]) / len(rows) if rows else 0.0

    print(f"{'kind':<14}{'n':>4}{'recall':>9}{'no-forbid':>11}{'behav':>8}{'p50':>7}")
    print("-" * 62)
    for kind, rows in sorted(by_kind.items()):
        lat = statistics.median(r["latency_s"] for r in rows)
        print(
            f"{kind:<14}{len(rows):>4}{pct(rows,'tool_recall'):>8.0f}%"
            f"{pct(rows,'no_forbidden'):>10.0f}%{pct(rows,'approval_ok'):>7.0f}%"
            f"{lat:>6.1f}s"
        )

    lat_all = [r["latency_s"] for r in results]
    print("-" * 62)
    print(
        f"{'OVERALL':<14}{len(results):>4}{pct(results,'tool_recall'):>8.0f}%"
        f"{pct(results,'no_forbidden'):>10.0f}%{pct(results,'approval_ok'):>7.0f}%"
        f"{statistics.median(lat_all):>6.1f}s"
    )
    print(
        f"\nlatency  p50={statistics.median(lat_all):.1f}s  "
        f"p95={sorted(lat_all)[int(len(lat_all) * 0.95) - 1]:.1f}s  "
        f"max={max(lat_all):.1f}s"
    )

    out = Path(__file__).resolve().parent.parent / "evals" / "last_run.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nPer-case detail: {out}")
    if settings.langsmith_tracing:
        print(f"Traces: https://smith.langchain.com  (project {settings.langsmith_project})")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
