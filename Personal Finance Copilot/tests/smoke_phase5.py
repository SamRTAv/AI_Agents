"""Phase 5 smoke test — human-in-the-loop approval. No LLM key required.

Uses a scripted stub model that requests add_expense on the first turn, so the
approval path can be tested deterministically without spending tokens.

Run:  python tests/smoke_phase5.py
"""

import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

from app.graph import build_graph  # noqa: E402

executed: list[dict] = []


@tool
def add_expense(date: str, amount: float, category: str) -> dict:
    """Record an expense."""
    executed.append({"date": date, "amount": amount, "category": category})
    return {"status": "ok"}


@tool
def summarize(start_date: str, end_date: str) -> dict:
    """Summarise spending."""
    return {"total_amount": 17749.0}


class ScriptedLLM:
    """Requests a tool until one has run, then answers in plain text.

    Decides from the message history, NOT an instance counter. That matters:
    section [2] builds a fresh instance to simulate a server restart, so any
    in-process counter resets while the graph state (on disk) does not. A
    counter-based stub would re-request the tool after resuming and pause the
    graph a second time — which is exactly the asymmetry the checkpointer
    exists to remove.
    """

    def __init__(self, tool_name: str, args: dict):
        self.tool_name, self.args = tool_name, args

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        tool_results = [m for m in messages if isinstance(m, ToolMessage)]

        if not tool_results:
            return AIMessage(
                content="",
                tool_calls=[
                    {"name": self.tool_name, "args": self.args, "id": "call_1"}
                ],
            )

        if "declined" in str(tool_results[-1].content).lower():
            return AIMessage(content="Understood — I did not save anything.")
        return AIMessage(content="Done.")


async def main() -> int:
    failures = []

    def check(label, cond, extra=""):
        print(f"  {'PASS' if cond else 'FAIL'}  {label}{'  ' + extra if extra else ''}")
        if not cond:
            failures.append(label)

    db = Path(tempfile.gettempdir()) / "copilot_hitl.db"
    db.unlink(missing_ok=True)
    args = {"date": "2025-09-28", "amount": 450.0, "category": "transport"}

    print("\n[1] a write tool pauses for approval")
    async with AsyncSqliteSaver.from_conn_string(str(db)) as saver:
        await saver.setup()
        graph = build_graph(
            ScriptedLLM("add_expense", args), saver, [add_expense, summarize]
        )
        cfg = {"configurable": {"thread_id": "hitl-approve"}}
        r = await graph.ainvoke({"messages": [HumanMessage(content="add 450")]}, cfg)
        check("graph paused", bool(r.get("__interrupt__")))
        check("tool did NOT run yet", not executed, f"executed={executed}")

        payload = r["__interrupt__"][0].value
        check("payload names the tool", payload.get("tool") == "add_expense", str(payload.get("tool")))
        check("payload carries the args", payload.get("args") == args)

    print("\n[2] the pause survives a server restart")
    async with AsyncSqliteSaver.from_conn_string(str(db)) as saver:
        graph = build_graph(
            ScriptedLLM("add_expense", args), saver, [add_expense, summarize]
        )
        cfg = {"configurable": {"thread_id": "hitl-approve"}}
        r = await graph.ainvoke(Command(resume="yes"), cfg)
        check("resumed after restart", not r.get("__interrupt__"))
        check("tool ran exactly once", len(executed) == 1, f"executed={executed}")
        check("ran with the approved args", executed and executed[0]["amount"] == 450.0)

    print("\n[3] declining does not execute the tool")
    executed.clear()
    db2 = Path(tempfile.gettempdir()) / "copilot_hitl2.db"
    db2.unlink(missing_ok=True)
    async with AsyncSqliteSaver.from_conn_string(str(db2)) as saver:
        await saver.setup()
        graph = build_graph(
            ScriptedLLM("add_expense", args), saver, [add_expense, summarize]
        )
        cfg = {"configurable": {"thread_id": "hitl-decline"}}
        await graph.ainvoke({"messages": [HumanMessage(content="add 450")]}, cfg)
        r = await graph.ainvoke(Command(resume="no"), cfg)
        check("nothing was written", not executed, f"executed={executed}")
        check("model still produced a reply", bool(r["messages"][-1].content))

    print("\n[4] read-only tools are NOT gated")
    async with AsyncSqliteSaver.from_conn_string(str(db2)) as saver:
        graph = build_graph(
            ScriptedLLM("summarize", {"start_date": "2025-09-01", "end_date": "2025-09-30"}),
            saver,
            [add_expense, summarize],
        )
        cfg = {"configurable": {"thread_id": "hitl-read"}}
        r = await graph.ainvoke({"messages": [HumanMessage(content="total?")]}, cfg)
        check("summarize ran without pausing", not r.get("__interrupt__"))

    db.unlink(missing_ok=True)
    db2.unlink(missing_ok=True)
    print(f"\n{'ALL PASSED' if not failures else 'FAILURES: ' + ', '.join(failures)}\n")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
