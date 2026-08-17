"""Phase 1 smoke test — runs without any API key.

Substitutes a stub LLM for the real one so we can prove the parts that are
actually ours: graph wiring, the add_messages reducer, checkpointer
persistence, and thread isolation.

Run:  python tests/smoke_phase1.py
"""

import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver  # noqa: E402

from app.graph import build_graph  # noqa: E402


class StubLLM:
    """Echoes how many messages it was given, so we can see history growing."""

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        # messages includes the SystemMessage prepended by chat_node
        human = [m for m in messages if isinstance(m, HumanMessage)]
        return AIMessage(
            content=f"[stub] saw {len(human)} human message(s); "
            f"latest={human[-1].content!r}"
        )


async def main() -> int:
    db = Path(tempfile.gettempdir()) / "copilot_smoke.db"
    db.unlink(missing_ok=True)
    failures = []

    def check(label, cond):
        print(f"  {'PASS' if cond else 'FAIL'}  {label}")
        if not cond:
            failures.append(label)

    cfg_a = {"configurable": {"thread_id": "thread-A"}}
    cfg_b = {"configurable": {"thread_id": "thread-B"}}

    print("\n[1] graph builds and answers")
    async with AsyncSqliteSaver.from_conn_string(str(db)) as saver:
        await saver.setup()
        graph = build_graph(llm=StubLLM(), checkpointer=saver, tools=[])

        r1 = await graph.ainvoke({"messages": [HumanMessage(content="hello")]}, cfg_a)
        check("returns a reply", bool(r1["messages"][-1].content))
        check("saw 1 human message", "saw 1 human" in r1["messages"][-1].content)

        print("\n[2] memory accumulates within a thread")
        r2 = await graph.ainvoke({"messages": [HumanMessage(content="again")]}, cfg_a)
        check("saw 2 human messages", "saw 2 human" in r2["messages"][-1].content)
        check("4 messages in state", len(r2["messages"]) == 4)

        print("\n[3] threads are isolated")
        r3 = await graph.ainvoke({"messages": [HumanMessage(content="other")]}, cfg_b)
        check("thread-B starts fresh", "saw 1 human" in r3["messages"][-1].content)

    print("\n[4] state survives a full restart (new connection, new graph)")
    async with AsyncSqliteSaver.from_conn_string(str(db)) as saver:
        graph = build_graph(llm=StubLLM(), checkpointer=saver, tools=[])
        snap = await graph.aget_state(cfg_a)
        restored = snap.values.get("messages", [])
        check("thread-A history restored", len(restored) == 4)
        check(
            "content intact",
            any(getattr(m, "content", "") == "again" for m in restored),
        )

        r4 = await graph.ainvoke({"messages": [HumanMessage(content="third")]}, cfg_a)
        check("continues from restored state", "saw 3 human" in r4["messages"][-1].content)

    db.unlink(missing_ok=True)
    print(f"\n{'ALL PASSED' if not failures else 'FAILURES: ' + ', '.join(failures)}\n")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
