"""Phase 3 smoke test — MCP tools over stdio. No LLM key required.

Proves the transport works end to end: spawns the expense server as a
subprocess, discovers its tools, and calls them for real.

Run:  python tests/smoke_phase3.py
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.tools.mcp_client import load_mcp_tools  # noqa: E402

EXPECTED = {"add_expense", "list_expenses", "summarize", "list_categories"}


def json_schema_of(tool) -> dict:
    """The JSON schema the LLM actually sees for a tool's arguments.

    `args_schema` is a pydantic model class when tools are built with
    StructuredTool, and a plain dict under some other wrappers. Normalise both
    so this test checks the schema itself rather than how it is stored.
    """
    schema = tool.args_schema
    if schema is None:
        return {}
    if isinstance(schema, dict):
        return schema
    if hasattr(schema, "model_json_schema"):
        return schema.model_json_schema()
    return {}


def unwrap(result):
    """MCP tools return content blocks, e.g. [{'type':'text','text':'<json>'}].

    The agent doesn't care — ToolNode puts the text straight into a ToolMessage
    and the model reads the JSON. Tests need the parsed object though.
    """
    if isinstance(result, list) and result and isinstance(result[0], dict):
        result = result[0].get("text", result)
    return json.loads(result) if isinstance(result, str) else result


async def main() -> int:
    failures = []

    def check(label, cond, extra=""):
        print(f"  {'PASS' if cond else 'FAIL'}  {label}{'  ' + extra if extra else ''}")
        if not cond:
            failures.append(label)

    print("\n[1] discover tools over stdio")
    tools = await load_mcp_tools()
    names = {t.name for t in tools}
    check("server started and returned tools", bool(tools), f"({len(tools)} found)")
    check("expected tool set present", EXPECTED <= names, f"{sorted(names)}")

    by_name = {t.name: t for t in tools}

    print("\n[2] tool schemas reached the client")
    if "add_expense" in by_name:
        props = json_schema_of(by_name["add_expense"]).get("properties", {})
        check("add_expense exposes typed args", "amount" in props, f"{sorted(props)}")
        check(
            "amount typed as number",
            props.get("amount", {}).get("type") == "number",
            str(props.get("amount")),
        )
        check(
            "date carries its description",
            "YYYY-MM-DD" in str(props.get("date", {}).get("description", "")),
            str(props.get("date", {}).get("description"))[:60],
        )

    print("\n[3] summarize returns real aggregates")
    result = await by_name["summarize"].ainvoke(
        {"start_date": "2025-09-01", "end_date": "2025-09-30"}
    )
    data = unwrap(result)
    total = data.get("total_amount")
    cats = {c["category"] for c in data.get("by_category", [])}
    check("total is the seeded 17749", total == 17749.0, f"got {total}")
    check("legacy labels were normalised", "Transportation" not in cats, f"{sorted(cats)}")
    check("transport merged into one bucket", "transport" in cats)

    print("\n[4] validation errors come back as content, not exceptions")
    # This is the behaviour we want: a tool error is returned to the model as a
    # tool result so it can read the message and retry with corrected arguments.
    # Raising would abort the graph instead. It also means the wording of the
    # error is effectively a prompt — it has to say how to fix the problem.
    bad = await by_name["summarize"].ainvoke(
        {"start_date": "01-09-2025", "end_date": "2025-09-30"}
    )
    text = bad[0]["text"] if isinstance(bad, list) and bad else str(bad)
    check("bad date is rejected", "Error" in text)
    check("message tells the model the fix", "YYYY-MM-DD" in text)
    check("message echoes the bad value", "01-09-2025" in text)

    print("\n[5] list_categories exposes the taxonomy")
    cats_result = await by_name["list_categories"].ainvoke({})
    taxonomy = unwrap(cats_result)
    check("taxonomy returned", "food" in taxonomy, f"{len(taxonomy)} categories")

    print(f"\n{'ALL PASSED' if not failures else 'FAILURES: ' + ', '.join(failures)}\n")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
