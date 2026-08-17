"""MCP client — turns the expense server's tools into LangChain tools.

Written against the `mcp` SDK directly rather than langchain-mcp-adapters.
That was forced: scripts/diagnose_mcp.py showed the raw SDK completing the
stdio handshake and listing all 4 tools, while langchain-mcp-adapters 0.3.2
failed with `McpError: Connection closed` on the same server, same
interpreter, same versions. The adapter is a thin convenience layer, so doing
its job here costs ~80 lines and removes the dependency.

Nothing downstream changes: these are ordinary BaseTool objects, so ToolNode
and bind_tools treat them exactly like a local @tool.

Concurrency model: one session per tool call. The stdio transport is built on
anyio task groups that dislike being held across unrelated tasks, and a
short-lived session avoids that class of bug entirely. The cost is spawning
the server process per call (~1-2s on Windows). If that becomes the
bottleneck, the fix is a single long-lived session pinned to one task with a
request queue in front of it — meaningfully more complex, so not yet.
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import Field, create_model

log = logging.getLogger("copilot.mcp")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
EXPENSE_SERVER = BASE_DIR / "mcp_server" / "main.py"

_JSON_TO_PY: dict[str, Any] = {
    "string": str,
    "number": float,
    "integer": int,
    "boolean": bool,
    "array": list,
    "object": dict,
}


@asynccontextmanager
async def _session():
    """Open a stdio session to the expense server.

    `sys.executable` so the subprocess runs on the same interpreter as this
    app — a different Python on PATH may not have fastmcp installed.
    """
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    if not EXPENSE_SERVER.exists():
        raise FileNotFoundError(f"MCP server not found at {EXPENSE_SERVER}")

    params = StdioServerParameters(
        command=sys.executable,
        args=[str(EXPENSE_SERVER)],
        env=dict(os.environ),
    )

    # The server draws a banner and logs to stderr; none of it is ours to show.
    with open(os.devnull, "w", encoding="utf-8") as errlog:
        async with stdio_client(params, errlog=errlog) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session


def _python_type(spec: dict) -> Any:
    """Map one JSON-schema property onto a Python type."""
    if "type" in spec:
        return _JSON_TO_PY.get(spec["type"], Any)
    # Optional params arrive as anyOf: [{type: X}, {type: "null"}]
    for option in spec.get("anyOf", []):
        if option.get("type") and option["type"] != "null":
            return _JSON_TO_PY.get(option["type"], Any)
    return Any


def _args_model(tool_name: str, schema: dict):
    """Build a pydantic model from the tool's JSON schema.

    This is what gives the LLM typed arguments. Losing it would undo the type
    hints we added to mcp_server/main.py.
    """
    properties = (schema or {}).get("properties", {})
    required = set((schema or {}).get("required", []))

    fields: dict[str, tuple] = {}
    for key, spec in properties.items():
        py_type = _python_type(spec)
        description = spec.get("description", "")
        if key in required:
            fields[key] = (py_type, Field(..., description=description))
        else:
            fields[key] = (
                py_type | None if py_type is not Any else Any,
                Field(spec.get("default", None), description=description),
            )

    model_name = "".join(part.capitalize() for part in tool_name.split("_")) + "Args"
    return create_model(model_name, **fields) if fields else create_model(model_name)


def _result_to_text(result: Any) -> str:
    """Flatten an MCP CallToolResult into text for the ToolMessage.

    Errors are returned as text rather than raised, deliberately: the model
    then reads the message and retries with corrected arguments instead of the
    graph aborting. That behaviour is covered by tests/smoke_phase3.py.
    """
    parts: list[str] = []
    for block in getattr(result, "content", None) or []:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)

    body = "\n".join(parts) if parts else "(tool returned no content)"
    if getattr(result, "isError", False):
        return f"Error from tool: {body}"
    return body


def _make_tool(spec: Any) -> BaseTool:
    name = spec.name

    async def call(**kwargs: Any) -> str:
        try:
            async with _session() as session:
                result = await session.call_tool(name, kwargs)
                return _result_to_text(result)
        except Exception as exc:
            # Keep transport failures in-band too, so one bad call does not
            # kill the whole conversation.
            log.error("MCP call to %s failed: %s", name, exc)
            return f"Error calling tool '{name}': {exc}"

    return StructuredTool.from_function(
        coroutine=call,
        name=name,
        description=spec.description or f"MCP tool {name}",
        args_schema=_args_model(name, getattr(spec, "inputSchema", {}) or {}),
    )


async def load_mcp_tools() -> list[BaseTool]:
    """Discover the server's tools at startup.

    Failure is non-fatal: if the expense server cannot start, the app still
    serves conversation and RAG. /health reports the tool count, so a degraded
    start is visible rather than silent.
    """
    try:
        async with _session() as session:
            listed = await session.list_tools()

        tools = [_make_tool(spec) for spec in listed.tools]
        log.info("Loaded %d MCP tool(s): %s", len(tools), [t.name for t in tools])
        return tools

    except BaseException as exc:
        for line in _explain(exc):
            log.error("MCP: %s", line)
        log.error(
            "Continuing without expense tools. Diagnose with:\n"
            "    python scripts/diagnose_mcp.py"
        )
        return []


def _explain(exc: BaseException, depth: int = 0) -> list[str]:
    """Flatten ExceptionGroups and __cause__ chains into readable lines."""
    pad = "  " * depth
    lines = [f"{pad}{type(exc).__name__}: {exc}"]
    for sub in getattr(exc, "exceptions", ()) or ():
        lines.extend(_explain(sub, depth + 1))
    nested = exc.__cause__ or exc.__context__
    if nested is not None and not getattr(exc, "exceptions", None):
        lines.extend(_explain(nested, depth + 1))
    return lines[:12]
