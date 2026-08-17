"""Find out exactly why the MCP tools are not loading.

    python scripts/diagnose_mcp.py

"Connection closed" only tells us the child process died. The reason is on the
server's stderr, which the client libraries discard by default. The raw mcp SDK
accepts an `errlog`, so this bypasses langchain-mcp-adapters to capture it.

That also splits the problem cleanly:
  raw OK,  adapter FAILS -> langchain-mcp-adapters
  raw FAILS              -> mcp / anyio / the server itself (stderr will say)
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SERVER = Path(__file__).resolve().parent.parent / "mcp_server" / "main.py"
PACKAGES = ("fastmcp", "mcp", "langchain-mcp-adapters", "anyio", "httpx", "pydantic")


def rule(t: str) -> None:
    print(f"\n{'=' * 66}\n{t}\n{'=' * 66}")


def chain(exc: BaseException, depth: int = 0) -> list[str]:
    pad = "  " * depth
    out = [f"{pad}{type(exc).__name__}: {exc}"]
    for sub in getattr(exc, "exceptions", ()) or ():
        out.extend(chain(sub, depth + 1))
    nested = exc.__cause__ or exc.__context__
    if nested is not None and not getattr(exc, "exceptions", None):
        out.extend(chain(nested, depth + 1))
    return out[:14]


async def raw_handshake(errlog) -> list[str]:
    """Handshake using the mcp SDK directly, with the child's stderr captured."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(command=sys.executable, args=[str(SERVER)])
    async with stdio_client(params, errlog=errlog) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            listed = await session.list_tools()
            return [t.name for t in listed.tools]


async def adapter_handshake() -> list[str]:
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient(
        {
            "expense": {
                "transport": "stdio",
                "command": sys.executable,
                "args": [str(SERVER)],
            }
        }
    )
    return [t.name for t in await client.get_tools()]


def main() -> int:
    rule("1. environment")
    print(f"python : {sys.executable}")
    print(f"server : {SERVER}  exists={SERVER.exists()}")
    for p in PACKAGES:
        try:
            print(f"  {p:<26} {version(p)}")
        except PackageNotFoundError:
            print(f"  {p:<26} MISSING")

    rule("2. server standalone (stdin closed immediately)")
    proc = subprocess.run(
        [sys.executable, str(SERVER)], input=b"", capture_output=True, timeout=60
    )
    err = proc.stderr.decode(errors="replace")
    print(f"exit code: {proc.returncode}")
    if "Traceback" in err:
        print("--- traceback ---")
        print(err[-2000:])
        print("\nThe server crashes on its own. Fix this first.")
        return 1
    print("starts cleanly")

    rule("3. RAW mcp SDK handshake  (server stderr captured)")
    logfile = Path(tempfile.gettempdir()) / "mcp_stderr.log"
    raw_ok = False
    with open(logfile, "w", encoding="utf-8", errors="replace") as errlog:
        try:
            names = asyncio.run(raw_handshake(errlog))
            raw_ok = True
            print(f"OK — {len(names)} tools: {names}")
        except BaseException as exc:
            print("FAILED:")
            for line in chain(exc):
                print(f"  {line}")

    captured = logfile.read_text(encoding="utf-8", errors="replace")
    print("\n--- server stderr during handshake ---")
    print(captured[-3000:] if captured.strip() else "(empty)")

    rule("4. langchain-mcp-adapters handshake")
    adapter_ok = False
    try:
        names = asyncio.run(adapter_handshake())
        adapter_ok = True
        print(f"OK — {len(names)} tools: {names}")
    except BaseException as exc:
        print("FAILED:")
        for line in chain(exc):
            print(f"  {line}")

    rule("verdict")
    if raw_ok and adapter_ok:
        print("Both work. Restart uvicorn fully (--reload does not re-run")
        print("startup after a pip install).")
        return 0
    if raw_ok and not adapter_ok:
        print("The MCP protocol is fine; langchain-mcp-adapters 0.3.2 is the")
        print("problem. Fix: talk to the server with the mcp SDK directly and")
        print("wrap the tools ourselves — tell Claude and it will rewrite")
        print("app/tools/mcp_client.py that way.")
        return 1
    print("The raw SDK handshake failed too. Read the server stderr above —")
    print("if it is empty, the child died before writing anything, which")
    print("points at the transport (anyio/mcp versions) rather than our code.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
