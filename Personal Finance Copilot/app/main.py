"""FastAPI entrypoint.

The important idea in this file is the split between *startup* and *per
request*:

  startup (once)   - open the checkpointer, build the LLM, launch the MCP
                     server, load the FAISS index, compile the graph.
  per request      - look up the already-compiled graph and invoke it.

Anything expensive that leaked into a request handler would be paid again on
every single user message.
"""

from __future__ import annotations

import json
import logging
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from app.config import get_llm, get_settings
from app.graph import build_graph
from app.schemas import (
    ApprovalPayload,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    HistoryResponse,
    ResumeRequest,
    Turn,
)
from app.tools.mcp_client import load_mcp_tools
from app.tools.rag import load_rag_tool

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("copilot")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    # AsyncExitStack keeps the checkpointer connection open for the app's whole
    # lifetime and closes it cleanly on shutdown.
    async with AsyncExitStack() as stack:
        log.info("Opening checkpointer at %s", settings.checkpoint_db)
        checkpointer = await stack.enter_async_context(
            AsyncSqliteSaver.from_conn_string(str(settings.checkpoint_db))
        )
        await checkpointer.setup()  # idempotent

        log.info(
            "Building LLM: provider=%s model=%s",
            settings.llm_provider,
            settings.active_model_name,
        )
        llm = get_llm(settings)

        # Both loaders are non-fatal, so a broken corpus or a dead MCP server
        # degrades the agent instead of blocking startup.
        tools: list = []
        tools.extend(await load_mcp_tools())
        tools.extend(load_rag_tool(settings))
        log.info("Tools available: %s", [t.name for t in tools])

        app.state.settings = settings
        app.state.tools = tools
        app.state.graph = build_graph(llm=llm, checkpointer=checkpointer, tools=tools)

        if settings.langsmith_tracing and settings.langsmith_api_key:
            log.info("LangSmith tracing ON  (project=%s)", settings.langsmith_project)
        else:
            log.warning("LangSmith tracing OFF — set LANGSMITH_TRACING and _API_KEY")

        log.info("Startup complete. Swagger UI at http://127.0.0.1:8000/docs")
        yield

    log.info("Shutdown complete.")


app = FastAPI(
    title="Personal Finance Copilot",
    version="0.5.0",
    description=(
        "A LangGraph agent that answers personal-finance questions by combining "
        "retrieval over curated regulator publications with tools over the "
        "user's own expense data.\n\n"
        "**Approval flow:** if `/chat` returns `status: \"approval_required\"`, "
        "the graph is paused. Send the same `thread_id` to `/resume` with "
        "`\"yes\"` or `\"no\"`. The pause is checkpointed to disk, so it "
        "survives a server restart."
    ),
    lifespan=lifespan,
)


def _thread_config(thread_id: str) -> dict:
    """LangGraph looks up saved state by thread_id. This is the whole of
    multi-user isolation: two thread_ids never see each other's messages."""
    return {"configurable": {"thread_id": thread_id}}


def _extract_interrupt(result: Any) -> dict | None:
    """Pull the interrupt payload out of a graph result, if it paused.

    LangGraph surfaces pending interrupts under the "__interrupt__" key. The
    shape has shifted between versions, so unwrap defensively.
    """
    if not isinstance(result, dict):
        return None
    raw = result.get("__interrupt__")
    if not raw:
        return None
    first = raw[0] if isinstance(raw, (list, tuple)) and raw else raw
    value = getattr(first, "value", first)
    return value if isinstance(value, dict) else {"question": str(value)}


def _as_response(thread_id: str, result: Any) -> ChatResponse:
    payload = _extract_interrupt(result)
    if payload:
        return ChatResponse(
            status="approval_required",
            thread_id=thread_id,
            approval=ApprovalPayload(
                tool=payload.get("tool"),
                args=payload.get("args"),
                question=payload.get("question", "Approve this action?"),
            ),
        )

    messages = result.get("messages", []) if isinstance(result, dict) else []
    answer = str(messages[-1].content) if messages else ""
    return ChatResponse(status="completed", thread_id=thread_id, answer=answer)


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    s = app.state.settings
    return HealthResponse(
        status="ok",
        llm_provider=s.llm_provider,
        model=s.active_model_name,
        tools_loaded=len(app.state.tools),
        tool_names=[t.name for t in app.state.tools],
        langsmith_tracing=bool(s.langsmith_tracing and s.langsmith_api_key),
        langsmith_project=s.langsmith_project if s.langsmith_tracing else None,
    )


@app.post("/chat", response_model=ChatResponse, tags=["chat"])
async def chat(req: ChatRequest) -> ChatResponse:
    """Send one message.

    You send a single message, but the graph sees the whole conversation: the
    checkpointer reloads prior messages for this `thread_id` before the model
    runs, and saves the updated state afterwards.

    If the model tries to write data (add_expense), this returns
    `approval_required` instead of an answer — call `/resume` next.
    """
    try:
        result = await app.state.graph.ainvoke(
            {"messages": [HumanMessage(content=req.message)]},
            config=_thread_config(req.thread_id),
        )
    except Exception as exc:
        log.exception("Graph invocation failed")
        raise HTTPException(status_code=502, detail=f"Agent error: {exc}") from exc

    return _as_response(req.thread_id, result)


@app.post("/resume", response_model=ChatResponse, tags=["chat"])
async def resume(req: ResumeRequest) -> ChatResponse:
    """Answer a pending approval and let the graph continue.

    `Command(resume=...)` hands the decision back to the exact `interrupt()`
    call that paused, and execution continues from there — not from the start.
    """
    try:
        result = await app.state.graph.ainvoke(
            Command(resume=req.decision),
            config=_thread_config(req.thread_id),
        )
    except Exception as exc:
        log.exception("Resume failed")
        raise HTTPException(
            status_code=502,
            detail=f"Resume failed (is this thread actually paused?): {exc}",
        ) from exc

    return _as_response(req.thread_id, result)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post("/chat/stream", tags=["chat"])
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    """Same as /chat, but streams the answer token by token over SSE.

    Event types:
      token      - a fragment of the answer
      tool_start - the model called a tool (name + args)
      tool_end   - that tool returned
      approval   - the graph paused; call /resume
      done       - finished
      error      - something failed mid-stream

    Swagger renders this poorly (it buffers). Test with curl -N instead.
    """

    async def generate() -> AsyncIterator[str]:
        config = _thread_config(req.thread_id)
        try:
            async for mode, chunk in app.state.graph.astream(
                {"messages": [HumanMessage(content=req.message)]},
                config=config,
                stream_mode=["messages", "updates"],
            ):
                if mode == "messages":
                    msg = chunk[0] if isinstance(chunk, tuple) else chunk
                    text = getattr(msg, "content", "")
                    if text:
                        yield _sse("token", {"text": str(text)})

                elif mode == "updates" and isinstance(chunk, dict):
                    if "__interrupt__" in chunk:
                        payload = _extract_interrupt(chunk) or {}
                        yield _sse("approval", payload)
                        return

                    for node, update in chunk.items():
                        if not isinstance(update, dict):
                            continue
                        for m in update.get("messages", []) or []:
                            for call in getattr(m, "tool_calls", None) or []:
                                yield _sse(
                                    "tool_start",
                                    {"tool": call["name"], "args": call.get("args", {})},
                                )
                            if node == "tools" and getattr(m, "name", None):
                                yield _sse("tool_end", {"tool": m.name})

            yield _sse("done", {"thread_id": req.thread_id})

        except Exception as exc:
            log.exception("Streaming failed")
            yield _sse("error", {"detail": str(exc)})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/history/{thread_id}", response_model=HistoryResponse, tags=["chat"])
async def history(thread_id: str) -> HistoryResponse:
    """Replay a conversation straight from the checkpointer.

    Useful for proving persistence: restart the server, call this, and the
    messages are still here.
    """
    snapshot = await app.state.graph.aget_state(_thread_config(thread_id))
    messages = snapshot.values.get("messages", []) if snapshot.values else []
    return HistoryResponse(
        thread_id=thread_id,
        messages=[Turn(role=m.type, content=str(m.content)) for m in messages],
    )


@app.delete("/history/{thread_id}", tags=["chat"], status_code=204)
async def clear_history(thread_id: str) -> None:
    """Forget a conversation. Handy while testing."""
    await app.state.graph.checkpointer.adelete_thread(thread_id)
