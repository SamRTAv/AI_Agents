"""Graph nodes.

Nodes are built by factory functions rather than defined at module level so the
LLM (and the bound tool list) can be injected at startup instead of being
created on import.
"""

from __future__ import annotations

import logging

from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.types import interrupt

from app.prompts import SYSTEM_PROMPT
from app.state import ChatState

log = logging.getLogger("copilot.nodes")

# Tools that CHANGE stored data. Reads (summarize, list_expenses,
# search_finance_docs) run freely; only writes need a human.
MUTATING_TOOLS = {"add_expense"}

APPROVE_WORDS = {"yes", "y", "approve", "approved", "ok", "okay", "confirm", "haan"}


def make_chat_node(llm):
    """The reasoning node — decides whether to answer or call a tool."""

    async def chat_node(state: ChatState) -> dict:
        messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
        response = await llm.ainvoke(messages)
        return {"messages": [response]}

    return chat_node


def make_approval_node():
    """Gate mutating tool calls behind a human decision.

    This has to live here, in our graph, rather than inside the MCP server:
    interrupt() suspends the LangGraph run using the checkpointer, and
    mcp_server/main.py has no idea LangGraph exists.
    """

    def approval_node(state: ChatState) -> dict:
        last = state["messages"][-1]
        calls = getattr(last, "tool_calls", None) or []

        needs_approval = [c for c in calls if c["name"] in MUTATING_TOOLS]
        if not needs_approval:
            # Read-only: fall straight through to the tools node.
            return {}

        call = needs_approval[0]

        # Everything from here suspends the graph. The payload is what the API
        # hands the user; state is checkpointed to disk, so the pause survives
        # a server restart.
        decision = interrupt(
            {
                "tool": call["name"],
                "args": call.get("args", {}),
                "question": (
                    f"Approve {call['name']} with {call.get('args', {})}? (yes/no)"
                ),
            }
        )

        approved = str(decision).strip().lower() in APPROVE_WORDS
        if approved:
            log.info("Approved: %s", call["name"])
            return {}

        log.info("Declined: %s", call["name"])
        # Every tool_call the model made must be answered with a ToolMessage —
        # providers reject the next turn otherwise. So instead of running the
        # tools, answer each call with a refusal and let the model respond.
        return {
            "messages": [
                ToolMessage(
                    content=(
                        f"The user declined this action. {c['name']} was not "
                        "executed and nothing was changed."
                    ),
                    tool_call_id=c["id"],
                )
                for c in calls
            ]
        }

    return approval_node


def route_after_approval(state: ChatState) -> str:
    """Approved (or nothing to approve) -> run tools. Declined -> back to model.

    Distinguished by what approval_node left at the end of the state: a
    ToolMessage means it injected refusals.
    """
    return "chat_node" if isinstance(state["messages"][-1], ToolMessage) else "tools"
