"""Graph assembly.

Shape with tools:

    START -> chat_node -> (tools_condition) -> approval -> tools -> chat_node
                                    |             |                    |
                                    v             +--(declined)--------+
                                   END

tools_condition still decides *whether* a tool was requested; the only change
is that its "tools" branch is remapped to the approval node, which either falls
through to the real tools or injects refusals and hands control back.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from app.nodes import make_approval_node, make_chat_node, route_after_approval
from app.state import ChatState


def build_graph(llm, checkpointer, tools: list | None = None):
    tools = tools or []

    # Binding is what tells the model which tools exist and their schemas.
    bound_llm = llm.bind_tools(tools) if tools else llm

    builder = StateGraph(ChatState)
    builder.add_node("chat_node", make_chat_node(bound_llm))
    builder.add_edge(START, "chat_node")

    if not tools:
        builder.add_edge("chat_node", END)
        return builder.compile(checkpointer=checkpointer)

    builder.add_node("approval", make_approval_node())
    builder.add_node("tools", ToolNode(tools))

    # tools_condition returns "tools" or END; remap "tools" through approval.
    builder.add_conditional_edges(
        "chat_node",
        tools_condition,
        {"tools": "approval", END: END},
    )
    builder.add_conditional_edges(
        "approval",
        route_after_approval,
        {"tools": "tools", "chat_node": "chat_node"},
    )
    # The loop back is what lets the model reason again after a tool result —
    # that is how it chains summarize into search_finance_docs.
    builder.add_edge("tools", "chat_node")

    return builder.compile(checkpointer=checkpointer)
