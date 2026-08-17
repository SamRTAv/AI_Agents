"""Graph state.

`add_messages` is a reducer: when a node returns {"messages": [x]}, LangGraph
appends x to the existing list rather than replacing it. Combined with the
checkpointer, this is what makes the bot remember a conversation even though
each HTTP request carries only one new message.
"""

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
