"""Request/response models.

These are also the source of the Swagger documentation at /docs, so the field
descriptions and examples here are what you will actually read while testing.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    thread_id: str = Field(
        ...,
        description="Conversation id. Reuse it to continue a conversation; "
        "change it to start a fresh one.",
        examples=["demo-1"],
    )
    message: str = Field(
        ...,
        min_length=1,
        description="The user's message.",
        examples=["What is the 50/30/20 budgeting rule?"],
    )


class ResumeRequest(BaseModel):
    thread_id: str = Field(
        ...,
        description="The thread that is paused awaiting approval.",
        examples=["demo-1"],
    )
    decision: str = Field(
        ...,
        description="'yes' approves the pending action; anything else rejects it.",
        examples=["yes"],
    )


class ApprovalPayload(BaseModel):
    """What the graph was about to do when it paused."""

    tool: str | None = Field(None, description="Tool awaiting approval.")
    args: dict[str, Any] | None = Field(None, description="Arguments it would run with.")
    question: str = Field(..., description="Question to put to the human.")


class ChatResponse(BaseModel):
    status: Literal["completed", "approval_required"] = Field(
        ...,
        description="'completed' means `answer` is filled. 'approval_required' "
        "means the graph is paused and you must call /resume.",
    )
    thread_id: str
    answer: str | None = Field(None, description="Set when status is 'completed'.")
    approval: ApprovalPayload | None = Field(
        None, description="Set when status is 'approval_required'."
    )


class Turn(BaseModel):
    role: str = Field(..., description="human | ai | tool | system")
    content: str


class HistoryResponse(BaseModel):
    thread_id: str
    messages: list[Turn]


class HealthResponse(BaseModel):
    status: str
    llm_provider: str
    model: str
    tools_loaded: int
    tool_names: list[str] = Field(
        default_factory=list, description="Tools the agent can currently call."
    )
    langsmith_tracing: bool
    langsmith_project: str | None
