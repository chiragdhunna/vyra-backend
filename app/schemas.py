"""Pydantic schemas shared by the REST API and the realtime protocol."""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

Role = Literal["user", "assistant"]


class ChatTurn(BaseModel):
    role: Role
    content: str


class VisionContext(BaseModel):
    """What the phone's on-device face detection currently sees.

    Only tiny labels ever reach the backend — raw camera frames stay on the
    phone by design.
    """

    present: bool = False
    smiling: bool = False
    eyes_open: float = 1.0  # 0..1, 1 = wide awake (min across visible faces)


class ChatRequest(BaseModel):
    """Stateless chat: the app owns history and sends the recent window."""

    messages: List[ChatTurn] = Field(default_factory=list, description="Oldest first; last item is the new user message.")
    user_name: Optional[str] = None
    vision: Optional[VisionContext] = None


class ChatResponse(BaseModel):
    text: str
    emotion: str = "neutral"
    provider: str
    model: str


class ConfigResponse(BaseModel):
    name: str = "vyra-backend"
    version: str
    provider: str
    model: str
    stt: str
    emotions: List[str]
    auth_required: bool


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    provider: str
