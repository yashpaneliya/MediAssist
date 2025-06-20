from pydantic import BaseModel

class AgentRequest(BaseModel):
    query: str
    session_id: str | None = None
    img_base64: str | None = None
    img_url: str | None = None
    drugs: list[str] | None = None

class AgentResponse(BaseModel):
    response: str
    session_id: str
    error: str | None = None
    status_code: int = 200  # Default to 200 OK