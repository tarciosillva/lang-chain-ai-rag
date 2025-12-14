from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query_text: str = Field(..., min_length=1, description="User query text")
    message_context: str = Field(default="", description="Conversation context")


class VoiceQueryRequest(BaseModel):
    audio_url: str = Field(..., description="URL to audio file")
    audio_auth: str = Field(..., description="Bearer token for audio URL authentication")
    message_context: str = Field(default="", description="Conversation context")