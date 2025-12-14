from pydantic import BaseModel, Field
from typing import List


class TextQueryResponse(BaseModel):
    response: str = Field(..., description="AI generated response")
    sources: List[str] = Field(default_factory=list, description="Source documents used")


class VoiceQueryResponse(BaseModel):
    query_text: str = Field(..., description="Transcribed query text")
    response: str = Field(..., description="AI generated response")
    audio_link: str = Field(..., description="URL to generated audio response")
    sources: List[str] = Field(default_factory=list, description="Source documents used")
