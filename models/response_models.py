from pydantic import BaseModel
from typing import List, Optional

class TextQueryResponse(BaseModel):
    response: str
    sources: List[Optional[str]]


class VoiceQueryResponse(BaseModel):
    query_text:str
    response: str
    audio_link:str
