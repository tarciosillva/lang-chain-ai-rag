from pydantic import BaseModel

class QueryRequest(BaseModel):
    query_text: str
    message_context: str

class VoiceQueryRequest(BaseModel):
    audio_url: str
    audio_auth:str
    message_context: str