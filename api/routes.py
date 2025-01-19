from fastapi import APIRouter, HTTPException
from models.request_models import QueryRequest,VoiceQueryRequest
from models.response_models import TextQueryResponse, VoiceQueryResponse
from services.query_service import process_query
from services.voice_query_service import voice_query

router = APIRouter()

@router.post("/query", response_model=TextQueryResponse)
def query_endpoint(request: QueryRequest):
    response = process_query(request.query_text, request.message_context)
    return response

@router.post("/voiceQuery", response_model=VoiceQueryResponse)
def query_endpoint(request: VoiceQueryRequest):
    response = voice_query(request.audio_url, request.audio_auth, request.message_context)
    return response