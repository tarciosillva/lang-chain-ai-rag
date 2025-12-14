import logging
from fastapi import APIRouter, HTTPException, status

from models.request_models import QueryRequest, VoiceQueryRequest
from models.response_models import TextQueryResponse, VoiceQueryResponse
from services.query_service import process_query
from services.voice_query_service import voice_query

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["queries"])


@router.post(
    "/query",
    response_model=TextQueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Process text query",
    description="Process a text query and return AI-generated response with sources"
)
async def query_endpoint(request: QueryRequest) -> TextQueryResponse:
    try:
        response = process_query(request.query_text, request.message_context)
        return TextQueryResponse(**response)
    except Exception as e:
        logger.error(f"Error in query endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing query"
        )


@router.post(
    "/voice-query",
    response_model=VoiceQueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Process voice query",
    description="Process a voice query from audio URL and return AI-generated response with audio"
)
async def voice_query_endpoint(request: VoiceQueryRequest) -> VoiceQueryResponse:
    try:
        response = voice_query(
            request.audio_url,
            request.audio_auth,
            request.message_context
        )
        return VoiceQueryResponse(**response)
    except ValueError as e:
        logger.error(f"Validation error in voice query endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error in voice query endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing voice query"
        )