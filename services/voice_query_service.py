import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict

import requests
from google.cloud import storage, texttospeech
from pydub import AudioSegment
from speech_recognition import AudioFile, Recognizer

from config.settings import Settings
from services.query_service import process_query

logger = logging.getLogger(__name__)

settings = Settings()

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = settings.google_credentials_path_str
os.environ["GCP_BUCKET_NAME"] = settings.gcp_bucket_name

EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002500-\U00002BEF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0001F900-\U0001F9FF"
    "\U0001FA70-\U0001FAFF"
    "\U00002600-\U000026FF"
    "\U0001F700-\U0001F77F"
    "]+",
    flags=re.UNICODE
)


def download_and_convert_audio(audio_url: str, audio_auth: str) -> Path:
    headers = {"Authorization": f"Bearer {audio_auth}"}
    response = requests.get(audio_url, headers=headers, timeout=30)
    
    if response.status_code != 200:
        raise ValueError(f"Error downloading audio: {response.status_code}")
    
    with NamedTemporaryFile(delete=False, suffix=".ogg") as temp_ogg:
        temp_ogg.write(response.content)
        temp_ogg_path = Path(temp_ogg.name)
    
    try:
        with NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
            temp_wav_path = Path(temp_wav.name)
        
        audio = AudioSegment.from_file(str(temp_ogg_path), format="ogg")
        audio.export(str(temp_wav_path), format="wav")
        
        logger.info(f"Audio OGG converted to WAV: {temp_wav_path}")
        return temp_wav_path
    finally:
        if temp_ogg_path.exists():
            temp_ogg_path.unlink()


def speech_to_text(file_path: Path) -> str:
    recognizer = Recognizer()
    with AudioFile(str(file_path)) as source:
        audio_data = recognizer.record(source)
    return recognizer.recognize_google(audio_data, language="pt-BR")


def text_to_speech(text: str) -> Path:
    client = texttospeech.TextToSpeechClient()
    
    input_text = texttospeech.SynthesisInput(text=text)
    
    voice = texttospeech.VoiceSelectionParams(
        language_code="pt-BR",
        name="pt-BR-Neural2-C",
    )
    
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )
    response = client.synthesize_speech(
        input=input_text,
        voice=voice,
        audio_config=audio_config
    )

    with NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
        temp_file.write(response.audio_content)
        temp_file_path = Path(temp_file.name)
    
    logger.info(f"Audio response generated: {temp_file_path}")
    return temp_file_path


def upload_to_cloud_storage(file_path: Path, bucket_name: str, blob_name: str) -> str:
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    blob.upload_from_filename(str(file_path))
    blob.make_public()
    
    logger.info(f"File uploaded to GCP Storage: {blob.public_url}")
    return blob.public_url


def _remove_emojis(text: str) -> str:
    return EMOJI_PATTERN.sub("", text)


def voice_query(audio_url: str, audio_auth: str, message_context: str) -> Dict[str, Any]:
    audio_path = None
    agent_audio_path = None
    
    try:
        audio_path = download_and_convert_audio(audio_url, audio_auth)
        
        query_text = speech_to_text(audio_path)
        logger.info(f"Transcription: {query_text}")

        agent_response = process_query(query_text, message_context)
        logger.info(f"Agent response generated")
        
        agent_text = agent_response["response"]
        agent_text_clean = _remove_emojis(agent_text)

        agent_audio_path = text_to_speech(agent_text_clean)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        blob_name = f"agent_responses/{timestamp}.mp3"

        audio_link = upload_to_cloud_storage(
            agent_audio_path,
            settings.gcp_bucket_name,
            blob_name
        )

        return {
            "query_text": query_text,
            "response": agent_text,
            "sources": agent_response["sources"],
            "audio_link": audio_link
        }
    finally:
        if audio_path and audio_path.exists():
            audio_path.unlink()
        if agent_audio_path and agent_audio_path.exists():
            agent_audio_path.unlink()
