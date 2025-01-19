import requests
import os
from tempfile import NamedTemporaryFile
from pydub import AudioSegment
from google.cloud import texttospeech
from google.cloud import storage
from speech_recognition import Recognizer, AudioFile
from .query_service import process_query
from config.settings import Settings
import speech_recognition as sr
import re
from datetime import datetime, timezone

settings = Settings()

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = settings.google_credentials_path
os.environ["GCP_BUCKET_NAME"] = settings.gcp_bucket_name

def download_and_convert_audio(audio_url:str, audio_auth :str):
    headers = {"Authorization": f"Bearer {audio_auth}"}
    response = requests.get(audio_url, headers=headers)
    
    if response.status_code != 200:
        raise ValueError(f"Error to download audio: {response.status_code}")
    
    temp_ogg = NamedTemporaryFile(delete=False, suffix=".ogg")
    temp_ogg.write(response.content)
    temp_ogg.close()

    temp_wav = NamedTemporaryFile(delete=False, suffix=".wav")
    audio = AudioSegment.from_file(temp_ogg.name, format="ogg")
    audio.export(temp_wav.name, format="wav")
    
    print(f"Audio OGG converted to WAV: {temp_wav.name}")
    return temp_wav.name


def speech_to_text(file_path):
    recognizer = Recognizer()
    with AudioFile(file_path) as source:
        audio_data = recognizer.record(source)
    return recognizer.recognize_google(audio_data, language="pt-BR")

def text_to_speech(text):
    client = texttospeech.TextToSpeechClient()
    
    input_text = texttospeech.SynthesisInput(text=text)
    
    voice = texttospeech.VoiceSelectionParams(
        language_code="pt-BR",
        name="pt-BR-Neural2-C",  # Nome da voz desejada
    )
    
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
    response = client.synthesize_speech(input=input_text, voice=voice, audio_config=audio_config)

    temp_file = NamedTemporaryFile(delete=False, suffix=".mp3")
    with open(temp_file.name, "wb") as out:
        out.write(response.audio_content)
    
    print(f"Áudio of the response genereted: {temp_file.name}")
    return temp_file.name


def upload_to_cloud_storage(file_path, bucket_name, blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    blob.upload_from_filename(file_path)    
    blob.make_public()
    
    print(f"File send to Storage GCP: {blob.public_url}")
    return blob.public_url

def voice_query(audio_url, audio_auth, message_context):
    audio_path = download_and_convert_audio(audio_url, audio_auth)
    
    query_text = speech_to_text(audio_path)
    print(f"Transcrição: {query_text}")

    agent_response = process_query(query_text, message_context)
    print(f"Resposta do agente: {agent_response}")
    
    agent_text = agent_response["response"]
    
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # Emojis de emoticons
        "\U0001F300-\U0001F5FF"  # Símbolos e pictogramas
        "\U0001F680-\U0001F6FF"  # Transportes e símbolos relacionados
        "\U0001F1E0-\U0001F1FF"  # Bandeiras (códigos regionais)
        "\U00002500-\U00002BEF"  # Formas geométricas
        "\U00002702-\U000027B0"  # Símbolos miscelâneos
        "\U000024C2-\U0001F251"  # Diversos outros símbolos
        "\U0001F900-\U0001F9FF"  # Suplemento de emoticons adicionais
        "\U0001FA70-\U0001FAFF"  # Objetos adicionais
        "\U00002600-\U000026FF"  # Símbolos miscelâneos
        "\U0001F700-\U0001F77F"  # Alquimia e símbolos relacionados
        "]+",
        flags=re.UNICODE
    )

    agent_text_clean = emoji_pattern.sub(r"", agent_text)

    agent_audio_path = text_to_speech(agent_text_clean)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    blob_name = f"agent_responses/{timestamp}.mp3"

    audio_link = upload_to_cloud_storage(agent_audio_path, settings.gcp_bucket_name, blob_name)

    os.remove(audio_path)
    os.remove(agent_audio_path)

    return {
        "query_text": query_text,
        "response": agent_text,
        "sources": agent_response["sources"],
        "audio_link": audio_link
    }
