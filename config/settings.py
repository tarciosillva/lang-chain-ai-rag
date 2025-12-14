from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)
    
    openai_api_key: str = Field(..., description="OpenAI API key")
    chroma_path: Path = Field(default=Path("chroma"), description="Path to ChromaDB persistence directory")
    google_credentials_path: Path = Field(..., description="Path to Google Cloud credentials JSON file")
    gcp_bucket_name: str = Field(..., description="GCP bucket name for audio storage")
    
    @property
    def chroma_path_str(self) -> str:
        return str(self.chroma_path)
    
    @property
    def google_credentials_path_str(self) -> str:
        return str(self.google_credentials_path)
