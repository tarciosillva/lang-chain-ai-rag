from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    openai_api_key: str
    chroma_path: str = "chroma"
    google_credentials_path:str
    gcp_bucket_name:str
    class Config:
        env_file = ".env"
