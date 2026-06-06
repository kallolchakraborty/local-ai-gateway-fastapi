import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # FastAPI Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Model configuration
    MODEL_ID: str = "Qwen/Qwen2.5-1.5B-Instruct"
    DEVICE: str = "auto"
    MAX_NEW_TOKENS: int = 2048
    
    # Security
    API_KEY: Optional[str] = None
    
    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
