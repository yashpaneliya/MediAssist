from pydantic import Field
from functools import lru_cache
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "MediAssist"
    DEBUG: bool = Field(default=False)
    API_VERSION: str = "v1"

    REDIS_HOST: str = Field(default="localhost")
    REDIS_PORT: int = Field(default=6379)
    REDIS_DB: int = Field(default=0)
    REDIS_PASSWORD: str | None = None
    REDIS_CACHE_TTL: int = Field(default=3600)  # 1 hour default cache TTL

    OPENAI_API_KEY: str = Field(..., env="OPENAI_API_KEY")
    OPENAI_API_BASE: str = Field(default="https://api.openai.com/v1")

    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    return Settings()
