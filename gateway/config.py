from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    vllm_base_url: str = Field(default="http://127.0.0.1:19100")
    gateway_token: str = Field(default="local-dev-token")
    gateway_log_path: Path = Field(default=Path("./logs/gateway.jsonl"))
    gateway_port: int = Field(default=18080, ge=1, le=65535)
    gateway_timeout_seconds: float = Field(default=120.0, gt=0)

    model_config = SettingsConfigDict(
        env_prefix="",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()
