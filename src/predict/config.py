"""Settings from the environment and `.env`. Constructed by the CLI when it needs them, never at import."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_HOME = Path.home() / ".predict"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    football_data_org_token: str
    odds_api_key: str = ""
    typesafe_api_key: str = ""
    claude_model: str = "claude-opus-5-5"
    jev_model: str = "jev-1.13.0"
    predict_home: Path = DEFAULT_HOME
