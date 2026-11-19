from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class WebSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    index_path: Path = Path("data/index.bin")
    corpus_dir: Path = Path("data/corpus")
    default_top_k: int = 10
    max_top_k: int = 100


def get_settings() -> WebSettings:
    return WebSettings()
