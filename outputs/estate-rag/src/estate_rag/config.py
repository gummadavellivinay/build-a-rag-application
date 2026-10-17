import re
from pathlib import Path
from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def safe_id(value: str) -> str:
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", value):
        raise ValueError("IDs must contain 1-64 letters, digits, underscores or hyphens")
    return value


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    openai_api_key: SecretStr = SecretStr("")
    pinecone_api_key: SecretStr = SecretStr("")
    chat_model: str = "gpt-4.1-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    pinecone_index: str = "estate-dense-v1"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    rerank_model: str = "bge-reranker-v2-m3"
    corpus_revision: str = "v1"
    data_dir: Path = Path("data")
    ocr_languages: str = "eng+spa+fra"
    api_tokens: dict[str, str] = {}
    inventory_base_url: str = ""
    inventory_api_token: SecretStr = SecretStr("")
    candidate_k: int = 20
    final_k: int = 6
    max_pages: int = 100
    max_file_mb: int = 20

    @field_validator("corpus_revision")
    @classmethod
    def validate_revision(cls, value):
        return safe_id(value)

    def namespace(self, tenant: str) -> str:
        # Length delimiter prevents collisions between tenant and revision components.
        tenant = safe_id(tenant)
        return f"{len(tenant)}:{tenant}:{self.corpus_revision}"

    def snapshot(self, tenant: str) -> Path:
        return self.data_dir / "snapshots" / safe_id(tenant) / self.corpus_revision

    def require_cloud(self):
        if not self.openai_api_key.get_secret_value() or not self.pinecone_api_key.get_secret_value():
            raise ValueError("Set OPENAI_API_KEY and PINECONE_API_KEY in .env")
