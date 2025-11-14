"""Configurações do sistema de indexação"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class OracleSettings(BaseSettings):
    """Configurações do Oracle Database"""

    user: str = Field(..., description="Oracle username")
    password: str = Field(..., description="Oracle password")
    dsn: str = Field(..., description="Oracle DSN")
    table: str = Field(default="documents", description="Table name")

    model_config = SettingsConfigDict(env_prefix="ORACLE_")


class PineconeSettings(BaseSettings):
    """Configurações do Pinecone"""

    api_key: str = Field(..., description="Pinecone API key")
    environment: str = Field(..., description="Pinecone environment")
    index_name: str = Field(..., description="Pinecone index name")

    model_config = SettingsConfigDict(env_prefix="PINECONE_")


class OpenAISettings(BaseSettings):
    """Configurações do OpenAI"""

    api_key: str = Field(..., description="OpenAI API key")
    model: str = Field(default="gpt-4-turbo-preview", description="Model for context generation")
    embedding_model: str = Field(default="text-embedding-3-large", description="Embedding model")

    model_config = SettingsConfigDict(env_prefix="OPENAI_")


class AnthropicSettings(BaseSettings):
    """Configurações do Anthropic"""

    api_key: str | None = Field(default=None, description="Anthropic API key")
    model: str = Field(default="claude-3-5-sonnet-20241022", description="Claude model")

    model_config = SettingsConfigDict(env_prefix="ANTHROPIC_")


class ChunkingSettings(BaseSettings):
    """Configurações de chunking"""

    chunk_size: int = Field(default=1000, description="Default chunk size")
    chunk_overlap: int = Field(default=200, description="Overlap between chunks")
    max_chunk_size: int = Field(default=2000, description="Maximum chunk size")

    model_config = SettingsConfigDict(env_prefix="")


class ContextSettings(BaseSettings):
    """Configurações de geração de contexto"""

    use_llm_context: bool = Field(default=True, description="Enable LLM context generation")
    context_prompt_template: str = Field(default="default", description="Context prompt template")
    batch_size: int = Field(default=10, description="Batch size for processing")

    model_config = SettingsConfigDict(env_prefix="")


class LoggingSettings(BaseSettings):
    """Configurações de logging"""

    log_level: str = Field(default="INFO", description="Log level")
    log_file: str = Field(default="indexer.log", description="Log file path")

    model_config = SettingsConfigDict(env_prefix="")


class Settings(BaseSettings):
    """Configurações gerais do sistema"""

    oracle: OracleSettings = Field(default_factory=OracleSettings)
    pinecone: PineconeSettings = Field(default_factory=PineconeSettings)
    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    anthropic: AnthropicSettings = Field(default_factory=AnthropicSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    context: ContextSettings = Field(default_factory=ContextSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        env_nested_delimiter="__"
    )


def get_settings() -> Settings:
    """Retorna a instância de configurações"""
    return Settings()
