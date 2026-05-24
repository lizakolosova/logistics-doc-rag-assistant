from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False)

    gemini_api_key: str
    gemini_model: str = "gemini-2.5-flash-lite"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    chroma_host: str = "chromadb"
    chroma_port: int = 8000
    chroma_collection: str = "documents"

    postgres_url: str

    chunk_size: int = 512
    chunk_overlap: int = 50

    top_k: int = 10
    retrieval_top_k: int = 20
    rerank_top_k: int = 5

    max_file_size_mb: int = 50
    max_pages: int = 200

    eval_data_path: str = "/app/eval_data/legal_qa_golden.json"

    prompt_chunk_max_chars: int = 1500

settings = Settings()