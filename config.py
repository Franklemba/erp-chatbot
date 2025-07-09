import os
import yaml
# from pydantic import BaseSettings
from pydantic_settings import BaseSettings
from typing import Any, Dict
from dotenv import load_dotenv

class Config(BaseSettings):
    docs_dir: str
    vectorstore_path: str
    hash_file: str
    embedding_model: str
    llm_provider: str
    llm_repo_id: str = None
    llm_temperature: float = 0.1
    llm_max_length: int = 512
    openai_model: str = None
    gemini_model: str = None
    groq_model: str = None
    huggingface_api_token: str = None
    openai_api_key: str = None
    google_api_key: str = None
    groq_api_key: str = None
    pinecone_api_key: str = None

    class Config:
        env_file = ".env"

    @classmethod
    def from_yaml(cls, yaml_path: str = "config.yaml"):
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        llm = data.get("llm", {})
        return cls(
            docs_dir=data.get("docs_dir", "erp_docs"),
            vectorstore_path=data.get("vectorstore_path", "vectorstore"),
            hash_file=data.get("hash_file", "vectorstore/.docs_hash"),
            embedding_model=data.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
            llm_provider=data.get("llm_provider", "huggingface"),
            llm_repo_id=llm.get("repo_id"),
            llm_temperature=llm.get("temperature", 0.1),
            llm_max_length=llm.get("max_length", 512),
            openai_model=llm.get("openai_model"),
            gemini_model=llm.get("gemini_model"),
            groq_model=llm.get("groq_model"),
        )

# Load .env for secrets
load_dotenv()

# Singleton config instance
config = Config.from_yaml()
config.huggingface_api_token = os.getenv("HUGGINGFACE_API_TOKEN")
config.openai_api_key = os.getenv("OPENAI_API_KEY")
config.google_api_key = os.getenv("GOOGLE_API_KEY") 
config.groq_api_key = os.getenv("GROQ_API_KEY")
config.pinecone_api_key = os.getenv("PINECONE_API_KEY")
