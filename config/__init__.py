# Configuration module for RAGCUN
"""
Configuration management for RAGCUN.

This module handles loading and managing configuration from environment variables
and configuration files.
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)


class Config:
    """Configuration class for RAGCUN."""
    
    # Embedding Model
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL", 
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "384"))
    
    # Language Model
    MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    
    # Retrieval
    TOP_K: int = int(os.getenv("TOP_K", "5"))
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
    
    # Data Paths
    DATA_DIR: Path = Path(os.getenv("DATA_DIR", "./data"))
    RAW_DATA_DIR: Path = Path(os.getenv("RAW_DATA_DIR", "./data/raw"))
    PROCESSED_DATA_DIR: Path = Path(os.getenv("PROCESSED_DATA_DIR", "./data/processed"))
    EMBEDDINGS_DIR: Path = Path(os.getenv("EMBEDDINGS_DIR", "./data/embeddings"))
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    @classmethod
    def ensure_directories(cls) -> None:
        """Ensure all required directories exist."""
        for dir_path in [
            cls.DATA_DIR,
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.EMBEDDINGS_DIR,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)


# Create a default config instance
config = Config()
