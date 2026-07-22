"""
Embedding Model Manager (SiliconFlow API)

Singleton pattern implementation to ensure only one embedding client instance
exists in memory across RAGRetriever and NoveltyVerifier.

Uses SiliconFlow Embedding API (OpenAI-compatible) instead of local SentenceTransformer.
"""

from typing import Optional, List, Union
from pathlib import Path
import threading
import os
import time
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI


def get_backend_dir() -> Path:
    """Get the backend directory.

    This file is at backend/src/core/embedding_manager.py,
    so we need to go up 2 levels to get backend.
    """
    core_dir = Path(__file__).parent
    return core_dir.parent.parent


def _load_config():
    """Load embedding API config from .env file."""
    env_path = Path(__file__).parent / ".env"
    load_dotenv(dotenv_path=env_path)
    return {
        "api_key": os.getenv("SILICONFLOW_API_KEY", ""),
        "model": os.getenv("SILICONFLOW_MODEL", "Qwen/Qwen3-Embedding-4B"),
        "base_url": os.getenv("SILICONFLOW_API_BASE", "https://api.siliconflow.cn/v1/embeddings"),
    }


class SiliconFlowEmbedder:
    """
    OpenAI-compatible wrapper for SiliconFlow Embedding API.

    Provides a SentenceTransformer-like .encode() interface so that
    existing code (rag_retriever, novelty_verifier, build_index)
    can use it as a drop-in replacement.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        config = _load_config()
        self.api_key = api_key or config["api_key"]
        self.model = model or config["model"]
        self.base_url = base_url or config["base_url"]

        if not self.api_key:
            raise ValueError(
                "SILICONFLOW_API_KEY is not set. "
                "Please configure it in backend/src/core/.env"
            )

        # Strip /embeddings suffix if present, OpenAI client auto-appends it
        clean_base = self.base_url.rstrip("/")
        if clean_base.endswith("/embeddings"):
            clean_base = clean_base[:-len("/embeddings")]

        self.client = OpenAI(api_key=self.api_key, base_url=clean_base)
        print(f"[SiliconFlowEmbedder] Initialized: model={self.model}, base_url={clean_base}")

    def encode(
        self,
        sentences: Union[str, List[str]],
        convert_to_numpy: bool = True,
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Encode text(s) into embedding vectors via SiliconFlow API.

        Args:
            sentences: A single string or list of strings to encode.
            convert_to_numpy: Always True for backwards compatibility (ignored).
            batch_size: Number of texts to encode per API call.

        Returns:
            numpy array of shape (n, embedding_dim) with float32 dtype.
        """
        if isinstance(sentences, str):
            sentences = [sentences]

        all_embeddings = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            result = self._call_api(batch)
            all_embeddings.extend(result)

        return np.array(all_embeddings, dtype=np.float32)

    def _call_api(self, texts: List[str]) -> List[List[float]]:
        """Call SiliconFlow embeddings API with retry logic."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=texts,
                )
                return [item.embedding for item in response.data]
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    print(f"[SiliconFlowEmbedder] API error (attempt {attempt+1}/{max_retries}), "
                          f"retrying in {wait}s: {e}")
                    time.sleep(wait)
                else:
                    raise RuntimeError(
                        f"SiliconFlow API call failed after {max_retries} attempts: {e}"
                    ) from e

        return []  # unreachable


class EmbeddingManager:
    """
    Singleton manager for embedding client.

    Ensures only one SiliconFlowEmbedder instance exists in memory,
    shared across RAGRetriever and NoveltyVerifier.
    """

    _instance: Optional['EmbeddingManager'] = None
    _lock = threading.Lock()

    def __new__(cls, model_path: Optional[str] = None):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_path: Optional[str] = None):
        if self._initialized:
            return

        # model_path is kept for backwards compatibility (ignored)
        self._embedder: Optional[SiliconFlowEmbedder] = None
        self._ref_count = 0
        self._initialized = True

    def get_model(self):
        """
        Get the shared embedding client instance.

        Returns:
            SiliconFlowEmbedder instance (has .encode() method)
        """
        if self._embedder is None:
            print("[EmbeddingManager] Connecting to SiliconFlow embedding API...")
            self._embedder = SiliconFlowEmbedder()
            print("[EmbeddingManager] API client ready")

        self._ref_count += 1
        print(f"[EmbeddingManager] Model reference count: {self._ref_count}")
        return self._embedder

    def release(self):
        """Release a reference to the client."""
        self._ref_count = max(0, self._ref_count - 1)
        print(f"[EmbeddingManager] Model reference count: {self._ref_count}")

    def is_loaded(self) -> bool:
        """Check if client is currently initialized."""
        return self._embedder is not None

    def get_ref_count(self) -> int:
        """Get current reference count."""
        return self._ref_count


def get_embedding_manager(model_path: Optional[str] = None) -> EmbeddingManager:
    """
    Get the singleton embedding manager instance.

    Args:
        model_path: Ignored (kept for backwards compatibility)

    Returns:
        EmbeddingManager singleton instance
    """
    return EmbeddingManager(model_path)


def get_shared_model(model_path: Optional[str] = None):
    """
    Convenience function to get the shared embedding client.

    Args:
        model_path: Ignored (kept for backwards compatibility)

    Returns:
        SiliconFlowEmbedder instance with .encode() method
    """
    manager = get_embedding_manager(model_path)
    return manager.get_model()
