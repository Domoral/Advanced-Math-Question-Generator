"""
Embedding Model Manager

Singleton pattern implementation to ensure only one embedding model instance
exists in memory across RAGRetriever and NoveltyVerifier.
"""

from typing import Optional
from sentence_transformers import SentenceTransformer
from pathlib import Path
import threading
import gc


def get_backend_dir() -> Path:
    """Get the backend directory.

    This file is at backend/src/core/embedding_manager.py,
    so we need to go up 2 levels to get backend.
    """
    core_dir = Path(__file__).parent
    return core_dir.parent.parent


def get_default_model_path() -> str:
    """Get the default embedding model path relative to backend directory."""
    return str(get_backend_dir() / "data" / "embedding" / "bge-base-zh-v1.5")


class EmbeddingManager:
    """
    Singleton manager for embedding model.
    
    Ensures only one model instance exists in memory, shared across
    RAGRetriever and NoveltyVerifier.
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
            
        self.model_path = model_path or get_default_model_path()
        self._model: Optional[SentenceTransformer] = None
        self._ref_count = 0
        self._initialized = True
    
    def get_model(self) -> SentenceTransformer:
        """
        Get the shared embedding model instance.
        
        Loads the model on first call, returns cached instance thereafter.
        
        Returns:
            SentenceTransformer model instance
        """
        if self._model is None:
            print(f"[EmbeddingManager] Loading model from {self.model_path}")
            self._model = SentenceTransformer(self.model_path)
            print(f"[EmbeddingManager] Model loaded successfully")
        
        self._ref_count += 1
        print(f"[EmbeddingManager] Model reference count: {self._ref_count}")
        return self._model
    
    def release(self):
        """
        Release a reference to the model.
        
        When reference count reaches 0, the model is unloaded from memory.
        """
        self._ref_count = max(0, self._ref_count - 1)
        print(f"[EmbeddingManager] Model reference count: {self._ref_count}")
        
        if self._ref_count == 0 and self._model is not None:
            self._unload_model()
    
    def _unload_model(self):
        """Unload model from memory and free resources."""
        print("[EmbeddingManager] Unloading model from memory")
        
        del self._model
        self._model = None
        
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("[EmbeddingManager] CUDA cache cleared")
        except ImportError:
            pass
        
        print("[EmbeddingManager] Model unloaded")
    
    def is_loaded(self) -> bool:
        """Check if model is currently loaded."""
        return self._model is not None
    
    def get_ref_count(self) -> int:
        """Get current reference count."""
        return self._ref_count


def get_embedding_manager(model_path: Optional[str] = None) -> EmbeddingManager:
    """
    Get the singleton embedding manager instance.
    
    Args:
        model_path: Path to the embedding model (only used on first call)
        
    Returns:
        EmbeddingManager singleton instance
    """
    return EmbeddingManager(model_path)


def get_shared_model(model_path: Optional[str] = None) -> SentenceTransformer:
    """
    Convenience function to get the shared embedding model.
    
    Args:
        model_path: Path to the embedding model (only used on first call)
        
    Returns:
        SentenceTransformer model instance
    """
    manager = get_embedding_manager(model_path)
    return manager.get_model()
