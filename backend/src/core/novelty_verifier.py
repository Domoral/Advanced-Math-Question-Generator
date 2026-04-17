"""
Novelty Verifier for Math Question Generation

This module provides novelty scoring for generated math questions by comparing
them against existing questions in the vector database using cosine similarity.

Core Formula:
    novelty = 1 - [β·max_sim + (1-β)·H_topk]^α

Where:
    - max_sim: Maximum cosine similarity with any question in the database
    - H_topk: Harmonic mean of top-k similarities
    - β: Weight for max_sim (default 0.7)
    - α: Non-linear stretch parameter (default 4)
    - k: Number of nearest neighbors to consider (default 5)
"""

import numpy as np
from typing import TYPE_CHECKING, Optional, List, Dict
from sentence_transformers import SentenceTransformer
import chromadb
from pathlib import Path

if TYPE_CHECKING:
    from question_node import QuestionNode


def get_project_root() -> Path:
    """Get the project root directory."""
    core_dir = Path(__file__).parent
    backend_dir = core_dir.parent
    return backend_dir.parent


def get_default_vector_db_path() -> str:
    """Get the default vector database path relative to project root."""
    return str(get_project_root() / "backend" / "data" / "vector_db")


def get_default_model_path() -> str:
    """Get the default embedding model path relative to project root."""
    return str(get_project_root() / "backend" / "data" / "embedding" / "bge-base-zh-v1.5")


class NoveltyVerifier:
    """
    Verifier for assessing the novelty of generated math questions.

    Uses embedding-based similarity search against a vector database of existing
    questions to compute a novelty score.
    """

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: str = "math_questions",
        model_name: Optional[str] = None,
        alpha: float = 4.0,
        beta: float = 0.7,
        k: int = 5
    ):
        """
        Initialize the novelty verifier.

        Args:
            persist_directory: Path to ChromaDB persistence directory
            collection_name: Name of the collection in ChromaDB
            model_name: Embedding model name or path
            alpha: Non-linear stretch parameter (α > 1)
            beta: Weight for max_sim (0 < β < 1)
            k: Number of nearest neighbors to consider
        """
        if persist_directory is None:
            persist_directory = get_default_vector_db_path()

        if model_name is None:
            model_name = get_default_model_path()

        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.alpha = alpha
        self.beta = beta
        self.k = k

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_collection(collection_name)

        # Initialize embedding model
        self.model = SentenceTransformer(model_name)

        print(f"[NoveltyVerifier] Loaded collection '{collection_name}' with {self.collection.count()} documents")

    def _build_query_text(self, node: 'QuestionNode') -> str:
        """
        Build query text from node information.

        Args:
            node: The question node containing problem information

        Returns:
            Concatenated string of problem information
        """
        parts = []

        # Add question content
        if node.question:
            parts.append(f"题目: {node.question}")

        # Add question type
        if hasattr(node, 'question_type') and node.question_type:
            parts.append(f"题型: {node.question_type}")

        # Add knowledge points
        if node.integrated_knowledge:
            kp_str = ", ".join(sorted(node.integrated_knowledge))
            parts.append(f"知识点: {kp_str}")

        # Add solving steps if available
        if hasattr(node, 'solving_steps') and node.solving_steps:
            parts.append(f"解题步骤: {node.solving_steps}")

        return "\n".join(parts)

    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding vector for text.

        Args:
            text: Input text to embed

        Returns:
            Embedding vector as numpy array
        """
        embedding = self.model.encode([text])
        return embedding

    def _search_similar(self, query_embedding: np.ndarray, top_k: int) -> List[Dict]:
        """
        Search for similar questions in the vector database.

        Args:
            query_embedding: The embedding vector to search
            top_k: Number of results to return

        Returns:
            List of search results with similarity scores
        """
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        formatted_results = []
        for i in range(len(results["ids"][0])):
            # Convert L2 distance to cosine similarity
            # Cosine similarity = 1 - (L2_distance^2 / 2) for normalized vectors
            l2_dist = results["distances"][0][i]
            cosine_sim = 1 - (l2_dist ** 2) / 2
            cosine_sim = max(0, min(1, cosine_sim))  # Clamp to [0, 1]

            formatted_results.append({
                "id": results["ids"][0][i],
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "cosine_similarity": cosine_sim
            })

        return formatted_results

    def _harmonic_mean(self, values: List[float]) -> float:
        """
        Calculate harmonic mean of a list of values.

        Args:
            values: List of positive values

        Returns:
            Harmonic mean
        """
        if not values or any(v <= 0 for v in values):
            return 0.0

        n = len(values)
        return n / sum(1.0 / v for v in values)

    def compute_novelty(
        self,
        node: 'QuestionNode',
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        k: Optional[int] = None
    ) -> float:
        """
        Compute novelty score for a question node.

        Formula: novelty = 1 - [β·max_sim + (1-β)·H_topk]^α

        Args:
            node: The question node to evaluate
            alpha: Non-linear stretch parameter (overrides default)
            beta: Weight for max_sim (overrides default)
            k: Number of nearest neighbors (overrides default)

        Returns:
            Novelty score between 0 and 1 (higher = more novel)
        """
        # Use instance defaults if not specified
        alpha = alpha if alpha is not None else self.alpha
        beta = beta if beta is not None else self.beta
        k = k if k is not None else self.k

        # Build query text
        query_text = self._build_query_text(node)
        print(f"[NoveltyVerifier] Query text length: {len(query_text)} chars")

        # Get embedding
        query_embedding = self._get_embedding(query_text)

        # Search for similar questions
        search_k = max(k, 5)  # Get at least k results
        results = self._search_similar(query_embedding, top_k=search_k)

        if not results:
            print("[NoveltyVerifier] No similar questions found, returning max novelty")
            return 1.0

        # Extract top-k similarities
        top_k_results = results[:k]
        similarities = [r["cosine_similarity"] for r in top_k_results]

        # Calculate max_sim
        max_sim = max(similarities)

        # Calculate H_topk (harmonic mean of top-k similarities)
        H_topk = self._harmonic_mean(similarities)

        # Compute combined similarity
        combined_sim = beta * max_sim + (1 - beta) * H_topk

        # Apply non-linear stretch and compute novelty
        novelty = 1 - (combined_sim ** alpha)

        # Clamp to [0, 1]
        novelty = max(0.0, min(1.0, novelty))

        print(f"[NoveltyVerifier] max_sim={max_sim:.4f}, H_topk={H_topk:.4f}, "
              f"combined={combined_sim:.4f}, novelty={novelty:.4f}")

        return novelty


def verify_novelty(
    node: 'QuestionNode',
    persist_directory: Optional[str] = None,
    alpha: float = 4.0,
    beta: float = 0.7,
    k: int = 5
) -> float:
    """
    Convenience function to compute novelty score for a question node.

    Args:
        node: The question node to evaluate
        persist_directory: Path to ChromaDB persistence directory
        alpha: Non-linear stretch parameter
        beta: Weight for max_sim
        k: Number of nearest neighbors

    Returns:
        Novelty score between 0 and 1
    """
    verifier = NoveltyVerifier(
        persist_directory=persist_directory,
        alpha=alpha,
        beta=beta,
        k=k
    )
    return verifier.compute_novelty(node)
