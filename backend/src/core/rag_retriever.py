"""
RAG Retriever for MCTS Question Generation

Simple wrapper for ChromaDB vector search to retrieve reference examples
based on knowledge points, difficulty, and question type.
"""

import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import os
from pathlib import Path


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


class RAGRetriever:
    """
    Simple RAG retriever for math question generation.
    
    Loads ChromaDB collection and provides search method.
    """
    
    def __init__(
        self,
        persist_directory: str = None,
        collection_name: str = "math_questions",
        model_name: str = None
    ):
        """
        Initialize RAG retriever.
        
        Args:
            persist_directory: Path to ChromaDB persistence directory
            collection_name: Name of the collection
            model_name: Embedding model name or path
        """
        # Use default path if not specified
        if persist_directory is None:
            persist_directory = get_default_vector_db_path()
        
        if model_name is None:
            model_name = get_default_model_path()
        
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_collection(collection_name)
        
        # Initialize embedding model
        self.model = SentenceTransformer(model_name)
        
        print(f"[RAG] Loaded collection '{collection_name}' with {self.collection.count()} documents")
    
    def search(
        self,
        knowledge_points: List[str],
        question_type: str,
        difficulty_range: tuple,
        top_k: int = 3
    ) -> List[Dict]:
        """
        Search for reference examples based on query parameters.
        
        Args:
            knowledge_points: List of knowledge points to query
            question_type: Type of question (单选题/多选题/ etc.)
            difficulty_range: Tuple of (min_difficulty, max_difficulty)
            top_k: Number of results to return
            
        Returns:
            List of retrieved documents with metadata
        """
        # Build query string from knowledge points, question type, and difficulty
        query_parts = []
        
        # Add knowledge points
        if knowledge_points:
            query_parts.append(f"知识点: {', '.join(knowledge_points)}")
        
        # Add question type
        if question_type:
            query_parts.append(f"题型: {question_type}")
        
        # Add difficulty hint with more detailed mixed difficulty labels
        min_diff, max_diff = difficulty_range
        diff_label = ""
        if max_diff <= 0.3:
            diff_label = "困难"
        elif min_diff >= 0.7:
            diff_label = "简单"
        elif min_diff >= 0.3 and max_diff <= 0.7:
            diff_label = "中等"
        else:
            # Mixed difficulty - more detailed classification
            difficulty_levels = []
            if min_diff < 0.3:
                difficulty_levels.append("困难")
            if min_diff < 0.7 and max_diff > 0.3:
                difficulty_levels.append("中等")
            if max_diff > 0.7:
                difficulty_levels.append("简单")
            diff_label = "、".join(difficulty_levels)
        query_parts.append(f"难度: {diff_label}")
        
        query_text = "\n".join(query_parts)
        
        print(f"[RAG] Query: {query_text[:100]}...")
        
        # Generate embedding for query
        query_embedding = self.model.encode([query_text])
        
        # Build filter for metadata
        filter_conditions = {}
        
        # Filter by question type if provided
        if question_type:
            filter_conditions["question_type"] = question_type
        
        # Execute search
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k,
            where=filter_conditions if filter_conditions else None,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results["ids"][0])):
            formatted_results.append({
                "id": results["ids"][0][i],
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i]
            })
        
        print(f"[RAG] Retrieved {len(formatted_results)} reference examples")
        return formatted_results
    
    def format_references(self, results: List[Dict]) -> str:
        """
        Format retrieved results as reference text for generator.
        
        Args:
            results: List of retrieved documents
            
        Returns:
            Formatted reference text
        """
        if not results:
            return ""
        
        ref_texts = []
        ref_texts.append("=== 参考例题 ===\n")
        
        for i, result in enumerate(results, 1):
            ref_texts.append(f"\n--- 参考例题 {i} ---")
            ref_texts.append(result["document"])
            ref_texts.append("")
        
        ref_texts.append("=== 参考例题结束 ===\n")
        
        return "\n".join(ref_texts)


# Global singleton instance (lazy loaded)
_retriever_instance: Optional[RAGRetriever] = None


def get_retriever() -> RAGRetriever:
    """Get or create global RAG retriever instance."""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = RAGRetriever()
    return _retriever_instance


def retrieve_references(
    knowledge_points: List[str],
    question_type: str,
    difficulty_range: tuple,
    top_k: int = 3,
    use_rag: bool = True
) -> str:
    """
    Convenience function to retrieve and format references.
    
    Args:
        knowledge_points: List of knowledge points
        question_type: Type of question
        difficulty_range: Tuple of (min, max) difficulty
        top_k: Number of results
        use_rag: Whether to use RAG (if False, returns empty string)
        
    Returns:
        Formatted reference text or empty string if use_rag is False
    """
    if not use_rag:
        return ""
    
    try:
        retriever = get_retriever()
        results = retriever.search(knowledge_points, question_type, difficulty_range, top_k)
        return retriever.format_references(results)
    except Exception as e:
        print(f"[RAG] Error retrieving references: {e}")
        return ""
