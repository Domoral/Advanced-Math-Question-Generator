"""
Generation Logger for Math Question Generation

This module provides CSV logging functionality for tracking all generated questions
during a generation task, regardless of whether they meet the save threshold.

Each record contains:
- iteration: MCTS search iteration number
- type: 'real' or 'virtual' (real = actual saved question, virtual = intermediate node)
- novelty: novelty score from novelty verifier
- llm_score: LLM evaluation score
- is_saved: whether the question was saved (met the threshold)
"""

import csv
import os
from typing import Optional
from pathlib import Path


class GenerationLogger:
    """
    Logger for tracking question generation statistics.
    
    Creates a CSV file in the output directory and records every generated
    question with its evaluation metrics.
    """
    
    def __init__(self, output_dir: str):
        """
        Initialize the generation logger.
        
        Args:
            output_dir: Directory where the CSV file will be created
        """
        self.output_dir = output_dir
        self.csv_path = os.path.join(output_dir, "generation_log.csv")
        self._initialized = False
        
    def _init_csv(self):
        """Initialize CSV file with headers if not exists."""
        if self._initialized:
            return
            
        # Create directory if not exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Write headers if file doesn't exist or is empty
        if not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0:
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'iteration',      # MCTS search iteration number
                    'type',           # 'real' or 'virtual'
                    'novelty',        # novelty score
                    'llm_score',      # LLM evaluation score
                    'is_saved',       # 'true' or 'false'
                    'total_score'     # novelty + llm_score
                ])
        
        self._initialized = True
        print(f"[GenerationLogger] CSV日志初始化: {self.csv_path}")
    
    def log_question(
        self,
        iteration: int,
        question_type: str,
        novelty: float,
        llm_score: float,
        is_saved: bool
    ):
        """
        Log a generated question to the CSV file.
        
        Args:
            iteration: MCTS search iteration number
            question_type: 'real' or 'virtual'
            novelty: novelty score from novelty verifier
            llm_score: LLM evaluation score
            is_saved: whether the question met the save threshold
        """
        self._init_csv()
        
        total_score = novelty + llm_score
        
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                iteration,
                question_type,
                f"{novelty:.4f}",
                f"{llm_score:.4f}",
                'true' if is_saved else 'false',
                f"{total_score:.4f}"
            ])
        
        print(f"[GenerationLogger] 记录: iteration={iteration}, type={question_type}, novelty={novelty:.4f}, "
              f"llm_score={llm_score:.4f}, is_saved={is_saved}")
    
    def get_log_path(self) -> str:
        """Get the path to the CSV log file."""
        return self.csv_path


# Convenience function for creating a logger instance
def create_generation_logger(output_dir: str) -> GenerationLogger:
    """
    Create a new GenerationLogger instance.
    
    Args:
        output_dir: Directory where the CSV file will be created
        
    Returns:
        GenerationLogger instance
    """
    return GenerationLogger(output_dir)
