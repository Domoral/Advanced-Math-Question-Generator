"""
MCTS Node implementation for Advanced Math Question Generation

This module defines the QuestionNode class used in the MCTS tree for generating
integrated mathematics problems by progressively combining knowledge points.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Set
import math


class QuestionNode:
    """
    A node in the MCTS tree representing a mathematics problem at a certain
    stage of construction.
    
    Each node contains:
    - A problem statement (may be empty for root)
    - Set of knowledge points already integrated
    - Queue of knowledge points waiting to be integrated
    - MCTS statistics (Q, N) for node selection
    - Tree structure references (parent, children)
    """
    
    def __init__(
        self,
        question: str = "",
        integrated_knowledge: Optional[Set[str]] = None,
        waiting_knowledge: Optional[List[str]] = None,
        parent: Optional['QuestionNode'] = None
    ):
        """
        Initialize a QuestionNode.
        
        Args:
            question: The problem statement at this node (empty for root)
            integrated_knowledge: Set of knowledge points already in the question
            waiting_knowledge: Ordered list of knowledge points to be added
            parent: Parent node in the MCTS tree
        """
        self.question = question
        self.integrated_knowledge = integrated_knowledge or set()
        self.waiting_knowledge = waiting_knowledge or []
        self.parent = parent
        
        # MCTS statistics
        self.Q = 0.0  # Total reward (sum of quality scores)
        self.N = 0    # Visit count
        
        # Tree structure
        self.children: List['QuestionNode'] = []
        
        # Node metadata (optional, for debugging/analysis)
        self.metadata = {
            'generation_step': 0,
            'merge_strategy': None,  # How this node was generated from parent
            'quality_details': None,  # Detailed evaluation from verifier
        }
    
    def is_terminal(self) -> bool:
        """
        Check if this is a terminal node (no more knowledge points to add).
        
        Returns:
            True if waiting_knowledge is empty, False otherwise
        """
        return len(self.waiting_knowledge) == 0
    
    def is_fully_expanded(self) -> bool:
        """
        Check if all possible children have been generated.
        
        In practice, this depends on how many aggregation strategies
        and reference questions we want to try for the next knowledge point.
        
        Returns:
            True if children list is non-empty (simplified assumption)
        """
        return len(self.children) > 0
    
    def next_knowledge_point(self) -> Optional[str]:
        """
        Get the next knowledge point to be integrated.
        
        Returns:
            The first element of waiting_knowledge, or None if empty
        """
        if self.waiting_knowledge:
            return self.waiting_knowledge[0]
        return None
    
    def remaining_knowledge(self) -> List[str]:
        """
        Get the list of knowledge points still waiting to be integrated.
        
        Returns:
            List of remaining knowledge points (excluding the next one)
        """
        if len(self.waiting_knowledge) > 1:
            return self.waiting_knowledge[1:]
        return []
    
    def uct_score(self, exploration_weight: float = 1.414) -> float:
        """
        Calculate the UCT (Upper Confidence Bound for Trees) score.
        
        UCT = Q/N + c * sqrt(ln(N_parent) / N)
        
        Args:
            exploration_weight: The 'c' parameter balancing exploration vs exploitation
            
        Returns:
            UCT score (higher is better)
        """
        if self.N == 0:
            return float('inf')  # Prioritize unexplored nodes
        
        if self.parent is None:
            # Root node has no parent N, use only exploitation term
            return self.Q / self.N
        
        parent_N = self.parent.N
        if parent_N == 0:
            return float('inf')
        
        exploitation = self.Q / self.N
        exploration = exploration_weight * math.sqrt(math.log(parent_N) / self.N)
        
        return exploitation + exploration
    
    def average_reward(self) -> float:
        """
        Get the average reward (quality) of this node.
        
        Returns:
            Q/N if visited, 0 otherwise
        """
        if self.N == 0:
            return 0.0
        return self.Q / self.N
    
    def update(self, reward: float):
        """
        Update node statistics after a rollout.
        
        Args:
            reward: The quality score obtained from simulation
        """
        self.N += 1
        self.Q += reward
    
    def add_child(self, child: 'QuestionNode'):
        """
        Add a child node to this node.
        
        Args:
            child: The child QuestionNode to add
        """
        self.children.append(child)
        child.parent = self
    
    def get_path_from_root(self) -> List['QuestionNode']:
        """
        Get the path from root to this node.
        
        Returns:
            List of nodes from root to this node (inclusive)
        """
        path = []
        current = self
        while current is not None:
            path.append(current)
            current = current.parent
        return list(reversed(path))
    
    def depth(self) -> int:
        """
        Get the depth of this node in the tree.
        
        Returns:
            Number of edges from root to this node
        """
        if self.parent is None:
            return 0
        return self.parent.depth() + 1
    
    def __hash__(self) -> int:
        """
        Make node hashable for use in sets/dicts.
        Uses question content and knowledge sets for identity.
        """
        return hash((
            self.question,
            frozenset(self.integrated_knowledge),
            tuple(self.waiting_knowledge)
        ))
    
    def __eq__(self, other: object) -> bool:
        """
        Check equality with another node.
        """
        if not isinstance(other, QuestionNode):
            return False
        return (
            self.question == other.question and
            self.integrated_knowledge == other.integrated_knowledge and
            self.waiting_knowledge == other.waiting_knowledge
        )
    
    def __repr__(self) -> str:
        """
        String representation for debugging.
        """
        return (
            f"QuestionNode("
            f"depth={self.depth()}, "
            f"integrated={len(self.integrated_knowledge)}, "
            f"waiting={len(self.waiting_knowledge)}, "
            f"Q={self.Q:.2f}, "
            f"N={self.N}, "
            f"avg_reward={self.average_reward():.2f}"
            f")"
        )


class QuestionMCTS(ABC): 
    """
    Abstract base class for MCTS algorithm adapted for question generation.
    
    Subclasses must implement the abstract methods to define:
    - How to select nodes (select)
    - How to expand nodes (expand)
    - How to simulate/rollout (simulate)
    - How to backpropagate rewards (backpropagate)
    """
    
    def __init__(self, exploration_weight: float = 1.414):
        """
        Initialize the MCTS searcher.
        
        Args:
            exploration_weight: The 'c' parameter for UCT calculation
        """
        self.exploration_weight = exploration_weight
    
    @abstractmethod
    def select(self, root: QuestionNode) -> QuestionNode:
        """
        Select a node to expand using UCT strategy.
        
        Traverse the tree from root, selecting child with highest UCT score
        until reaching a node that is not fully expanded.
        
        Args:
            root: The root node of the tree
            
        Returns:
            The selected node for expansion
        """
        pass
    
    @abstractmethod
    def expand(self, node: QuestionNode) -> List[QuestionNode]:
        """
        Expand a node by generating its children.
        
        Generate multiple candidate problems by integrating the next
        knowledge point in different ways.
        
        Args:
            node: The node to expand
            
        Returns:
            List of generated child nodes
        """
        pass
    
    @abstractmethod
    def simulate(self, node: QuestionNode, alpha: float = 0.5) -> float:
        """
        Simulate/rollout from a node to get a reward.
        
        This implements a two-phase evaluation strategy:
        
        Case A: Terminal Node (waiting_knowledge is empty)
        - Directly evaluate node.question with verifier
        - reward = verifier_score
        
        Case B: Non-Terminal Node
        Phase 1: Evaluate Current Node (Intermediate Quality)
        - Use verifier to evaluate the quality of node's current question
        - This captures the quality of intermediate products
        - If quality >= save_threshold (5.0), save the question immediately
        
        Phase 2: Evaluate Partial Aggregation (Potential Quality)
        - Take waiting_knowledge[0:3] (at most first 3 remaining knowledge points)
        - Use generator to aggregate these into current question in one step
        - Use verifier to evaluate this virtual question
        - This tests the "potential" of the current path
        
        Phase 3: Combine Scores
        - reward = alpha * current_score + (1 - alpha) * potential_score
        - alpha (default 0.5): balances current quality vs future potential
        
        Args:
            node: The node to simulate from
            alpha: Weight for current_score (0.5 means equal weight)
            
        Returns:
            The combined reward (quality score) obtained
        """
        pass
    
    @abstractmethod
    def backpropagate(self, node: QuestionNode, reward: float):
        """
        Backpropagate the reward up the tree.
        
        Update Q and N statistics for the node and all its ancestors.
        
        Args:
            node: The node where simulation ended
            reward: The reward to backpropagate
        """
        pass
    
    @abstractmethod
    def search(self, root: QuestionNode, num_iterations: int) -> QuestionNode:
        """
        Run MCTS search for a given number of iterations.
        
        Args:
            root: The root node
            num_iterations: Number of MCTS iterations to run
            
        Returns:
            The best node found (typically highest average reward)
        """
        pass
