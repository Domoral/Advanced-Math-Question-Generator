"""
MCTS Node implementation for Advanced Math Question Generation

This module defines the QuestionNode class used in the MCTS tree for generating
integrated mathematics problems by progressively combining knowledge points.
"""

from typing import List, Optional, Set
from llm_client import generator, verifier, extract_score, parse_generator_output
import math
import json
import os
from datetime import datetime


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
        Check if node has reached max children (3).
        
        Returns:
            True if children count >= 3
        """
        return len(self.children) >= 3
    
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


class QuestionMCTS:
    """
    MCTS algorithm implementation for math question generation.
    """
    
    def __init__(
        self,
        exploration_weight: float = 1.414,
        alpha: float = 0.5,
        save_threshold: float = 5.0,
        output_dir: str = "./output"
    ):
        """
        Initialize the MCTS searcher.
        
        Args:
            exploration_weight: The 'c' parameter for UCT calculation
            alpha: Weight for current_score in simulate (0.5 means equal weight)
            save_threshold: Quality threshold for saving questions (default 5.0)
            output_dir: Directory to save generated questions
        """
        self.exploration_weight = exploration_weight
        self.alpha = alpha
        self.save_threshold = save_threshold
        self.output_dir = output_dir
        self.leaf_count = 0  # Count of terminal nodes generated
        
        # Create output directory if not exists
        os.makedirs(output_dir, exist_ok=True)
    
    def select(self, root: QuestionNode) -> QuestionNode:
        """
        Select a node to expand using UCT strategy.
        
        Traverse from root, selecting child with highest UCT score
        until reaching a node that is not fully expanded.
        
        Args:
            root: The root node of the tree
            
        Returns:
            The selected node for expansion
        """
        node = root
        
        while node.is_fully_expanded() and not node.is_terminal():
            # Select child with highest UCT score
            if not node.children:
                break
            node = max(node.children, key=lambda c: c.uct_score(self.exploration_weight))
        
        return node
    
    def expand(self, node: QuestionNode) -> QuestionNode:
        """
        Expand a node by generating a new child.
        
        Pop the first knowledge point from waiting_knowledge,
        use generator to create a new integrated problem.
        
        Args:
            node: The node to expand
            
        Returns:
            The newly generated child node
        """
        if node.is_terminal():
            raise RuntimeError("Cannot expand terminal node")
        
        # Get next knowledge point
        new_skill = node.waiting_knowledge[0]
        remaining = node.waiting_knowledge[1:]
        
        # Generate new problem using LLM
        try:
            raw_output = generator(node, new_skill)
            parsed = parse_generator_output(raw_output)
            
            # Create child node
            child = QuestionNode(
                question=parsed.get('problem_statement', ''),
                integrated_knowledge=node.integrated_knowledge | {new_skill},
                waiting_knowledge=remaining,
                parent=node
            )
            
            # Copy metadata
            child.metadata['generation_step'] = node.depth() + 1
            child.metadata['merge_strategy'] = parsed.get('integration_rationale', '')
            
            node.add_child(child)
            return child
            
        except Exception as e:
            # If generation fails, create a fallback child with error marker
            child = QuestionNode(
                question=f"[Generation Error: {str(e)}]",
                integrated_knowledge=node.integrated_knowledge | {new_skill},
                waiting_knowledge=remaining,
                parent=node
            )
            node.add_child(child)
            return child
    
    def simulate(self, node: QuestionNode) -> float:
        """
        Simulate/rollout from a node to get a reward.
        
        Case A: Terminal Node - directly evaluate with verifier
        Case B: Non-Terminal Node - two-phase evaluation
        
        Args:
            node: The node to simulate from
            
        Returns:
            The combined reward (quality score) obtained
        """
        # Case A: Terminal node
        if node.is_terminal():
            try:
                verifier_output = verifier(node)
                score = extract_score(verifier_output)
                if score is None:
                    score = 0.0
                
                # Save if quality meets threshold
                if score >= self.save_threshold:
                    self.save_question(node, score)
                
                return score
            except Exception:
                return 0.0
        
        # Case B: Non-terminal node - two-phase evaluation
        # Phase 1: Evaluate current node
        try:
            verifier_output = verifier(node)
            current_score = extract_score(verifier_output)
            if current_score is None:
                current_score = 0.0
            
            # Save if quality meets threshold
            if current_score >= self.save_threshold:
                self.save_question(node, current_score)
        except Exception:
            current_score = 0.0
        
        # Phase 2: Evaluate potential (partial aggregation)
        try:
            # Take at most first 3 remaining knowledge points
            knowledge_to_test = node.waiting_knowledge[:3]
            if knowledge_to_test:
                # Convert list to combined string
                combined_skill = "、".join(knowledge_to_test)
                
                # Generate virtual question
                virtual_output = generator(node, combined_skill)
                virtual_parsed = parse_generator_output(virtual_output)
                
                # Create temporary node for verification
                temp_node = QuestionNode(
                    question=virtual_parsed.get('problem_statement', ''),
                    integrated_knowledge=node.integrated_knowledge | set(knowledge_to_test),
                    waiting_knowledge=[]  # Virtual node is terminal
                )
                
                # Verify virtual question
                virtual_verifier_output = verifier(temp_node)
                potential_score = extract_score(virtual_verifier_output)
                if potential_score is None:
                    potential_score = 0.0
            else:
                potential_score = current_score
        except Exception:
            potential_score = 0.0
        
        # Phase 3: Combine scores
        reward = self.alpha * current_score + (1 - self.alpha) * potential_score
        return reward
    
    def save_question(self, node: QuestionNode, score: float):
        """
        Save a high-quality question to JSON file.
        
        Args:
            node: The QuestionNode containing the question to save
            score: The quality score from verifier (>= save_threshold)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = "terminal" if node.is_terminal() else "intermediate"
        filename = f"{prefix}_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        data = {
            "question": node.question,
            "integrated_knowledge": sorted(list(node.integrated_knowledge)),
            "quality_score": score,
            "timestamp": timestamp
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def backpropagate(self, node: QuestionNode, reward: float):
        """
        Backpropagate the reward up the tree.
        
        Update Q and N statistics for the node and all its ancestors.
        
        Args:
            node: The node where simulation ended
            reward: The reward to backpropagate
        """
        current = node
        while current is not None:
            current.update(reward)
            current = current.parent
    
    def search(
        self,
        root: QuestionNode,
        max_iterations: int = 100,
        target_leaf_nodes: int = 4
    ):
        """
        Run MCTS search.
        
        High-quality questions are saved to files during simulation via
        save_question method. No return value needed.
        
        Args:
            root: The root node
            max_iterations: Maximum number of MCTS iterations
            target_leaf_nodes: Stop when this many terminal nodes are generated
        """
        self.leaf_count = 0
        
        for iteration in range(max_iterations):
            # Check termination condition
            if self.leaf_count >= target_leaf_nodes:
                break
            
            # 1. Select
            selected = self.select(root)
            
            # 2. Expand (if not terminal)
            if not selected.is_terminal():
                new_node = self.expand(selected)
                # Move to newly expanded node for simulation
                selected = new_node
            
            # Count if this is a new terminal node
            if selected.is_terminal() and selected.N == 0:
                self.leaf_count += 1
            
            # 3. Simulate
            reward = self.simulate(selected)
            
            # 4. Backpropagate
            self.backpropagate(selected, reward)
