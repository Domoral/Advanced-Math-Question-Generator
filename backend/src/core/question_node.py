"""
MCTS Node implementation for Advanced Math Question Generation

This module defines the QuestionNode class used in the MCTS tree for generating
integrated mathematics problems by progressively combining knowledge points.
"""

from typing import List, Optional, Set, Dict, Any
from core.llm_client import generator, verifier, extract_score, parse_generator_output, parse_verifier_output, optimizer
from core.rag_retriever import retrieve_references
from core.novelty_verifier import NoveltyVerifier
from core.generation_logger import GenerationLogger
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
        parent: Optional['QuestionNode'] = None,
        difficulty_range: Optional[tuple] = None,
        question_type: Optional[str] = None
    ):
        """
        Initialize a QuestionNode.
        
        Args:
            question: The problem statement at this node (empty for root)
            integrated_knowledge: Set of knowledge points already in the question
            waiting_knowledge: Ordered list of knowledge points to be added
            parent: Parent node in the MCTS tree
            difficulty_range: Tuple of (min_difficulty, max_difficulty) for target difficulty range
            question_type: Type of question (单选题/多选题/填空题/计算题/证明题/应用题)
        """
        self.question = question
        self.integrated_knowledge = integrated_knowledge or set()
        self.waiting_knowledge = waiting_knowledge or []
        self.parent = parent
        
        # Difficulty range (inherited from parent if not specified)
        if difficulty_range is not None:
            self.difficulty_range = difficulty_range
        elif parent is not None:
            self.difficulty_range = parent.difficulty_range
        else:
            self.difficulty_range = (0.3, 0.7)  # Default range
        
        # Question type (inherited from parent if not specified)
        if question_type is not None:
            self.question_type = question_type
        elif parent is not None:
            self.question_type = parent.question_type
        else:
            self.question_type = "计算题"  # Default type
        
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
        
        # Reward history and optimization tracking
        self.reward_history: List[float] = []  # History of all rewards received
        self.last_reward: Optional[float] = None  # Last reward received (combined reward from simulate)
        self.this_score: Optional[float] = None  # Current node's own quality score (phase 1 score)
        self.solving_steps: Optional[str] = None  # Solving steps for this node's problem (from generator)
        self.deduction_details: Optional[str] = None  # Deduction details from verifier
        self.deduction_points: str = ""  # Natural language description of deduction points
        self.novelty_score: Optional[float] = None  # Novelty score from novelty verifier (0-1)
        self.llm_score: Optional[float] = None  # LLM verifier score (0-10)
        self.needs_optimization: bool = False  # Flag indicating if node needs optimization
        self.optimization_attempts: int = 0  # Number of optimization attempts made
        self.max_optimization_attempts: int = 3  # Maximum optimization attempts allowed
        
        # Node type flag
        self.is_root: bool = (parent is None)  # True if this is the root node
    
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
    
    def update(self, reward: float, this_score: Optional[float] = None, deduction_points: Optional[str] = None):
        """
        Update node statistics after a rollout.
        
        Args:
            reward: The combined quality score obtained from simulation (for backpropagation)
            this_score: The current node's own quality score (phase 1 score, for optimization decision)
            deduction_points: Natural language description of deduction points
        """
        self.N += 1
        self.Q += reward
        self.reward_history.append(reward)
        self.last_reward = reward
        if this_score is not None:
            self.this_score = this_score
        if deduction_points:
            self.deduction_points = deduction_points
        
    def should_optimize(self, threshold: float) -> bool:
        """
        Check if this node should be optimized instead of expanded.
        
        A node should be optimized if:
        1. It is NOT the root node (root nodes should always use generator)
        2. It has a this_score (own quality score) below the threshold
        3. It hasn't exceeded max optimization attempts
        
        Args:
            threshold: The quality threshold below which optimization is triggered
            
        Returns:
            True if node should be optimized, False otherwise
        """
        # Root node should never be optimized - always use generator for fusion
        if self.is_root:
            return False
        # Nodes without this_score should use generator
        if self.this_score is None:
            return False
        if self.this_score >= threshold:
            return False
        if self.optimization_attempts >= self.max_optimization_attempts:
            return False
        return True
    
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
        alpha: float = 0,
        save_threshold: float = 9.4,
        output_dir: str = None,  # Default to project data directory
        difficulty_range: tuple = (0.3, 0.7),
        question_type: str = "计算题",
        need_optimize_threshold: float = 8.5,  # Threshold below which nodes need optimization
        use_rag: bool = False  # Whether to use RAG for retrieving reference examples
    ):
        """
        Initialize the MCTS searcher.
        
        Args:
            exploration_weight: The 'c' parameter for UCT calculation
            alpha: Weight for current_score in simulate (0.5 means equal weight)
            save_threshold: Quality threshold for saving questions (default 5.0)
            output_dir: Base directory to save generated questions
            difficulty_range: Target difficulty range (min, max)
            question_type: Target question type
            need_optimize_threshold: Quality threshold below which nodes trigger optimization
            use_rag: Whether to use RAG for retrieving reference examples
        """
        self.exploration_weight = exploration_weight
        self.alpha = alpha
        self.save_threshold = save_threshold
        self.leaf_count = 0  # Count of terminal nodes generated
        self.difficulty_range = difficulty_range
        self.question_type = question_type
        self.need_optimize_threshold = need_optimize_threshold
        self.use_rag = use_rag
        
        # Set default output directory to project data folder
        if output_dir is None:
            # Get project root (backend's parent directory)
            backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            project_root = os.path.dirname(backend_dir)
            output_dir = os.path.join(project_root, "data", "generated_questions")
        
        # Create timestamped subdirectory for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(output_dir, timestamp)
        
        # Create output directory if not exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize generation logger for tracking all generated questions
        self.logger = GenerationLogger(self.output_dir)
        
        print(f"[INFO] 输出目录: {self.output_dir}")
        print(f"[INFO] 难度区间: {difficulty_range[0]:.1f} - {difficulty_range[1]:.1f}")
        print(f"[INFO] 题型: {question_type}")
        print(f"[INFO] 优化阈值: {need_optimize_threshold} (低于此分数将触发优化)")
        print(f"[INFO] RAG模式: {'开启' if use_rag else '关闭'}")
    
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
        print("  [Select] 开始选择节点...")
        node = root
        depth = 0
        
        while node.is_fully_expanded() and not node.is_terminal():
            # Select child with highest UCT score
            if not node.children:
                break
            node = max(node.children, key=lambda c: c.uct_score(self.exploration_weight))
            depth += 1
        
        this_score_str = f"{node.this_score:.2f}" if node.this_score is not None else "N/A"
        last_reward_str = f"{node.last_reward:.2f}" if node.last_reward is not None else "N/A"
        print(f"  [Select] 完成选择节点 (深度: {depth}, 已综合: {len(node.integrated_knowledge)} 个知识点, 自身得分: {this_score_str}, 综合得分: {last_reward_str})")
        return node

    def expand(self, node: QuestionNode) -> QuestionNode:
        """
        Expand a node by either optimizing it or generating a new child.

        Decision logic:
        - If node has low this_score (below need_optimize_threshold) and hasn't exceeded
          max optimization attempts: call optimizer to create an optimized child
        - Otherwise: use generator to create a new integrated child with next skill

        Args:
            node: The node to expand

        Returns:
            The newly generated child node (optimized or integrated)
        """
        if node.is_terminal():
            raise RuntimeError("Cannot expand terminal node")

        # Check if node needs optimization instead of further fusion
        if node.should_optimize(self.need_optimize_threshold):
            print(f"  [Expand] 节点自身得分 {node.this_score:.2f} 低于阈值 {self.need_optimize_threshold}，执行优化而非融合...")
            return self._expand_with_optimizer(node)
        else:
            # Get next knowledge point for normal fusion
            new_skill = node.waiting_knowledge[0]
            score_info = f"自身得分 {node.this_score:.2f}" if node.this_score else "无自身得分"
            print(f"  [Expand] {score_info}，继续融合知识点: {new_skill}...")
            return self._expand_with_generator(node)

    def _expand_with_optimizer(self, node: QuestionNode) -> QuestionNode:
        """
        Expand a node by creating an optimized version as a child.
        
        Args:
            node: The parent node to optimize
            
        Returns:
            The optimized child node
        """
        print(f"  [Expand-Optimize] 开始优化节点 (第 {node.optimization_attempts + 1}/{node.max_optimization_attempts} 次尝试)...")
        
        try:
            # Call optimizer with deduction points
            from llm_client import optimizer as optimizer_func, parse_optimizer_output
            
            raw_output = optimizer_func(node, node.deduction_points)
            
            if not raw_output:
                print(f"  [Expand-Optimize] 优化失败: API 返回空结果，回退到普通融合")
                node.optimization_attempts += 1
                return self._expand_with_generator(node)
            
            print(f"  [Expand-Optimize] Optimizer 原始输出:")
            print(f"    - 长度: {len(raw_output)}")
            print(f"    - 前500字符: {raw_output[:500]}")
            
            parsed = parse_optimizer_output(raw_output)
            optimized_problem = parsed.get('optimized_problem', '')
            
            if not optimized_problem:
                print(f"  [Expand-Optimize] 优化失败: 无法提取优化后的题目，回退到普通融合")
                node.optimization_attempts += 1
                return self._expand_with_generator(node)
            
            # Create child node with optimized problem
            # The child keeps the same knowledge structure as parent (optimization doesn't add new skills)
            child = QuestionNode(
                question=optimized_problem,
                integrated_knowledge=node.integrated_knowledge.copy(),  # Same skills as parent
                waiting_knowledge=node.waiting_knowledge.copy(),  # Same waiting list as parent
                parent=node
            )
            
            # Copy parent's properties
            child.difficulty_range = node.difficulty_range
            child.question_type = node.question_type
            
            # Update metadata
            child.metadata['generation_step'] = node.depth() + 1
            child.metadata['is_optimized_version'] = True
            child.metadata['parent_node_reward'] = node.last_reward
            
            # Save solving_steps to node attribute (not metadata)
            child.solving_steps = parsed.get('solving_steps', None)
            
            node.add_child(child)
            node.optimization_attempts += 1
            
            print(f"  [Expand-Optimize] 优化子节点创建完成 (子节点数: {len(node.children)})")
            print(f"    - 优化后题目: {parsed.get('optimized_problem', 'N/A')[:80]}...")
            if parsed.get('solving_steps'):
                print(f"    - 解题步骤: {parsed.get('solving_steps', 'N/A')[:80]}...")
            
            return child
            
        except Exception as e:
            print(f"  [Expand-Optimize] 优化失败: {str(e)}，回退到普通融合")
            node.optimization_attempts += 1
            return self._expand_with_generator(node)
    
    def _expand_with_generator(self, node: QuestionNode) -> QuestionNode:
        """
        Expand a node by generating a new child with the next skill.
        
        Pop the first knowledge point from waiting_knowledge,
        use generator to create a new integrated problem.
        
        Args:
            node: The node to expand
            
        Returns:
            The newly generated child node
        """
        # Get next knowledge point
        new_skill = node.waiting_knowledge[0]
        remaining = node.waiting_knowledge[1:]
        
        print(f"  [Expand-Generator] 开始扩展节点，添加知识点: {new_skill}...")
        
        # Retrieve reference examples if RAG is enabled
        reference_examples = ""
        if self.use_rag:
            # Build knowledge points list for query (already integrated + new skill)
            query_knowledge_points = list(node.integrated_knowledge | {new_skill})
            reference_examples = retrieve_references(
                knowledge_points=query_knowledge_points,
                question_type=self.question_type,
                difficulty_range=self.difficulty_range,
                top_k=3,
                use_rag=True
            )
            if reference_examples:
                print(f"  [Expand-RAG] 检索到参考例题")
        
        # Generate new problem using LLM
        try:
            raw_output = generator(
                node, 
                new_skill, 
                difficulty_range=self.difficulty_range, 
                question_type=self.question_type,
                reference_examples=reference_examples
            )
            
            print(f"  [Expand] Generator 原始输出:")
            print(f"    - 长度: {len(raw_output)}")
            print(f"    - 前500字符: {raw_output[:500]}")
            
            parsed = parse_generator_output(raw_output)
            
            print(f"  [Expand] Parser 提取结果:")
            print(f"    - problem_statement: {repr(parsed.get('problem_statement', 'NOT FOUND'))}")
            print(f"    - solving_steps: {repr(parsed.get('solving_steps', 'NOT FOUND'))}")
            
            # Create child node
            child = QuestionNode(
                question=parsed.get('problem_statement', ''),
                integrated_knowledge=node.integrated_knowledge | {new_skill},
                waiting_knowledge=remaining,
                parent=node
            )
            
            # Copy metadata
            child.metadata['generation_step'] = node.depth() + 1
            child.solving_steps = parsed.get('solving_steps', None)
            
            node.add_child(child)
            print(f"  [Expand] 完成扩展节点 (子节点数: {len(node.children)})")
            return child
            
        except Exception as e:
            # If generation fails, create a fallback child with error marker
            print(f"  [Expand] 扩展失败: {str(e)}")
            child = QuestionNode(
                question=f"[Generation Error: {str(e)}]",
                integrated_knowledge=node.integrated_knowledge | {new_skill},
                waiting_knowledge=remaining,
                parent=node
            )
            node.add_child(child)
            return child
    
    def simulate(self, node: QuestionNode, iteration: int = 0) -> float:
        """
        Simulate/rollout from a node to get a reward.
        
        Case A: Terminal Node - directly evaluate with verifier + novelty
        Case B: Non-Terminal Node - two-phase evaluation with novelty
        
        Args:
            node: The node to simulate from
            iteration: Current MCTS iteration number
            
        Returns:
            The combined reward (quality score) obtained
        """
        is_terminal = node.is_terminal()
        node_type = "终端" if is_terminal else "中间"
        print(f"  [Simulate] 开始模拟评估 ({node_type}节点)...")
        
        # Initialize novelty verifier
        novelty_verifier = NoveltyVerifier()

        # Phase 1: Evaluate current node (for both terminal and non-terminal)
        print(f"    [Simulate-Phase1] 评估当前节点...")
        try:
            # LLM verifier
            verifier_output = verifier(node, difficulty_range=self.difficulty_range)
            print(f"    [Simulate-Phase1] Verifier 原始输出:")
            print(f"{'='*60}")
            print(verifier_output)
            print(f"{'='*60}")

            parsed_verifier = parse_verifier_output(verifier_output)
            llm_current_score = extract_score(verifier_output)
            print(f"    [Simulate-Phase1] Extracted LLM score: {llm_current_score}")
            if llm_current_score is None:
                print(f"    [Simulate-Phase1] Warning: Could not extract score from verifier output")
                llm_current_score = 0.0

            # Novelty verifier
            print(f"    [Simulate-Phase1] 调用 Novelty Verifier 计算新颖度...")
            novelty_current_score = novelty_verifier.compute_novelty(node)
            print(f"    [Simulate-Phase1] Novelty Verifier 计算完成: {novelty_current_score:.4f}")

            # Combined score
            current_score = llm_current_score + novelty_current_score

            # Build deduction points description
            # Build deduction points description
            # Start with verifier's deduction details
            deduction_points = parsed_verifier.get('deduction_details', "") or ""

            # Append novelty deduction if below threshold
            if novelty_current_score < 0.4:
                novelty_deduction = f"- 新颖性不足: 新颖度得分 {novelty_current_score:.2f} < 0.4，题目与题库中现有题目过于相似"
                if deduction_points:
                    deduction_points += "\n" + novelty_deduction
                else:
                    deduction_points = novelty_deduction

            node.deduction_points = deduction_points if deduction_points else "符合要求"

            # Also save to deduction_details for backward compatibility
            node.deduction_details = node.deduction_points

            print(f"    [Simulate-Phase1] Verifier 详细评价:")
            print(f"      - LLM总分: {llm_current_score:.2f}")
            print(f"      - 新颖度: {novelty_current_score:.2f}")
            print(f"      - 综合总分: {current_score:.2f}")
            for dim_name, score_val in parsed_verifier['scores'].items():
                print(f"      - {dim_name}: {score_val}分")

            node.this_score = current_score  # Save the node's own quality score
            node.llm_score = llm_current_score  # Save LLM score
            node.novelty_score = novelty_current_score  # Save novelty score

            # Log to CSV (real node)
            is_saved = current_score >= self.save_threshold
            self.logger.log_question(
                iteration=iteration,
                question_type='real',
                novelty=novelty_current_score,
                llm_score=llm_current_score,
                is_saved=is_saved
            )

            # Save if quality meets threshold
            if is_saved:
                self.save_question(node, current_score, llm_score=llm_current_score, novelty_score=novelty_current_score)
                print(f"    [Simulate-Phase1] 当前节点得分: {current_score:.2f} (LLM:{llm_current_score:.2f} + Novelty:{novelty_current_score:.2f}) (已保存)")
            else:
                print(f"    [Simulate-Phase1] 当前节点得分: {current_score:.2f} (LLM:{llm_current_score:.2f} + Novelty:{novelty_current_score:.2f})")
        except Exception as e:
            import traceback
            print(f"    [Simulate-Phase1] 评估失败: {type(e).__name__}: {str(e)}")
            print(f"    [Simulate-Phase1] 详细错误信息:")
            traceback.print_exc()
            current_score = 0.0

        # For terminal node, return current score directly (no Phase 2)
        if is_terminal:
            print(f"  [Simulate] 终端节点评估完成，得分: {current_score:.2f}")
            return current_score

        # Phase 2: Evaluate potential (only for non-terminal nodes)
        print(f"    [Simulate-Phase2] 评估潜在能力...")
        try:
            # Take at most first 3 remaining knowledge points
            knowledge_to_test = node.waiting_knowledge[:3]
            if knowledge_to_test:
                # Convert list to combined string
                combined_skill = "、".join(knowledge_to_test)
                
                # Generate virtual question
                virtual_output = generator(node, combined_skill)
                virtual_parsed = parse_generator_output(virtual_output)
                
                print(f"    [Simulate-Phase2] Parser 提取结果:")
                print(f"      - problem_statement: {repr(virtual_parsed.get('problem_statement', 'NOT FOUND'))}")
                print(f"      - solving_steps: {repr(virtual_parsed.get('solving_steps', 'NOT FOUND'))}")
                
                # Create temporary node for verification
                # Inherit difficulty_range and question_type from parent node
                temp_node = QuestionNode(
                    question=virtual_parsed.get('problem_statement', ''),
                    integrated_knowledge=node.integrated_knowledge | set(knowledge_to_test),
                    waiting_knowledge=[],  # Virtual node is terminal
                    parent=node  # Set parent to inherit difficulty_range and question_type
                )
                
                # Verify virtual question with LLM verifier
                virtual_verifier_output = verifier(temp_node)
                print(f"    [Simulate-Phase2] Verifier 原始输出:")
                print(f"{'='*60}")
                print(virtual_verifier_output)
                print(f"{'='*60}")
                
                virtual_parsed_verifier = parse_verifier_output(virtual_verifier_output)
                llm_potential_score = extract_score(virtual_verifier_output)
                if llm_potential_score is None:
                    llm_potential_score = 0.0

                # Novelty verifier for virtual node
                print(f"    [Simulate-Phase2] 调用 Novelty Verifier 计算虚拟节点新颖度...")
                novelty_potential_score = novelty_verifier.compute_novelty(temp_node)
                print(f"    [Simulate-Phase2] Novelty Verifier 计算完成: {novelty_potential_score:.4f}")

                # Combined score for virtual node
                potential_score = llm_potential_score + novelty_potential_score

                # Build deduction points description for virtual node
                # Start with verifier's deduction details
                virtual_deduction_points = virtual_parsed_verifier.get('deduction_details', "") or ""

                # Append novelty deduction if below threshold
                if novelty_potential_score < 0.4:
                    virtual_novelty_deduction = f"- 新颖性不足: 新颖度得分 {novelty_potential_score:.2f} < 0.4，题目与题库中现有题目过于相似"
                    if virtual_deduction_points:
                        virtual_deduction_points += "\n" + virtual_novelty_deduction
                    else:
                        virtual_deduction_points = virtual_novelty_deduction

                temp_node.deduction_points = virtual_deduction_points if virtual_deduction_points else "符合要求"

                # Also save to deduction_details for backward compatibility
                temp_node.deduction_details = temp_node.deduction_points

                # Save scores to temp_node
                temp_node.llm_score = llm_potential_score
                temp_node.novelty_score = novelty_potential_score

                print(f"    [Simulate-Phase2] Verifier 详细评价:")
                print(f"      - LLM总分: {llm_potential_score:.2f}")
                print(f"      - 新颖度: {novelty_potential_score:.2f}")
                print(f"      - 综合总分: {potential_score:.2f}")
                for dim_name, dim_score in virtual_parsed_verifier['scores'].items():
                    print(f"      - {dim_name}: {dim_score}分")
                print(f"    [Simulate-Phase2] 潜在得分: {potential_score:.2f} (LLM:{llm_potential_score:.2f} + Novelty:{novelty_potential_score:.2f})")

                # Log to CSV (virtual node)
                is_virtual_saved = potential_score >= self.save_threshold
                self.logger.log_question(
                    iteration=iteration,
                    question_type='virtual',
                    novelty=novelty_potential_score,
                    llm_score=llm_potential_score,
                    is_saved=is_virtual_saved
                )

                # Save virtual question if quality meets threshold
                if is_virtual_saved:
                    print(f"    [Simulate-Phase2] 虚拟节点得分达标，保存中间产物...")
                    self.save_question(temp_node, potential_score, is_virtual=True, llm_score=llm_potential_score, novelty_score=novelty_potential_score)
                    print(f"    [Simulate-Phase2] 虚拟节点已保存")
                
                # Clean up virtual node immediately (important for memory management)
                del temp_node
                temp_node = None
            else:
                potential_score = current_score
                print(f"    [Simulate-Phase2] 无剩余知识点，跳过")
        except Exception as e:
            import traceback
            print(f"    [Simulate-Phase2] 评估失败: {type(e).__name__}: {str(e)}")
            print(f"    [Simulate-Phase2] 详细错误信息:")
            traceback.print_exc()
            potential_score = 0.0
        
        # Phase 3: Combine scores
        reward = self.alpha * current_score + (1 - self.alpha) * potential_score
        print(f"  [Simulate] 完成评估，综合得分: {reward:.2f} (当前{current_score:.2f} * {self.alpha} + 潜在{potential_score:.2f} * {1-self.alpha})")
        return reward
    
    def save_question(self, node: QuestionNode, score: float, is_virtual: bool = False,
                      llm_score: Optional[float] = None, novelty_score: Optional[float] = None):
        """
        Save a high-quality question to JSON file and Markdown file.

        Args:
            node: The QuestionNode containing the question to save
            score: The quality score from verifier (>= save_threshold)
            is_virtual: Whether this is a virtual node (default: False)
            llm_score: LLM verifier score (0-10)
            novelty_score: Novelty verifier score (0-1)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if is_virtual:
            prefix = "virtual"
        elif node.is_terminal():
            prefix = "terminal"
        else:
            prefix = "intermediate"

        filename = f"{prefix}_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)

        # Get solving steps from node attribute
        solving_steps = getattr(node, 'solving_steps', None) or ""

        # Get question type and difficulty range
        question_type = getattr(node, 'question_type', '计算题')
        difficulty_range = getattr(node, 'difficulty_range', (0.3, 0.7))

        # Get scores from node if not provided
        if llm_score is None:
            llm_score = getattr(node, 'llm_score', None) or 0.0
        if novelty_score is None:
            novelty_score = getattr(node, 'novelty_score', None) or 0.0

        data = {
            "question": node.question,
            "solving_steps": solving_steps,
            "question_type": question_type,
            "difficulty_range": difficulty_range,
            "integrated_knowledge": sorted(list(node.integrated_knowledge)),
            "quality_score": score,
            "llm_score": llm_score,
            "novelty_score": novelty_score,
            "timestamp": timestamp,
            "is_virtual": is_virtual
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        # Also save as Markdown file
        self._save_markdown(node, score, prefix, timestamp, is_virtual, llm_score, novelty_score)
    
    def _save_markdown(self, node: QuestionNode, score: float, prefix: str, timestamp: str,
                        is_virtual: bool = False, llm_score: Optional[float] = None,
                        novelty_score: Optional[float] = None):
        """
        Save question as Markdown file.

        Args:
            node: The QuestionNode containing the question to save
            score: The quality score from verifier
            prefix: Filename prefix (virtual/terminal/intermediate)
            timestamp: Timestamp string
            is_virtual: Whether this is a virtual node
            llm_score: LLM verifier score (0-10)
            novelty_score: Novelty verifier score (0-1)
        """
        md_filename = f"{prefix}_{timestamp}.md"
        md_filepath = os.path.join(self.output_dir, md_filename)

        # Helper function to process LaTeX for Markdown
        def process_latex_for_markdown(text: str) -> str:
            """Process LaTeX text to make it compatible with Markdown preview."""
            if not text:
                return text

            # First, convert double backslashes to single backslashes
            text = text.replace('\\\\', '\\')

            # Replace \[ ... \] with $$ ... $$ (display math)
            import re
            text = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', text, flags=re.DOTALL)

            # Replace \( ... \) with $ ... $ (inline math)
            text = re.sub(r'\\\((.*?)\\\)', r'$\1$', text, flags=re.DOTALL)

            # Handle standalone math environments (equation, align, etc.)
            # Wrap environment blocks with $$
            env_pattern = r'(\\begin\{(equation|align|align\*|gather|gather\*|multline|multline\*)\}.*?\\end\{\2\})'
            text = re.sub(env_pattern, r'$$\1$$', text, flags=re.DOTALL)

            return text

        # Process question text
        question_text = process_latex_for_markdown(node.question)

        # Get question type and difficulty range
        question_type = getattr(node, 'question_type', '计算题')
        difficulty_range = getattr(node, 'difficulty_range', (0.3, 0.7))

        # Get solving steps from node attribute
        solving_steps = process_latex_for_markdown(getattr(node, 'solving_steps', None) or "")

        # Use provided scores or get from node
        if llm_score is None:
            llm_score = getattr(node, 'llm_score', None) or 0.0
        if novelty_score is None:
            novelty_score = getattr(node, 'novelty_score', None) or 0.0

        # Build markdown content
        md_content = f"""# 高等数学综合题

## 题目

{question_text if question_text else "（无题目内容）"}

---

## 解题步骤

{solving_steps if solving_steps else "（暂无解题步骤）"}

---

## 题目信息

- **题型**: {question_type}
- **质量评分**: {score:.1f}/10.0 (LLM: {llm_score:.1f} + 新颖度: {novelty_score:.2f})
- **知识点**: {', '.join(sorted(node.integrated_knowledge)) if node.integrated_knowledge else '无'}
- **节点类型**: {'虚拟节点' if is_virtual else ('终端节点' if node.is_terminal() else '中间节点')}
- **生成时间**: {timestamp}
- **难度区间**: {difficulty_range[0]:.1f} - {difficulty_range[1]:.1f}

---

*由高等数学综合题生成系统自动生成*
"""
        
        with open(md_filepath, 'w', encoding='utf-8') as f:
            f.write(md_content)
    
    def backpropagate(self, node: QuestionNode, reward: float):
        """
        Backpropagate the reward up the tree.
        
        Update Q and N statistics for the node and all its ancestors.
        
        Args:
            node: The node where simulation ended
            reward: The reward to backpropagate
        """
        print(f"  [Backpropagate] 开始回溯传播 (奖励: {reward:.2f})...")
        current = node
        depth = 0
        # Pass this_score and deduction points from the simulated node to its ancestors
        this_score = node.this_score if hasattr(node, 'this_score') else None
        deduction_points = node.deduction_points if hasattr(node, 'deduction_points') else ""
        while current is not None:
            # Only pass this_score and deduction_points to the simulated node itself
            current.update(
                reward,
                this_score if current == node else None,
                deduction_points if current == node else None
            )
            depth += 1
            current = current.parent
        print(f"  [Backpropagate] 完成回溯传播 (更新 {depth} 个节点)")
    
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
            print(f"\n[迭代 {iteration + 1}/{max_iterations}] 开始 MCTS 循环...")
            
            # Check termination condition
            if self.leaf_count >= target_leaf_nodes:
                print(f"\n已达到目标终端节点数 ({target_leaf_nodes})，停止搜索。")
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
                print(f"  [信息] 生成新的终端节点 (当前: {self.leaf_count}/{target_leaf_nodes})")
            
            # 3. Simulate
            reward = self.simulate(selected, iteration=iteration + 1)
            
            # 4. Backpropagate
            self.backpropagate(selected, reward)
            
            print(f"[迭代 {iteration + 1}] 完成")
        
        print(f"\n[搜索结束] 共执行 {iteration + 1} 次迭代，生成 {self.leaf_count} 个终端节点")
