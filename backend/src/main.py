"""
Main entry point for Advanced Math Question Generator

Pipeline:
1. Knowledge Selection (GUI) -> Select target knowledge points
2. Initialize MCTS Root Node with waiting knowledge list
3. Run MCTS loop (Select -> Expand -> Simulate -> Backpropagate)
4. Save high-quality generated questions
"""

import os
from typing import List
from knowledge_selector import select_knowledge
from question_node import QuestionNode, QuestionMCTS


def main():
    """
    Main function to run the advanced math question generation system.
    """
    print("=" * 60)
    print("高等数学综合题生成系统")
    print("Advanced Math Question Generator")
    print("=" * 60)
    
    # Step 1: Knowledge Selection
    print("\n[步骤 1/3] 请选择需要综合的知识点...")
    print("[Step 1/3] Please select knowledge points to integrate...\n")
    
    selection_result = select_knowledge()
    
    if not selection_result or not selection_result.get('knowledge_points'):
        print("未选择任何知识点，程序退出。")
        print("No knowledge points selected. Exiting.")
        return
    
    selected_knowledge = selection_result['knowledge_points']
    difficulty_range = selection_result.get('difficulty_range', (0.3, 0.7))
    question_type = selection_result.get('question_type', '计算题')
    
    print(f"\n已选择 {len(selected_knowledge)} 个知识点:")
    print(f"Selected {len(selected_knowledge)} knowledge points:")
    for i, kp in enumerate(selected_knowledge, 1):
        print(f"  {i}. {kp}")
    
    print(f"\n难度区间: {difficulty_range[0]:.1f} - {difficulty_range[1]:.1f}")
    print(f"题型: {question_type}")
    
    # Step 2: Initialize MCTS Root Node
    print("\n" + "=" * 60)
    print("[步骤 2/3] 初始化 MCTS 搜索树...")
    print("[Step 2/3] Initializing MCTS search tree...")
    print("=" * 60)
    
    # Create root node with empty question and full waiting list
    root = QuestionNode(
        question="",  # Root has no question yet
        integrated_knowledge=set(),  # No knowledge integrated yet
        waiting_knowledge=selected_knowledge,  # All selected knowledge waiting to be added
        parent=None,
        difficulty_range=difficulty_range,  # Pass difficulty range to root node
        question_type=question_type  # Pass question type to root node
    )
    
    print(f"根节点创建完成")
    print(f"  - 待综合知识点: {len(selected_knowledge)} 个")
    print(f"  - 已综合知识点: 0 个")
    print(f"Root node created")
    print(f"  - Waiting knowledge: {len(selected_knowledge)}")
    print(f"  - Integrated knowledge: 0")
    
    # Step 3: Run MCTS Search
    print("\n" + "=" * 60)
    print("[步骤 3/3] 开始 MCTS 搜索...")
    print("[Step 3/3] Starting MCTS search...")
    print("=" * 60)
    
    # Initialize MCTS searcher with configuration
    mcts = QuestionMCTS(
        exploration_weight=1.414,  # UCT exploration parameter (sqrt(2))
        alpha=0.5,  # Weight for current score in simulation
        save_threshold=7.5,  # Minimum quality score to save question
        output_dir="./generated_question",  # Directory to save generated questions
        difficulty_range=difficulty_range,  # Target difficulty range
        question_type=question_type,  # Target question type
        need_optimize_threshold=6.0  # Threshold below which nodes trigger optimization
    )

    print("\nMCTS 配置:")
    print("MCTS Configuration:")
    print(f"  - 探索权重 (Exploration weight): {mcts.exploration_weight}")
    print(f"  - 模拟权重 (Alpha): {mcts.alpha}")
    print(f"  - 保存阈值 (Save threshold): {mcts.save_threshold}")
    print(f"  - 输出目录 (Output directory): {mcts.output_dir}")
    print(f"  - 优化阈值 (Optimize threshold): {mcts.need_optimize_threshold}")
    
    print("\n开始搜索循环 (Select -> Expand -> Simulate -> Backpropagate)...")
    print("Starting search loop...\n")
    
    # Run MCTS search
    # Stop when 4 terminal nodes are generated or max 100 iterations
    mcts.search(
        root=root,
        max_iterations=100,
        target_leaf_nodes=4
    )
    
    # Step 4: Summary
    print("\n" + "=" * 60)
    print("搜索完成!")
    print("Search completed!")
    print("=" * 60)
    
    print(f"\n统计信息:")
    print(f"Statistics:")
    print(f"  - 总迭代次数: {sum(node.N for node in root.children)}")
    print(f"  - 生成终端节点: {mcts.leaf_count}")
    print(f"  - 树深度: {max((node.depth() for node in get_all_nodes(root)), default=0)}")
    
    # Check output files in the timestamped subdirectory
    output_files = []
    if os.path.exists(mcts.output_dir):
        output_files = [f for f in os.listdir(mcts.output_dir) if f.endswith('.json')]
    
    print(f"\n已保存题目数量: {len(output_files)}")
    print(f"Saved questions: {len(output_files)}")
    
    if output_files:
        print(f"\n保存位置: {os.path.abspath(mcts.output_dir)}")
        print(f"Output location: {os.path.abspath(mcts.output_dir)}")
        print("\n文件列表:")
        print("Files:")
        for f in sorted(output_files):
            print(f"  - {f}")
    
    print("\n" + "=" * 60)
    print("程序执行完毕!")
    print("Execution completed!")
    print("=" * 60)


def get_all_nodes(root: QuestionNode) -> List[QuestionNode]:
    """
    Get all nodes in the tree (for statistics).
    
    Args:
        root: The root node of the tree
        
    Returns:
        List of all nodes in the tree
    """
    nodes = []
    stack = [root]
    
    while stack:
        node = stack.pop()
        nodes.append(node)
        stack.extend(node.children)
    
    return nodes


if __name__ == "__main__":
    main()
