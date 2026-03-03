# 高数题目生成器 - 工程框架

## 1. 核心机制

基于蒙特卡洛树搜索（MCTS）逐步聚合知识点生成综合题目。

**树结构**：
- 根节点：空题目，包含完整知识点队列
- 中间节点：已聚合部分知识点的题目
- 叶节点：知识点队列为空的最终题目

**搜索流程**（标准MCTS四阶段）：
1. **Select**: UCT策略选择待扩展节点
2. **Expand**: Generator聚合下一个知识点，生成多个子节点
3. **Simulate**: Verifier评估当前节点 + 一次性聚合评估，加权得reward
4. **Backpropagate**: 反向传播更新路径上所有节点的Q、N

## 2. 核心模块

### 2.1 QuestionNode (`question_node.py`)

节点数据结构：
```
question: str                          # 当前题目内容
integrated_knowledge: Set[str]         # 已聚合知识点
waiting_knowledge: List[str]           # 待聚合知识点（拓扑排序）
parent: QuestionNode                   # 父节点引用
children: List[QuestionNode]           # 子节点列表
Q: float                               # 累计reward
N: int                                 # 访问次数
```

关键方法：
- `is_terminal()`: 检查waiting_knowledge是否为空
- `uct_score()`: 计算UCT分数用于Select阶段
- `update(reward)`: 更新Q、N统计

### 2.2 Prompt Templates (`prompt_templates.py`)

**question_generator**: 
- 输入：existing_problem, existing_skills, new_skill, reference_examples
- 输出：key-value格式的新题目（problem_statement, final_answer等）

**question_verifier**:
- 输入：problem_statement, required_skills, reference_examples  
- 输出：六维度评分（Single Answer, Exact Answer, Dual Skill Integration, Clarity, Tractability, Grammar）
- 总分：\boxed{score} 格式便于提取

### 2.3 QuestionMCTS (抽象类，`question_node.py`)

需实现四个接口：
- `select(root)`: UCT遍历选择节点
- `expand(node)`: 调用Generator生成子节点
- `simulate(node)`: 双阶段评估（当前节点 + 一次性全聚合）
- `backpropagate(node, reward)`: 反向传播更新统计

## 3. Simulate策略详解

```
reward = α * current_score + (1-α) * full_agg_score
```

| 阶段 | 操作 | 目的 |
|-----|------|------|
| current_score | Verifier评估node.question | 衡量中间产物可用性 |
| full_agg_score | Generator一次性聚合所有waiting_knowledge，Verifier评估 | 衡量路径潜力 |
| α | 超参数（默认0.5） | 平衡当前质量 vs 未来潜力 |

## 4. 数据结构

### 4.1 知识点拓扑

```python
knowledge_topology = {
    "不定积分": ["定积分"],      # 不定积分是定积分的前置
    "定积分": ["重积分", "曲线积分"],
    "导数定义": ["微分中值定理"],
    # ...
}
```

初始化时根据拓扑关系对waiting_knowledge排序。

### 4.2 参考题库

```python
reference_bank = {
    "不定积分": [
        {"question": "...", "difficulty": 3, "answer": "..."},
        # ...
    ],
    # ...
}
```

Expand阶段从中选取参考样例传入Generator。

## 5. 关键超参数

| 参数 | 说明 | 建议值 |
|-----|------|-------|
| exploration_weight (c) | UCT探索系数 | 1.414 |
| α | Simulate阶段当前分权重 | 0.5 |
| num_children_per_expand | 每次扩展生成的子节点数 | 3-5 |
| max_iterations | MCTS最大迭代次数 | 100-500 |
| top_k_keep | 每层保留的高质量节点数 | 5-10 |

## 6. 实现优先级

1. **基础框架**：QuestionNode + Prompt Templates
2. **MCTS核心**：实现四个抽象接口
3. **评估管线**：Verifier调用 + 分数提取
4. **聚合策略**：Generator调用 + 子节点生成
5. **优化增强**：剪枝策略、并行搜索、结果缓存

---

*版本: v2.0*  
*基于: prompt_templates.py, question_node.py*
