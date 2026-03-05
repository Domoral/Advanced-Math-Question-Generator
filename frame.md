# 高数题目生成器 - 工程框架

## 1. MCTS 算法流程

### 1.1 Select（选择）
从 root 开始遍历：
- 若 cur_node 已完全扩展（子节点数 ≥ 3），选择 UCT 最大的子节点，cur_node 下移，重复
- 若 cur_node 未完全扩展（子节点数 < 3），进入 Expand

### 1.2 Expand（扩展）
1. 从 cur_node.waiting_knowledge pop 第一个知识点
2. 调用 generator 与 cur_node.question 聚合，生成新子节点
3. 新子节点加入 cur_node.children
4. **cur_node 下移至新子节点**，进入 Rollout

### 1.3 Rollout（模拟/评估）

**情况 A：新子节点是叶节点**（waiting_knowledge 为空）
- 调用 verifier 评估 node.question
- reward = verifier_score

**情况 B：新子节点非叶节点**
1. 调用 verifier 评估当前 node.question → current_score
2. 取 waiting_knowledge[0:3]（最多前 3 个），调用 generator 一次性聚合 → virtual_question
3. 调用 verifier 评估 virtual_question → potential_score
4. reward = α × current_score + (1-α) × potential_score

### 1.4 Backpropagate（反向传播）
从 cur_node 沿 parent 链向上：
- 每个节点：Q += reward, N += 1
- 直到到达 root

---

## 2. 即时保存机制

任何节点被 verifier 评估后，若质量达标，立即保存：
- **中间节点**：quality ≥ 5.0/6 → 保存（高质量半成品）
- **叶节点**：quality ≥ 5.0/6 → 保存（高质量综合题）

保存内容：question, integrated_knowledge, depth, quality_score

---

## 3. 终止条件

满足任一即停止搜索：
1. **真实叶节点数 ≥ 4**（主要终止条件）
2. **总迭代次数 ≥ max_iterations**（保护性上限）

---

## 4. 核心模块

### 4.1 QuestionNode (`question_node.py`)

```python
question: str                          # 当前题目内容
integrated_knowledge: Set[str]         # 已聚合知识点
waiting_knowledge: List[str]           # 待聚合知识点（拓扑排序）
parent: QuestionNode                   # 父节点引用
children: List[QuestionNode]           # 子节点列表
Q: float                               # 累计reward
N: int                                 # 访问次数
```

### 4.2 LLM Client (`llm_client.py`)

- **generator(node, new_skill)**：聚合新知识点生成子节点题目
- **verifier(node)**：评估题目质量，返回六维度评分
- **extract_score(output)**：从 \boxed{} 中提取分数

### 4.3 Prompt Templates (`prompt_templates.py`)

- **question_generator**：输入 existing_problem, new_skill，输出 key-value 格式新题目
- **question_verifier**：输入 problem_statement，输出六维度评分 + \boxed{total_score}

---

## 5. 超参数设置

| 参数 | 值 | 说明 |
|-----|---|------|
| exploration_weight (c) | 1.414 | UCT 探索系数 |
| α | 0.5 | Rollout 中 current_score 权重 |
| children_per_node | 3 | 每个节点最大子节点数（完全扩展阈值）|
| max_rollout_knowledge | 3 | Rollout 虚拟聚合时最多取前 3 个知识点 |
| target_leaf_nodes | 4 | 终止条件：真实叶节点数 |
| save_threshold | 5.0 | 保存阈值（中间节点和叶节点相同）|
| max_iterations | 100-500 | 保护性迭代上限 |

---

## 6. 实现优先级

1. **基础框架**：QuestionNode + Prompt Templates + LLM Client
2. **MCTS 核心**：实现 Select/Expand/Rollout/Backpropagate
3. **即时保存**：质量达标节点保存到文件
4. **终止控制**：叶节点计数 + 迭代上限
5. **实验调优**：根据实验结果调整超参数

---

*版本: v3.0*  
*更新: 2026-03-01*
