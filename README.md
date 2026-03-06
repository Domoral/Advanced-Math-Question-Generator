# Advanced-Math-Question-Generator

基于蒙特卡洛树搜索（MCTS）的高等数学综合题自动生成系统。

## 项目简介

本项目利用大语言模型（LLM）和蒙特卡洛树搜索算法，自动将多个高等数学知识点融合生成高质量的综合性题目。系统支持知识点依赖关系管理、智能题目生成、质量评估与筛选。

## Quick Start

### 1. 环境准备

```bash
# 创建 conda 环境
conda create -n amqg python=3.11
conda activate amqg

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置 API Key

在项目根目录创建 `.env` 文件：

```bash
# 从 DeepSeek 平台获取 API Key: https://platform.deepseek.com/
DEEPSEEK_API_KEY=your_api_key_here
MODEL_NAME=deepseek-reasoner
```

### 3. 运行程序

```bash
python main.py
```

运行后：
1. 在弹出的 GUI 中选择需要综合的知识点
2. 系统自动执行 MCTS 搜索生成题目
3. 高质量题目将保存到 `generated_question/` 目录

## 文件结构

```
.
├── main.py                      # 主程序入口
├── knowledge_selector.py        # 知识点选择 GUI
├── question_node.py             # MCTS 节点与搜索算法
├── llm_client.py               # LLM API 调用客户端
├── prompt_templates_EN.py      # 英文提示词模板
├── prompt_templates_CN.py      # 中文提示词模板
├── .env                        # API Key 配置文件（需自行创建）
└── generated_question/        # 生成的题目保存目录
```

## 各文件介绍

### main.py
主程序入口，串联整个系统流程：
- 调用知识点选择器获取目标知识点
- 初始化 MCTS 根节点
- 执行 MCTS 搜索循环
- 输出统计信息和保存结果

### knowledge_selector.py
知识点选择 GUI 模块：
- 提供可视化界面选择高数知识点
- 自动处理知识点前置依赖关系
- 按拓扑顺序返回选中的知识点列表

### question_node.py
MCTS 核心算法实现：
- `QuestionNode`: 搜索树节点类，存储题目状态和 MCTS 统计信息
- `QuestionMCTS`: MCTS 搜索器，实现 Select/Expand/Simulate/Backpropagate 四步循环

### llm_client.py
LLM API 客户端：
- `generator()`: 调用 LLM 生成综合题目
- `verifier()`: 调用 LLM 评估题目质量
- `extract_score()`: 从评估结果中提取分数
- `parse_generator_output()`: 解析中英文格式的生成结果

### prompt_templates_EN.py / prompt_templates_CN.py
提示词模板文件：
- `question_generator`: 题目生成提示词
- `question_verifier`: 题目质量评估提示词

## 系统流程

```
┌─────────────────┐
│ 知识点选择 (GUI) │
└────────┬────────┘
         ▼
┌─────────────────┐
│ 初始化 MCTS 根节点│
└────────┬────────┘
         ▼
┌─────────────────────────────┐
│      MCTS 搜索循环           │
│  Select → Expand → Simulate │
│    ↑                  ↓     │
│  Backpropagate ←──────┘     │
└────────┬────────────────────┘
         ▼
┌─────────────────┐
│ 保存高质量题目  │
└─────────────────┘
```

## 配置说明

### MCTS 参数（main.py 中修改）

```python
mcts = QuestionMCTS(
    exploration_weight=1.414,  # UCT 探索参数
    alpha=0.5,                 # 当前得分与潜在得分的权重
    save_threshold=5.0,        # 保存题目的最低质量分数
    output_dir="./generated_question"  # 输出目录
)
```

### 搜索终止条件

```python
mcts.search(
    root=root,
    max_iterations=100,        # 最大迭代次数
    target_leaf_nodes=4        # 目标终端节点数
)
```

## 输出格式

生成的题目以 JSON 格式保存：

```json
{
  "question": "题目内容（LaTeX 格式）",
  "integrated_knowledge": ["知识点1", "知识点2", ...],
  "quality_score": 5.5,
  "timestamp": "20240306_123456"
}
```

## 依赖项

- Python >= 3.9
- openai >= 1.0
- python-dotenv
- tkinter（Python 标准库）

