# Backend - Advanced Math Question Generator

## 目录结构

```
backend/
├── src/
│   ├── __init__.py
│   ├── api/                        # REST API 服务
│   │   └── main.py                 # FastAPI 入口，定义所有接口
│   └── core/                       # 核心业务模块
│       ├── .env                    # 环境变量（API Keys）
│       ├── __init__.py
│       ├── llm_client.py           # DeepSeek API 客户端
│       ├── question_node.py        # MCTS 节点与搜索算法
│       ├── rag_retriever.py        # RAG 向量检索（ChromaDB）
│       ├── embedding_manager.py    # Embedding 模型管理器
│       ├── novelty_verifier.py     # 题目新颖性校验
│       ├── prompt_templates_CN.py  # 中文提示词模板
│       └── prompt_templates_EN.py  # 英文提示词模板
├── build_index.py                  # 向量库建库脚本
├── data/                           # 数据目录
│   ├── documents/                  # 题库文档（JSON）
│   ├── vector_db/                  # ChromaDB 持久化数据
│   └── logs/                       # 任务执行日志
├── README.md
└── requirements.txt
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 环境变量

在 `src/core/.env` 中配置以下环境变量：

```env
# DeepSeek LLM（必需）
DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_MODEL=deepseek-v4-flash
DEEPSEEK_API_BASE=https://api.deepseek.com

# SiliconFlow Embedding API（必需，替代本地模型）
SILICONFLOW_API_KEY=your_siliconflow_api_key
SILICONFLOW_MODEL=Qwen/Qwen3-Embedding-4B
SILICONFLOW_API_BASE=https://api.siliconflow.cn/v1/embeddings
```

## 构建向量库

首次使用需要构建向量库：

```bash
python build_index.py
```

该脚本读取 `data/documents/` 中的 JSON 题目文件，通过 SiliconFlow Embedding API 生成向量，存入 ChromaDB。

## 启动 API 服务

```bash
cd src/api
uvicorn main:app --host 0.0.0.0 --port 8000

# 或后台运行
nohup uvicorn main:app --host 0.0.0.0 --port 8000 > /dev/null 2>&1 &
```

启动后访问：
- API 文档（Swagger UI）: `http://localhost:8000/docs`
- ReDoc 文档: `http://localhost:8000/redoc`

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/` | API 信息 |
| `POST` | `/generate` | 发起题目生成任务 |
| `GET` | `/status/{task_id}` | 查询任务状态和实时日志 |
| `GET` | `/questions` | 获取已生成题目列表 |
| `GET` | `/questions/{id}` | 获取指定题目详情 |

### 生成接口示例

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "knowledge_points": ["极限", "导数", "积分"],
    "difficulty_range": [0.3, 0.7],
    "question_type": "计算题",
    "use_rag": true,
    "max_iterations": 100,
    "target_leaf_nodes": 4,
    "save_threshold": 8.4
  }'
```

### 查询任务状态

```bash
curl http://localhost:8000/status/20260428_120000
```

## 核心模块说明

### llm_client.py
- `generator()`: 调用 DeepSeek API 生成融合题目
- `verifier()`: 调用 DeepSeek API 评估题目质量并打分
- 配置自动从 `.env` 加载

### question_node.py
- `QuestionNode`: MCTS 搜索树节点
- `QuestionMCTS`: 实现 Select → Expand → Simulate → Backpropagate 搜索循环
- 支持 RAG 增强、多轮优化、自动保存

### rag_retriever.py
- 基于 ChromaDB 的向量检索
- 根据知识点、题型、难度检索相似题目示例
- 使用 SiliconFlow API 生成查询 embedding

### embedding_manager.py
- 单例模式管理 embedding 模型
- 已从本地 `sentence_transformers` 迁移到 SiliconFlow 远程 API
- 统一管理，避免重复实例化

### novelty_verifier.py
- 验证生成题目与已有题库的新颖性
- 防止生成高度重复的题目

## MCTS 配置参数

```python
mcts = QuestionMCTS(
    exploration_weight=1.414,    # UCT 探索参数
    alpha=0.5,                   # 当前得分与潜在得分权重
    save_threshold=8.4,          # 保存题目的最低分数
    need_optimize_threshold=7.5, # 触发多轮优化阈值
    use_rag=True                 # 是否启用 RAG
)
```

## 日志

任务日志存储在 `data/logs/` 目录，文件名格式为 `{task_id}.log`。

## 依赖项

```
fastapi>=0.100.0
uvicorn>=0.23.0
pydantic>=2.0.0
openai>=1.0.0
python-dotenv>=1.0.0
chromadb>=0.4.0
tqdm>=4.65.0
```
