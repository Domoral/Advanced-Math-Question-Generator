# Advanced-Math-Question-Generator

基于蒙特卡洛树搜索（MCTS）的高等数学综合题自动生成系统。结合大语言模型（DeepSeek）与 RAG 向量检索（ChromaDB + SiliconFlow Embedding），实现多知识点融合的智能题目生成与质量评估。

## 项目架构

```
Advanced-Math-Question-Generator/
├── frontend/                   # React (TypeScript) 前端
│   ├── src/
│   │   ├── App.tsx
│   │   ├── components/         # UI 组件
│   │   └── index.tsx
│   ├── package.json
│   └── tsconfig.json
├── backend/                    # Python (FastAPI) 后端
│   ├── src/
│   │   ├── api/                # REST API 服务
│   │   │   └── main.py         # FastAPI 入口
│   │   └── core/               # 核心业务模块
│   │       ├── .env            # 环境变量（API Keys 等）
│   │       ├── llm_client.py   # DeepSeek API 客户端
│   │       ├── question_node.py # MCTS 搜索算法
│   │       ├── rag_retriever.py # RAG 向量检索
│   │       ├── embedding_manager.py # Embedding 模型管理
│   │       ├── novelty_verifier.py  # 新颖性校验
│   │       ├── prompt_templates_CN.py
│   │       └── prompt_templates_EN.py
│   ├── build_index.py          # 向量库建库脚本
│   ├── data/                   # 数据文件
│   │   ├── documents/          # 题库文档
│   │   ├── vector_db/          # ChromaDB 持久化数据
│   │   └── logs/               # 生成任务日志
│   └── requirements.txt
├── deploy/                     # 部署配置
│   ├── docker/
│   │   ├── docker-compose.yml  # Docker Compose 编排
│   │   ├── Dockerfile.backend
│   │   └── Dockerfile.frontend
│   ├── nginx/
│   │   ├── nginx.conf          # Nginx 配置（含 SSL）
│   │   └── ssl/                # SSL 证书目录
│   │       ├── certificate.crt
│   │       ├── private.key
│   │       └── renew-ssl.sh    # SSL 自动续期脚本
│   └── scripts/
│       └── deploy.sh           # 一键部署脚本
└── docs/
```

## 部署架构

```
┌─────────────────────────────────────────────┐
│                  Docker Host                  │
│                                               │
│  ┌──────────────────────┐                    │
│  │   Nginx (Docker)     │                    │
│  │   - React 静态文件    │                    │
│  │   - SSL/TLS 终端     │                    │
│  └──────┬───────────────┘                    │
│         │ /api/* → 172.17.0.1:8000           │
│         ▼                                     │
│  ┌──────────────────────┐                    │
│  │  Backend (Host)      │                    │
│  │  FastAPI :8000       │                    │
│  └──────────────────────┘                    │
└─────────────────────────────────────────────┘
```

## Quick Start

### 1. 环境要求

- **Node.js** >= 18（前端构建）
- **Python** >= 3.11（后端运行）
- **Docker & Docker Compose**（部署 Nginx）

### 2. 配置环境变量

在 `backend/src/core/.env` 中配置 API Key：

```env
# Qwen LLM - 题目生成器（必需）
QWEN_API_KEY=your_qwen_api_key_here
QWEN_MODEL=qwen3.7-plus
QWEN_API_BASE=https://llm-e0yuxebkc2f8ofxa.cn-beijing.maas.aliyuncs.com

# DeepSeek LLM - 题目验证器（必需）
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DEEPSEEK_MODEL=deepseek-v4-flash
DEEPSEEK_API_BASE=https://api.deepseek.com

# SiliconFlow Embedding（必需，替代本地 embedding 模型）
SILICONFLOW_API_KEY=your_siliconflow_api_key_here
SILICONFLOW_MODEL=Qwen/Qwen3-Embedding-4B
SILICONFLOW_API_BASE=https://api.siliconflow.cn/v1/embeddings
```

### 3. 安装依赖

```bash
# 后端 Python 依赖
cd backend
pip install -r requirements.txt

# 前端 Node.js 依赖
cd ../frontend
npm install
```

### 4. 构建向量库（首次使用）

```bash
cd backend
python build_index.py
```

### 5. 构建前端

```bash
cd frontend
npm run build
```

### 6. 启动服务

```bash
# 启动后端（在后台运行）
cd backend/src/api
uvicorn main:app --host 0.0.0.0 --port 8000 &

# 或者使用 nohup 保持运行
nohup uvicorn main:app --host 0.0.0.0 --port 8000 > /dev/null 2>&1 &

# 启动 Nginx（Docker Compose）
cd ../../deploy/docker
docker compose up -d

# 或者使用一键部署脚本
cd ../..
bash deploy/scripts/deploy.sh
```

### 7. 访问

- **前端页面**: `https://amqg.tech` 或 `http://localhost`
- **API 文档**: `http://localhost:8000/docs`（Swagger UI）

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/` | API 信息 |
| `POST` | `/generate` | 发起题目生成任务 |
| `GET` | `/status/{task_id}` | 查询任务状态和日志 |
| `GET` | `/questions` | 获取已生成题目列表 |
| `GET` | `/questions/{id}` | 获取指定题目详情 |

### 生成请求示例

```json
{
  "knowledge_points": ["极限", "导数", "积分"],
  "difficulty_range": [0.3, 0.7],
  "question_type": "计算题",
  "use_rag": true,
  "max_iterations": 100,
  "target_leaf_nodes": 4
}
```

## SSL 证书配置

项目使用 Let's Encrypt 免费 SSL 证书，证书文件位于 `deploy/nginx/ssl/`。

### 首次申请证书

```bash
# 安装 certbot
sudo apt install certbot

# 申请证书（需要 80 端口空闲）
sudo certbot certonly --standalone -d amqg.tech -d www.amqg.tech

# 复制证书到项目目录
sudo cp /etc/letsencrypt/live/amqg.tech/fullchain.pem deploy/nginx/ssl/certificate.crt
sudo cp /etc/letsencrypt/live/amqg.tech/privkey.pem deploy/nginx/ssl/private.key
sudo chmod 644 deploy/nginx/ssl/certificate.crt
sudo chmod 600 deploy/nginx/ssl/private.key
```

### 自动续期

```bash
# 编辑 root crontab
sudo crontab -e

# 添加定时任务（每天 2:30 检查续期）
30 2 * * * certbot renew --quiet --deploy-hook "/home/ubuntu/Advanced-Math-Question-Generator/deploy/nginx/ssl/renew-ssl.sh" >> /home/ubuntu/Advanced-Math-Question-Generator/deploy/nginx/ssl/renew.log 2>&1
```

## 核心模块说明

### llm_client.py
双模型 LLM 客户端，避免自验证循环：
- **Qwen** (`qwen_client`) → `generator()`: 生成融合题目
- **DeepSeek** (`deepseek_client`) → `verifier()`: 独立评分验证
- 自动从 `backend/src/core/.env` 加载配置

### question_node.py
MCTS 核心算法实现：
- `QuestionNode`: 搜索树节点，记录题目状态和 MCTS 统计信息
- `QuestionMCTS`: 四步搜索循环（Select / Expand / Simulate / Backpropagate）
- 支持 RAG 增强生成和多轮优化

### rag_retriever.py
基于 ChromaDB 的向量检索模块：
- 根据知识点、题型、难度从向量库检索相似题目作为参考
- 通过 SiliconFlow Embedding API 生成查询向量

### embedding_manager.py
Embedding 模型管理器（单例模式）：
- **旧方案**: 本地加载 `sentence_transformers`（BGE 模型）
- **当前方案**: 调用 SiliconFlow API（`Qwen/Qwen3-Embedding-4B`）
- 统一管理 embedding 实例，避免重复加载

### novelty_verifier.py
题目新颖性校验模块，防止生成与已有题库高度重复的题目。

### build_index.py
批量建库脚本，将 `data/documents/` 中的题目数据向量化存入 ChromaDB。

## MCTS 配置参数

```python
mcts = QuestionMCTS(
    exploration_weight=1.414,   # UCT 探索参数
    alpha=0.5,                  # 当前得分与潜在得分权重
    save_threshold=8.4,         # 保存题目的最低质量分数
    difficulty_range=(0.3, 0.7), # 难度范围
    question_type="计算题",      # 题型
    need_optimize_threshold=7.5, # 触发优化的分数阈值
    use_rag=True                # 是否启用 RAG
)

mcts.search(
    root=root,
    max_iterations=100,         # 最大迭代次数
    target_leaf_nodes=4         # 目标融合知识点数
)
```

## Embedding 模型变更说明

系统已从本地 `sentence_transformers` 迁移到远程 SiliconFlow Embedding API：

| | 旧方案 | 新方案 |
|---|---|---|
| 模型 | `BAAI/bge-base-zh-v1.5`（本地） | `Qwen/Qwen3-Embedding-4B`（API） |
| 加载方式 | `SentenceTransformer(model_path)` | HTTP API 调用 |
| 硬件要求 | 需要 GPU/较大内存 | 无要求 |
| 配置 | `sentence_transformers>=2.2.0` | `SILICONFLOW_API_KEY` 环境变量 |

相关环境变量（`backend/src/core/.env`）：

```env
SILICONFLOW_API_KEY=your_key_here
SILICONFLOW_MODEL=Qwen/Qwen3-Embedding-4B
SILICONFLOW_API_BASE=https://api.siliconflow.cn/v1/embeddings
```

## Docker 部署说明

```bash
# 1. 构建前端
cd frontend && npm run build && cd ..

# 2. 启动 Nginx 容器
cd deploy/docker
docker compose up -d

# 3. 启动后端（在宿主机运行）
cd ../../backend/src/api
nohup uvicorn main:app --host 0.0.0.0 --port 8000 &

# 查看容器状态
docker ps

# 查看容器日志
docker logs math-question-nginx

# 停止服务
docker compose down
```

## Nginx 配置要点

- HTTP → HTTPS 自动重定向
- `/api/` 路径代理到后端 `172.17.0.1:8000`
- 前端路由支持 SPA（`try_files $uri $uri/ /index.html`）
- 静态资源缓存策略（1 年过期 + immutable）
- Gzip 压缩
- TLS 1.2 / 1.3

## 输出格式

生成的题目以 JSON 格式保存：

```json
{
  "question": "题目内容（LaTeX 格式）",
  "integrated_knowledge": ["极限", "导数", "积分"],
  "difficulty": 0.65,
  "question_type": "计算题",
  "quality_score": 8.5,
  "metadata": {
    "timestamp": "20260428_120000"
  }
}
```

## 依赖项

### 前端
- React 19
- TypeScript 4
- React Scripts 5（Create React App）

### 后端
- Python >= 3.11
- FastAPI + Uvicorn
- OpenAI SDK（DeepSeek & SiliconFlow API）
- ChromaDB（向量存储）
- Pydantic（数据验证）
- python-dotenv（环境变量）

## 常用命令

```bash
# 查看后端日志
tail -f backend/data/logs/*.log

# 证书信息
sudo certbot certificates

# 测试 SSL 连接
openssl s_client -connect amqg.tech:443 -servername amqg.tech

# 测试 certbot 自动续期
sudo certbot renew --dry-run

# 手动执行续期脚本
sudo /home/ubuntu/Advanced-Math-Question-Generator/deploy/nginx/ssl/renew-ssl.sh
```
