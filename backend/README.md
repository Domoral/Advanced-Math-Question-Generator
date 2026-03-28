# Backend - Advanced Math Question Generator

## 目录结构

```
backend/
├── src/                    # 源代码
│   ├── __init__.py
│   ├── main.py            # 程序入口
│   ├── knowledge_selector.py   # 知识点选择器 (GUI)
│   ├── question_node.py   # MCTS 节点实现
│   ├── llm_client.py      # LLM API 客户端
│   ├── prompt_templates_CN.py  # 中文提示模板
│   └── prompt_templates_EN.py  # 英文提示模板
├── tests/                 # 测试文件
└── requirements.txt       # Python 依赖

```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行程序

```bash
python src/main.py
```

## 环境变量

创建 `.env` 文件：

```
DEEPSEEK_API_KEY=your_api_key_here
MODEL_NAME=deepseek-reasoner
```
