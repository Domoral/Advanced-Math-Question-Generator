"""
批量建库脚本
处理 documents/zy1000 中的 JSON 文件，通过 SiliconFlow Embedding API 生成 embedding 并存入 ChromaDB
"""

import json
import sys
from pathlib import Path
from tqdm import tqdm
import chromadb

# 添加 src 目录到 path，以导入 core 模块
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core.embedding_manager import SiliconFlowEmbedder


def get_backend_dir() -> Path:
    """获取项目根目录（build_index.py 所在目录）"""
    return Path(__file__).parent


def json_to_string(file_path: Path) -> str:
    """将 JSON 无损转化为字符串"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 按固定顺序拼接所有字段
    parts = []
    
    # 1. knowledge_points（列表转字符串）
    if 'knowledge_points' in data:
        kp = data['knowledge_points']
        if isinstance(kp, list):
            parts.append(f"knowledge_points: {', '.join(kp)}")
        else:
            parts.append(f"knowledge_points: {kp}")
    
    # 2. difficulty
    if 'difficulty' in data:
        parts.append(f"difficulty: {data['difficulty']}")
    
    # 3. question_type
    if 'question_type' in data:
        parts.append(f"question_type: {data['question_type']}")
    
    # 4. content（题目内容，放最后）
    if 'content' in data:
        parts.append(f"content: {data['content']}")
    
    # 5. 其他字段（如果有）
    known_fields = {'content', 'difficulty', 'question_type', 'knowledge_points'}
    for key, value in data.items():
        if key not in known_fields:
            if isinstance(value, list):
                value = ', '.join(str(v) for v in value)
            parts.append(f"{key}: {value}")
    
    return '\n'.join(parts)


def main():
    # 配置（使用相对路径）
    backend_dir = get_backend_dir()
    docs_dir = backend_dir / 'data' / 'documents' / 'zy1000'
    persist_dir = str(backend_dir / 'data' / 'vector_db')
    
    print(f"📁 文档目录: {docs_dir}")
    print(f"💾 向量库目录: {persist_dir}")
    
    # 初始化 SiliconFlow Embedding API 客户端
    print("\n🚀 连接 SiliconFlow Embedding API...")
    model = SiliconFlowEmbedder()
    
    print("🚀 初始化 ChromaDB...")
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(
        name="math_questions",
        metadata={
            "hnsw:space": "cosine",
            "hnsw:construction_ef": 200,
            "hnsw:search_ef": 50,
            "hnsw:M": 16
        }
    )
    
    # 获取所有 JSON 文件
    json_files = list(docs_dir.glob('*.json'))
    print(f"📊 找到 {len(json_files)} 个 JSON 文件\n")
    
    # 批量处理 - 减小 batch_size 以减少 API 调用频率
    batch_size = 16
    documents = []
    metadatas = []
    ids = []
    
    for i, file_path in enumerate(tqdm(json_files, desc="处理文档")):
        try:
            # JSON 转字符串
            text = json_to_string(file_path)
            
            # 读取原始数据用于 metadata
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 生成 ID
            doc_id = f"doc_{i:06d}"
            
            documents.append(text)
            metadatas.append({
                'source_file': file_path.name,
                'knowledge_points': ','.join(data.get('knowledge_points', [])),
                'difficulty': data.get('difficulty', ''),
                'question_type': data.get('question_type', '')
            })
            ids.append(doc_id)
            
            # 批量写入
            if len(documents) >= batch_size:
                print(f"\n  ↳ 调用 API 生成 {batch_size} 个 embedding...")
                embeddings = model.encode(documents)
                collection.add(
                    ids=ids,
                    documents=documents,
                    embeddings=embeddings.astype('float32').tolist(),
                    metadatas=metadatas
                )
                # 手动释放内存
                del documents, metadatas, ids, embeddings
                documents = []
                metadatas = []
                ids = []
                
        except Exception as e:
            print(f"\n❌ 处理 {file_path.name} 失败: {e}")
            continue
    
    # 写入剩余文档
    if documents:
        print(f"\n  ↳ 调用 API 生成 {len(documents)} 个 embedding...")
        embeddings = model.encode(documents)
        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings.astype('float32').tolist(),
            metadatas=metadatas
        )
        del embeddings
    
    # 统计
    total = collection.count()
    print(f"\n✅ 完成！向量库共有 {total} 条文档")


if __name__ == '__main__':
    main()
