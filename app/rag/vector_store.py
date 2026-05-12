"""ChromaDB向量存储模块"""
import os
import time
import uuid
from typing import List, Dict, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import settings
from app.llm.llm import build_embeddings


# Gemini 原生 embedding 对大批量不稳定，减小批大小 + 加大批间间隔
# 同时对"连接断开 / 超时 / 5xx"等网络错误也进行指数退避重试
EMBED_BATCH_SIZE = 10
EMBED_BATCH_SLEEP_SEC = 1.0
EMBED_MAX_RETRY = 6


def _is_retryable_error(msg: str) -> bool:
    """判断异常是否可重试：配额限流 / 连接断开 / 超时 / 服务端临时错误"""
    lowered = msg.lower()
    keywords = [
        "429", "resource_exhausted", "quota",
        "server disconnected", "disconnect",
        "connection", "timeout", "timed out",
        "unavailable", "503", "502", "500",
        "remotedisconnected", "protocolerror",
        "temporarily",
    ]
    return any(k in lowered for k in keywords)


class ChromaVectorStore:
    """基于ChromaDB的本地向量存储"""

    def __init__(self, persist_dir: str = None, collection_name: str = None):
        self.persist_dir = persist_dir or settings.CHROMA_PERSIST_DIR
        self.collection_name = collection_name or settings.CHROMA_COLLECTION_NAME

        # 确保目录存在
        os.makedirs(self.persist_dir, exist_ok=True)

        # 初始化ChromaDB客户端（本地持久化）
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        # 初始化Embedding模型（Gemini）
        self.embeddings = build_embeddings()

        # 文本分块器（chunk_size 提高到 2500，减少请求总数）
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2500,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
        )

    def _embed_with_rate_limit(self, chunks: List[str]) -> List[List[float]]:
        """按批次 + 限速 + 指数退避调用 Gemini embedding，规避 100 RPM 限制及连接断开"""
        all_vectors: List[List[float]] = []
        total = len(chunks)
        for i in range(0, total, EMBED_BATCH_SIZE):
            batch = chunks[i:i + EMBED_BATCH_SIZE]
            delay = 4.0
            success = False
            for attempt in range(EMBED_MAX_RETRY):
                try:
                    vectors = self.embeddings.embed_documents(batch)
                    all_vectors.extend(vectors)
                    print(f"  [embed] {min(i + EMBED_BATCH_SIZE, total)}/{total}")
                    success = True
                    break
                except Exception as e:
                    msg = str(e)
                    if _is_retryable_error(msg):
                        wait = min(delay, 60)
                        reason = msg.splitlines()[0][:120]
                        print(f"  [embed] 可重试错误({reason}) → 等待 {wait:.1f}s 后重试 ({attempt + 1}/{EMBED_MAX_RETRY})")
                        time.sleep(wait)
                        delay *= 2
                        continue
                    raise
            if not success:
                raise RuntimeError(f"Embedding 连续 {EMBED_MAX_RETRY} 次失败（持续触发限流或连接异常）")
            # 批间小睡，避免打满 100 RPM 且让连接喘口气
            if i + EMBED_BATCH_SIZE < total:
                time.sleep(EMBED_BATCH_SLEEP_SEC)
        return all_vectors

    def _embed_query_with_retry(self, query: str) -> List[float]:
        """embed_query 同样带重试，涵盖配额与网络类错误"""
        delay = 2.0
        for attempt in range(EMBED_MAX_RETRY):
            try:
                return self.embeddings.embed_query(query)
            except Exception as e:
                msg = str(e)
                if _is_retryable_error(msg):
                    wait = min(delay, 60)
                    print(f"  [embed-query] 可重试错误，等待 {wait:.1f}s 后重试 ({attempt + 1}/{EMBED_MAX_RETRY})")
                    time.sleep(wait)
                    delay *= 2
                    continue
                raise
        raise RuntimeError("查询 embedding 连续失败（持续触发限流或连接异常）")

    def add_documents(self, texts: List[str], metadatas: Optional[List[Dict]] = None, source: str = "unknown") -> int:
        """添加文档到向量库

        Args:
            texts: 文本列表
            metadatas: 元数据列表
            source: 文档来源标识

        Returns:
            添加的文档块数量
        """
        # 分块
        chunks = []
        chunk_metadatas = []
        for i, text in enumerate(texts):
            splits = self.text_splitter.split_text(text)
            for j, chunk in enumerate(splits):
                chunks.append(chunk)
                meta = metadatas[i] if metadatas and i < len(metadatas) else {}
                chunk_metadatas.append({
                    **meta,
                    "source": source,
                    "chunk_index": j,
                    "total_chunks": len(splits)
                })

        if not chunks:
            return 0

        print(f"  [embed] 开始处理 {len(chunks)} 个 chunk（批大小={EMBED_BATCH_SIZE}）...")
        # 生成embeddings（带限流 + 指数退避）
        embeddings = self._embed_with_rate_limit(chunks)

        # 生成唯一ID
        ids = [str(uuid.uuid4()) for _ in chunks]

        # 添加到Chroma
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=chunk_metadatas
        )

        return len(chunks)

    def add_file(self, file_path: str) -> int:
        """从文件添加文档

        Args:
            file_path: 文件路径（支持.txt, .md, .pdf）

        Returns:
            添加的文档块数量
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        # 读取文件内容
        file_ext = os.path.splitext(file_path)[1].lower()

        if file_ext in ['.txt', '.md']:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        elif file_ext == '.pdf':
            try:
                from langchain_community.document_loaders import PyPDFLoader
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                content = "\n".join([doc.page_content for doc in docs])
            except ImportError:
                raise ImportError("需要安装 pypdf: pip install pypdf")
        else:
            # 尝试作为文本文件读取
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

        filename = os.path.basename(file_path)
        return self.add_documents(
            texts=[content],
            metadatas=[{"filename": filename, "file_path": file_path}],
            source=filename
        )

    def search(self, query: str, k: int = 5) -> List[Dict]:
        """相似度检索

        Args:
            query: 查询文本
            k: 返回结果数量

        Returns:
            检索结果列表，包含 document, metadata, score
        """
        # 生成查询embedding（带重试）
        query_embedding = self._embed_query_with_retry(query)

        # 检索
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )

        # 格式化结果
        formatted = []
        if results and results['documents']:
            for i, doc in enumerate(results['documents'][0]):
                formatted.append({
                    "document": doc,
                    "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                    "score": 1 - results['distances'][0][i]  # 转换距离为相似度分数
                })

        return formatted

    def get_collection_stats(self) -> Dict:
        """获取集合统计信息"""
        return {
            "name": self.collection_name,
            "count": self.collection.count(),
            "persist_dir": self.persist_dir
        }

    def delete_collection(self):
        """删除当前集合"""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def list_sources(self) -> List[str]:
        """列出所有文档来源"""
        results = self.collection.get(include=["metadatas"])
        sources = set()
        if results and results['metadatas']:
            for meta in results['metadatas']:
                if meta and 'source' in meta:
                    sources.add(meta['source'])
        return list(sources)
