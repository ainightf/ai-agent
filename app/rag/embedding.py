"""Embedding模块（Gemini）"""
from typing import List

from config.settings import settings
from app.llm.llm import build_embeddings


class EmbeddingManager:
    """Embedding管理器"""

    def __init__(self):
        self.embeddings = build_embeddings()

    def embed_text(self, text: str) -> List[float]:
        """将单个文本转换为向量"""
        return self.embeddings.embed_query(text)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """将多个文本转换为向量"""
        return self.embeddings.embed_documents(texts)

    def get_model_name(self) -> str:
        """获取当前使用的模型名称"""
        return settings.EMBEDDING_MODEL
