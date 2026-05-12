"""检索器模块 - 基于ChromaDB"""
from typing import List, Dict, Optional

from app.rag.vector_store import ChromaVectorStore


class ChromaRetriever:
    """基于ChromaDB的文档检索器"""

    def __init__(self, vector_store: ChromaVectorStore = None):
        self.vector_store = vector_store or ChromaVectorStore()

    def retrieve(self, query: str, k: int = 5, min_score: float = 0.0) -> List[Dict]:
        """检索相关文档

        Args:
            query: 查询文本
            k: 返回数量
            min_score: 最低相似度阈值

        Returns:
            过滤后的检索结果
        """
        results = self.vector_store.search(query, k=k)

        # 按分数过滤
        if min_score > 0:
            results = [r for r in results if r['score'] >= min_score]

        return results

    def retrieve_with_context(self, query: str, k: int = 5) -> str:
        """检索并格式化为上下文字符串

        Args:
            query: 查询文本
            k: 返回数量

        Returns:
            格式化的上下文字符串，适合直接放入prompt
        """
        results = self.retrieve(query, k=k)

        if not results:
            return "未找到相关文档内容。"

        context_parts = []
        for i, result in enumerate(results, 1):
            source = result['metadata'].get('source', '未知来源')
            score = result['score']
            context_parts.append(
                f"[文档{i}] (来源: {source}, 相关度: {score:.2f})\n{result['document']}"
            )

        return "\n\n---\n\n".join(context_parts)

    def get_stats(self) -> Dict:
        """获取检索器统计信息"""
        return self.vector_store.get_collection_stats()
