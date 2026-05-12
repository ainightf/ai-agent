"""
RAG 工具 - 知识库查询能力
"""
from typing import Optional
from langchain_core.tools import Tool

import sys
sys.path.append("../..")
from config.settings import settings
from app.rag.retriever import ChromaRetriever
from app.llm.llm import LLM
from app.llm.prompt import get_rag_prompt


class RAGTool:
    """RAG 知识库查询工具"""
    
    def __init__(
        self,
        retriever: Optional[ChromaRetriever] = None,
        llm: Optional[LLM] = None
    ):
        """
        初始化 RAG 工具
        
        Args:
            retriever: 检索器实例
            llm: LLM 实例
        """
        self.retriever = retriever or ChromaRetriever()
        self.llm = llm or LLM()
        self.prompt = get_rag_prompt()
    
    def query(self, question: str) -> str:
        """
        查询知识库
        
        Args:
            question: 用户问题
            
        Returns:
            基于知识库的回答
        """
        # 检索相关文档
        context = self.retriever.retrieve_with_context(question)
        
        if not context or context == "未找到相关文档内容。":
            return "知识库中未找到相关信息。"
        
        # 构建提示词
        prompt_text = self.prompt.format(
            context=context,
            question=question
        )
        
        # 调用 LLM 生成回答
        from langchain_core.messages import HumanMessage
        response = self.llm.chat([HumanMessage(content=prompt_text)])
        
        return response
    
    def add_knowledge(self, texts: list, metadatas: Optional[list] = None) -> str:
        """
        添加知识到知识库
        
        Args:
            texts: 文本列表
            metadatas: 元数据列表
            
        Returns:
            添加结果消息
        """
        try:
            count = self.retriever.vector_store.add_documents(texts, metadatas)
            return f"成功添加 {count} 个文档块到知识库。"
        except Exception as e:
            return f"添加知识失败：{str(e)}"
    
    def get_tool(self) -> Tool:
        """
        获取 LangChain Tool 实例
        
        Returns:
            Tool 实例
        """
        return Tool(
            name="rag_query",
            description="从知识库中查询信息。当需要查找项目文档、内部知识、或已存储的信息时使用。",
            func=self.query
        )
