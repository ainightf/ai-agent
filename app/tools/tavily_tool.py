"""
Tavily 搜索工具 - 网络搜索能力
"""
from typing import Optional, List, Dict, Any
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import Tool

import sys
sys.path.append("../..")
from config.settings import settings


class TavilyTool:
    """Tavily 搜索工具封装"""
    
    def __init__(self, max_results: int = 5):
        """
        初始化 Tavily 搜索工具
        
        Args:
            max_results: 最大返回结果数
        """
        self.max_results = max_results
        
        # 初始化 Tavily 搜索
        self.search = TavilySearchResults(
            max_results=self.max_results,
            tavily_api_key=settings.TAVILY_API_KEY
        )
    
    def search_web(self, query: str) -> List[Dict[str, Any]]:
        """
        执行网络搜索
        
        Args:
            query: 搜索查询
            
        Returns:
            搜索结果列表
        """
        try:
            results = self.search.invoke(query)
            return results
        except Exception as e:
            return [{"error": str(e)}]
    
    def search_and_format(self, query: str) -> str:
        """
        搜索并格式化结果
        
        Args:
            query: 搜索查询
            
        Returns:
            格式化的搜索结果字符串
        """
        results = self.search_web(query)
        
        if not results:
            return "未找到相关搜索结果。"
        
        if isinstance(results[0], dict) and "error" in results[0]:
            return f"搜索出错：{results[0]['error']}"
        
        formatted = []
        for i, result in enumerate(results, 1):
            title = result.get("title", "无标题")
            content = result.get("content", "无内容")
            url = result.get("url", "")
            formatted.append(f"[{i}] {title}\n{content}\n链接: {url}")
        
        return "\n\n".join(formatted)
    
    def get_tool(self) -> Tool:
        """
        获取 LangChain Tool 实例
        
        Returns:
            Tool 实例
        """
        return Tool(
            name="tavily_search",
            description="搜索互联网获取最新信息。当需要查找实时信息、新闻、或任何需要网络搜索的内容时使用。",
            func=self.search_and_format
        )
