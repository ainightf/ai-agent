"""
Agent 核心逻辑 - 智能体主控制器
"""
from typing import List, Optional, Dict, Any
from langchain_core.tools import Tool

import sys
sys.path.append("../..")
from config.settings import settings
from app.llm.llm import LLM
from app.llm.prompt import AGENT_SYSTEM_PROMPT
from app.memory.memory import Memory
from app.agent.planner import Planner
from app.agent.executor import Executor
from app.tools.tavily_tool import TavilyTool
from app.tools.rag_tool import RAGTool


class Agent:
    """智能体主类"""
    
    def __init__(
        self,
        llm: Optional[LLM] = None,
        memory: Optional[Memory] = None,
        tools: Optional[List[Tool]] = None
    ):
        """
        初始化智能体
        
        Args:
            llm: LLM 实例
            memory: 记忆实例
            tools: 工具列表
        """
        self.llm = llm or LLM()
        self.memory = memory or Memory()
        self.planner = Planner(llm=self.llm)
        self.executor = Executor()
        
        # 初始化默认工具
        self._init_default_tools()
        
        # 注册额外工具
        if tools:
            self.executor.register_tools(tools)
    
    def _init_default_tools(self) -> None:
        """初始化默认工具"""
        # Tavily 搜索工具
        try:
            tavily_tool = TavilyTool()
            self.executor.register_tool(tavily_tool.get_tool())
        except Exception:
            pass  # Tavily 未配置时跳过
        
        # RAG 工具
        try:
            rag_tool = RAGTool(llm=self.llm)
            self.executor.register_tool(rag_tool.get_tool())
        except Exception:
            pass  # RAG 未配置时跳过
    
    def chat(self, user_input: str) -> str:
        """
        与用户对话
        
        Args:
            user_input: 用户输入
            
        Returns:
            AI 回复
        """
        # 获取工具描述
        tools = self.executor.get_tools()
        tools_description = self.planner.get_tools_description(tools)
        
        # 规划决策
        decision = self.planner.plan(
            user_input=user_input,
            tools_description=tools_description,
            chat_history=self.memory.get_history()
        )
        
        # 执行工具（如果需要）
        tool_result = ""
        if decision.need_tool:
            execution_result = self.executor.execute(decision)
            if execution_result.success:
                tool_result = f"\n\n工具执行结果：\n{execution_result.output}"
            else:
                tool_result = f"\n\n工具执行失败：{execution_result.error}"
        
        # 生成最终回复
        context = ""
        if tool_result:
            context = f"根据以下信息回答用户问题：{tool_result}\n\n"
        
        response = self.llm.chat_with_history(
            user_input=context + user_input if context else user_input,
            history=self.memory.get_history(),
            system_prompt=AGENT_SYSTEM_PROMPT.format(tools=tools_description)
        )
        
        # 保存到记忆
        self.memory.add_message_pair(user_input, response)
        
        return response
    
    def reset(self) -> None:
        """重置对话"""
        self.memory.clear()
    
    def add_tool(self, tool: Tool) -> None:
        """添加工具"""
        self.executor.register_tool(tool)
    
    def get_history(self) -> List[Dict[str, str]]:
        """获取对话历史"""
        return self.memory.get_history()
