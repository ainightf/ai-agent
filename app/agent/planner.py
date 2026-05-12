"""
Planner 规划器 - 决策是否需要调用工具
"""
import json
from typing import Dict, Any, Optional, List
from langchain_core.messages import HumanMessage, AIMessage

import sys
sys.path.append("../..")
from config.settings import settings
from app.llm.llm import LLM
from app.llm.prompt import get_planner_prompt


class PlannerDecision:
    """规划器决策结果"""
    
    def __init__(
        self,
        need_tool: bool,
        tool_name: Optional[str] = None,
        tool_input: Optional[str] = None,
        reasoning: str = ""
    ):
        self.need_tool = need_tool
        self.tool_name = tool_name
        self.tool_input = tool_input
        self.reasoning = reasoning
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "need_tool": self.need_tool,
            "tool_name": self.tool_name,
            "tool_input": self.tool_input,
            "reasoning": self.reasoning
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlannerDecision":
        return cls(
            need_tool=data.get("need_tool", False),
            tool_name=data.get("tool_name"),
            tool_input=data.get("tool_input"),
            reasoning=data.get("reasoning", "")
        )


class Planner:
    """规划器 - 决定是否需要使用工具"""
    
    def __init__(self, llm: Optional[LLM] = None):
        """
        初始化规划器
        
        Args:
            llm: LLM 实例
        """
        self.llm = llm or LLM()
        self.prompt = get_planner_prompt()
    
    def plan(
        self,
        user_input: str,
        tools_description: str,
        chat_history: List[Dict[str, str]] = None
    ) -> PlannerDecision:
        """
        分析用户输入，决定是否需要使用工具
        
        Args:
            user_input: 用户输入
            tools_description: 可用工具描述
            chat_history: 对话历史
            
        Returns:
            规划决策结果
        """
        chat_history = chat_history or []
        
        # 构建消息
        messages = []
        
        # 添加对话历史
        for msg in chat_history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
        
        # 构建提示词
        prompt_text = self.prompt.format(
            tools=tools_description,
            chat_history=messages,
            input=user_input
        )
        
        # 调用 LLM
        response = self.llm.chat([HumanMessage(content=prompt_text)])
        
        # 解析结果
        return self._parse_response(response)
    
    def _parse_response(self, response: str) -> PlannerDecision:
        """
        解析 LLM 响应
        
        Args:
            response: LLM 响应
            
        Returns:
            规划决策结果
        """
        try:
            # 尝试提取 JSON
            start = response.find("{")
            end = response.rfind("}") + 1
            if start != -1 and end > start:
                json_str = response[start:end]
                data = json.loads(json_str)
                return PlannerDecision.from_dict(data)
        except json.JSONDecodeError:
            pass
        
        # 解析失败，返回默认决策（不使用工具）
        return PlannerDecision(
            need_tool=False,
            reasoning="无法解析规划结果，将直接回答用户问题。"
        )
    
    def get_tools_description(self, tools: list) -> str:
        """
        获取工具描述字符串
        
        Args:
            tools: 工具列表
            
        Returns:
            工具描述字符串
        """
        descriptions = []
        for tool in tools:
            descriptions.append(f"- {tool.name}: {tool.description}")
        return "\n".join(descriptions)
