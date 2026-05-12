"""
Executor 执行器 - 执行工具调用
"""
from typing import Dict, Any, Optional, List
from langchain_core.tools import Tool

import sys
sys.path.append("../..")
from app.agent.planner import PlannerDecision


class ExecutionResult:
    """执行结果"""
    
    def __init__(
        self,
        success: bool,
        tool_name: str,
        tool_input: str,
        output: str,
        error: Optional[str] = None
    ):
        self.success = success
        self.tool_name = tool_name
        self.tool_input = tool_input
        self.output = output
        self.error = error
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "tool_name": self.tool_name,
            "tool_input": self.tool_input,
            "output": self.output,
            "error": self.error
        }


class Executor:
    """执行器 - 执行工具调用"""
    
    def __init__(self, tools: List[Tool] = None):
        """
        初始化执行器
        
        Args:
            tools: 工具列表
        """
        self.tools: Dict[str, Tool] = {}
        if tools:
            for tool in tools:
                self.register_tool(tool)
    
    def register_tool(self, tool: Tool) -> None:
        """
        注册工具
        
        Args:
            tool: 工具实例
        """
        self.tools[tool.name] = tool
    
    def register_tools(self, tools: List[Tool]) -> None:
        """
        批量注册工具
        
        Args:
            tools: 工具列表
        """
        for tool in tools:
            self.register_tool(tool)
    
    def execute(self, decision: PlannerDecision) -> ExecutionResult:
        """
        执行工具调用
        
        Args:
            decision: 规划决策
            
        Returns:
            执行结果
        """
        if not decision.need_tool:
            return ExecutionResult(
                success=True,
                tool_name="",
                tool_input="",
                output="无需调用工具",
                error=None
            )
        
        tool_name = decision.tool_name
        tool_input = decision.tool_input
        
        if tool_name not in self.tools:
            return ExecutionResult(
                success=False,
                tool_name=tool_name,
                tool_input=tool_input,
                output="",
                error=f"工具 '{tool_name}' 不存在"
            )
        
        try:
            tool = self.tools[tool_name]
            output = tool.invoke(tool_input)
            return ExecutionResult(
                success=True,
                tool_name=tool_name,
                tool_input=tool_input,
                output=output,
                error=None
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                tool_name=tool_name,
                tool_input=tool_input,
                output="",
                error=str(e)
            )
    
    def execute_tool(self, tool_name: str, tool_input: str) -> ExecutionResult:
        """
        直接执行指定工具
        
        Args:
            tool_name: 工具名称
            tool_input: 工具输入
            
        Returns:
            执行结果
        """
        decision = PlannerDecision(
            need_tool=True,
            tool_name=tool_name,
            tool_input=tool_input
        )
        return self.execute(decision)
    
    def get_tool_names(self) -> List[str]:
        """获取所有已注册的工具名称"""
        return list(self.tools.keys())
    
    def get_tools(self) -> List[Tool]:
        """获取所有已注册的工具"""
        return list(self.tools.values())
