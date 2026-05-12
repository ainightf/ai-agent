"""摘要Agent - 总结对话和文档内容"""
from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import settings
from app.llm.llm import build_chat_model
from app.memory.memory import PersistentMemory
from app.memory.shared_memory import SharedMemory


class SummaryAgent:
    """摘要生成Agent - 总结对话历史和文档内容"""
    
    def __init__(self):
        self.llm = build_chat_model(temperature=0.3)
        self.memory = PersistentMemory()
        self.shared_memory = SharedMemory()
    
    def run(self, query: str, session_id: str) -> str:
        """生成摘要
        
        Args:
            query: 用户请求（如"总结一下之前的对话"）
            session_id: 会话ID
            
        Returns:
            摘要内容
        """
        try:
            # 1. 获取完整会话历史
            full_history = self.memory.get_full_session_history(session_id)
            
            # 2. 获取共享上下文
            shared_context = self.shared_memory.get_agent_summary(session_id)
            
            # 3. 格式化历史
            history_str = self._format_full_history(full_history)
            
            # 4. 生成摘要
            system_prompt = """你是一个专业的摘要助手。根据对话历史和上下文信息，生成简洁、有条理的摘要。

规则：
1. 按主题或时间线组织摘要
2. 突出关键信息和结论
3. 标注涉及了哪些Agent（文档问答、品牌查询等）
4. 如果对话较短，简要概括即可"""

            user_prompt = f"""用户请求：{query}

对话历史：
{history_str}

共享上下文信息：
{shared_context}

请生成摘要："""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm.invoke(messages)
            summary = response.content
            
            # 5. 保存摘要到共享上下文
            self.shared_memory.set_context(
                session_id,
                "session_summary",
                summary[:500],
                "summary"
            )
            
            return summary
            
        except Exception as e:
            return f"摘要生成出错：{str(e)}"
    
    def _format_full_history(self, history: list) -> str:
        """格式化完整历史"""
        if not history:
            return "暂无对话历史"
        
        lines = []
        for msg in history[-30:]:  # 最多取30条
            agent = msg.get('agent_name', 'unknown')
            role = msg['role']
            content = msg['content'][:200]
            lines.append(f"[{agent}] {role}: {content}")
        
        return "\n".join(lines)
