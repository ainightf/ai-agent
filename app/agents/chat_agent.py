"""通用对话Agent - 处理闲聊和常识问答"""
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from config.settings import settings
from app.llm.llm import build_chat_model
from app.memory.memory import PersistentMemory
from app.memory.shared_memory import SharedMemory


class ChatAgent:
    """通用对话Agent - 处理闲聊和常识问答"""
    
    def __init__(self):
        self.llm = build_chat_model(temperature=0.7)
        self.memory = PersistentMemory()
        self.shared_memory = SharedMemory()
    
    def run(self, query: str, session_id: str) -> str:
        """执行对话
        
        Args:
            query: 用户输入
            session_id: 会话ID
            
        Returns:
            回复内容
        """
        try:
            # 1. 获取对话历史
            history = self.memory.get_history(session_id, agent_name="chat", limit=10)
            
            # 2. 获取共享上下文（了解其他Agent做了什么）
            shared_context = self.shared_memory.get_agent_summary(session_id)
            
            # 3. 构建消息列表
            system_prompt = f"""你是一个友好、知识渊博的AI助手。你可以回答各种问题、进行闲聊。

背景信息（来自其他Agent的上下文）：
{shared_context}

规则：
1. 回答简洁有帮助
2. 如果用户的问题与公司文档相关，提示可以使用 /upload 上传文档后再询问
3. 如果涉及商标品牌查询，提示用户直接问品牌名即可
4. 保持友好和专业"""

            messages = [SystemMessage(content=system_prompt)]
            
            # 添加历史消息
            for msg in history[-6:]:
                if msg['role'] == 'user':
                    messages.append(HumanMessage(content=msg['content']))
                elif msg['role'] == 'assistant':
                    messages.append(AIMessage(content=msg['content']))
            
            # 添加当前消息
            messages.append(HumanMessage(content=query))
            
            # 4. 调用LLM
            response = self.llm.invoke(messages)
            return response.content
            
        except Exception as e:
            return f"对话出错：{str(e)}"
