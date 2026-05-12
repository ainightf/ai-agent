"""路由Agent - 负责意图识别和任务分发"""
from typing import Dict, Optional
from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import settings
from app.llm.llm import build_chat_model
from app.memory.memory import PersistentMemory
from app.memory.shared_memory import SharedMemory
from app.agents.doc_qa_agent import DocQAAgent
from app.agents.brand_agent import BrandAgent
from app.agents.summary_agent import SummaryAgent
from app.agents.chat_agent import ChatAgent


class RouterAgent:
    """路由Agent - 分析用户意图并分发到合适的Agent"""
    
    AGENT_MAP = {
        "doc_qa": "文档问答",
        "brand": "品牌商标查询",
        "summary": "摘要总结",
        "chat": "通用对话"
    }
    
    def __init__(self):
        self.llm = build_chat_model(temperature=0)
        self.memory = PersistentMemory()
        self.shared_memory = SharedMemory()
        
        # 初始化各Agent
        self.agents = {
            "doc_qa": DocQAAgent(),
            "brand": BrandAgent(),
            "summary": SummaryAgent(),
            "chat": ChatAgent()
        }
    
    def route(self, user_input: str, session_id: str) -> Dict:
        """路由用户输入到合适的Agent
        
        Args:
            user_input: 用户输入
            session_id: 会话ID
            
        Returns:
            {"agent_name": str, "response": str}
        """
        # 1. 获取上下文
        context = self.shared_memory.get_agent_summary(session_id)
        recent_history = self.memory.get_recent_context(session_id, limit=5)
        
        # 2. 意图识别
        agent_key = self._classify_intent(user_input, context, recent_history)
        
        # 3. 记录用户消息
        self.memory.add_message(session_id, "router", "user", user_input)
        
        # 4. 分发到目标Agent
        target_agent = self.agents[agent_key]
        response = target_agent.run(user_input, session_id)
        
        # 5. 记录Agent响应
        self.memory.add_message(session_id, agent_key, "assistant", response)
        
        # 6. 更新共享上下文
        self.shared_memory.set_context(
            session_id, 
            "last_agent", 
            agent_key, 
            "router"
        )
        self.shared_memory.set_context(
            session_id,
            "last_query",
            user_input[:100],
            "router"
        )
        
        return {
            "agent_name": self.AGENT_MAP.get(agent_key, agent_key),
            "response": response
        }
    
    def _classify_intent(self, user_input: str, context: str, history: str) -> str:
        """使用LLM进行意图分类"""
        system_prompt = """你是一个意图分类器。根据用户输入，判断应该由哪个Agent处理。

可选Agent：
- doc_qa: 文档问答 — 用户询问与公司文档、知识库相关的问题（如"文档里提到了什么"、"帮我查下XX文件的内容"）
- brand: 品牌商标查询 — 用户询问品牌、商标注册、商标审核相关问题（如"查下XX品牌的商标"、"XX商标注册了吗"）
- summary: 摘要总结 — 用户要求总结对话、文档、或之前的内容（如"总结一下"、"帮我概括"）
- chat: 通用对话 — 闲聊、常识问答、与上述不相关的其他问题

只返回agent的key，不要返回其他内容。例如返回: doc_qa"""

        user_prompt = f"""当前对话上下文：
{context}

最近对话历史：
{history}

用户最新输入：{user_input}

请判断应由哪个Agent处理，只返回key："""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        try:
            response = self.llm.invoke(messages)
            intent = response.content.strip().lower()
            
            # 验证返回值
            if intent in self.agents:
                return intent
            
            # 模糊匹配
            for key in self.agents:
                if key in intent:
                    return key
            
            # 默认fallback到chat
            return "chat"
        except Exception as e:
            print(f"意图分类出错: {e}, 使用默认chat")
            return "chat"
    
    def get_available_agents(self) -> Dict:
        """获取所有可用Agent信息"""
        return self.AGENT_MAP
