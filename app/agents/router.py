"""路由Agent - 负责意图识别和任务分发"""
import re
from typing import Dict, Optional
from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import settings
from app.llm.llm import build_chat_model
from app.memory.memory import PersistentMemory
from app.memory.shared_memory import SharedMemory
from app.agents.doc_qa_agent import DocQAAgent
from app.agents.email_agent import EmailAgent
from app.agents.summary_agent import SummaryAgent
from app.agents.chat_agent import ChatAgent


# 邮件序号指代正则：#3 / 第3封 / 第三封 / 3封邮件 / 看第 / 读第 / 摘要下第 ...
_EMAIL_SEQ_PATTERN = re.compile(
    r"#\s*\d+"
    r"|第\s*[一二两三四五六七八九十\d]+\s*封"
    r"|\d+\s*封邮件"
    r"|(?:看|读|展开|打开|摘要)\s*(?:一下|下)?\s*第"
)


class RouterAgent:
    """路由Agent - 分析用户意图并分发到合适的Agent"""

    AGENT_MAP = {
        "doc_qa": "文档问答",
        "email": "邮件助手",
        "summary": "摘要总结",
        "chat": "通用对话"
    }

    # === 方案A：规则前置关键词 ===
    _RULE_KEYWORDS = {
        "email": [
            "邮件", "邮箱", "收件箱", "发邮件", "查邮件", "写邮件",
            "回复邮件", "邮件摘要", "未读邮件",
            "inbox", "email", "mail",
        ],
        "doc_qa": [
            "文档", "知识库", "手册", "pdf", "报告", "档案", "说明书",
        ],
        "summary": [
            "总结", "概括", "摘要一下", "摘要对话", "summarize",
        ],
    }

    def __init__(self):
        self.llm = build_chat_model(temperature=0)
        self.memory = PersistentMemory()
        self.shared_memory = SharedMemory()

        # 初始化各Agent
        self.agents = {
            "doc_qa": DocQAAgent(),
            "email": EmailAgent(),
            "summary": SummaryAgent(),
            "chat": ChatAgent()
        }

        # 方案B：进程内简易 LRU 缓存（query -> intent）
        self._intent_cache: Dict[str, str] = {}
        self._intent_cache_max = 256

    # === 规则匹配 ===
    def _rule_classify(self, user_input: str) -> Optional[str]:
        text = user_input.lower().strip()
        if not text:
            return None
        # ① 序号式指代邮件 → email（解决“看下第七封”这类没邮件关键词的情况）
        if _EMAIL_SEQ_PATTERN.search(user_input):
            return "email"
        # ② 关键词匹配
        for agent_key, keywords in self._RULE_KEYWORDS.items():
            for kw in keywords:
                if kw.lower() in text:
                    return agent_key
        return None
    
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
        agent_key = self._classify_intent(user_input, context, recent_history, session_id)
        
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
    
    def _classify_intent(self, user_input: str, context: str, history: str, session_id: Optional[str] = None) -> str:
        """意图分类（方案A规则前置 + 方案B缓存 + LLM 兄底 + session 回退）"""
        # 规范化缓存 key
        cache_key = user_input.strip().lower()

        # A. 规则前置
        hit = self._rule_classify(user_input)
        if hit:
            return hit

        # B. 缓存
        if cache_key in self._intent_cache:
            return self._intent_cache[cache_key]

        system_prompt = """你是一个意图分类器。根据用户输入，判断应该由哪个Agent处理。

可选Agent：
- doc_qa: 文档问答 — 用户询问与公司文档、知识库相关的问题（如"文档里提到了什么"、"帮我查下XX文件的内容"）
- email: 邮件助手 — 用户涉及邮件相关操作（如"查看邮件"、"收件箱"、"发邮件"、"查邮件"、"inbox"、"email"、"邮件摘要"、"发送邮件给XX"）
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
            resolved = None
            if intent in self.agents:
                resolved = intent
            else:
                for key in self.agents:
                    if key in intent:
                        resolved = key
                        break
            if resolved is None:
                resolved = "chat"

            # 写入缓存（容量满时简易淘汰）
            if len(self._intent_cache) >= self._intent_cache_max:
                try:
                    self._intent_cache.pop(next(iter(self._intent_cache)))
                except StopIteration:
                    pass
            self._intent_cache[cache_key] = resolved
            return resolved
        except Exception as e:
            print(f"意图分类出错: {e}")
            # LLM 失败（如 429 配额耗尽）：优先回退到该 session 上次使用的 agent，保留上下文
            if session_id:
                try:
                    row = self.shared_memory.get_context(session_id, "last_agent")
                    if row and row.get("value") in self.agents:
                        print(f"[Router] 回退到上次 agent：{row['value']}")
                        return row["value"]
                except Exception:
                    pass
            return "chat"
    
    def get_available_agents(self) -> Dict:
        """获取所有可用Agent信息"""
        return self.AGENT_MAP
