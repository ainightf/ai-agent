"""文档问答Agent - 基于RAG检索回答文档相关问题"""
from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import settings
from app.llm.llm import build_chat_model
from app.rag.retriever import ChromaRetriever
from app.memory.memory import PersistentMemory
from app.memory.shared_memory import SharedMemory


class DocQAAgent:
    """文档问答Agent - 使用ChromaDB检索文档并回答"""

    def __init__(self):
        self.llm = build_chat_model(temperature=0.3)
        self.retriever = ChromaRetriever()
        self.memory = PersistentMemory()
        self.shared_memory = SharedMemory()

    def run(self, query: str, session_id: str) -> str:
        """执行文档问答

        Args:
            query: 用户问题
            session_id: 会话ID

        Returns:
            回答字符串
        """
        try:
            # 1. 检索相关文档
            context = self.retriever.retrieve_with_context(query, k=5)

            # 2. 获取对话历史
            history = self.memory.get_history(
                session_id, agent_name="doc_qa", limit=5)
            history_str = self._format_history(history)

            # 3. 构建prompt
            system_prompt = """你是一个专业的文档问答助手。基于检索到的文档内容回答用户问题。

规则：
1. 只根据提供的文档内容回答，不要编造信息
2. 如果文档中没有相关内容，明确告知用户
3. 引用来源时注明是哪个文档
4. 回答要简洁准确"""

            user_prompt = f"""对话历史：
{history_str}

检索到的文档内容：
{context}

用户问题：{query}

请基于文档内容回答："""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]

            response = self.llm.invoke(messages)
            answer = response.content

            # 4. 保存关键信息到共享上下文
            self.shared_memory.set_context(
                session_id,
                "last_doc_query",
                {"query": query, "answer_preview": answer[:100]},
                "doc_qa"
            )

            return answer

        except Exception as e:
            return f"文档问答出错：{str(e)}"

    def _format_history(self, history: list) -> str:
        """格式化对话历史"""
        if not history:
            return "无历史对话"
        lines = []
        for msg in history[-5:]:
            role = "用户" if msg['role'] == 'user' else "助手"
            lines.append(f"{role}: {msg['content'][:150]}")
        return "\n".join(lines)
