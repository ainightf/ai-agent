"""
LLM 封装 - 统一大模型调用接口（Gemini）
"""
from typing import List, Dict, Optional
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage

import sys
sys.path.append("../..")
from config.settings import settings


def build_chat_model(temperature: Optional[float] = None,
                     model: Optional[str] = None,
                     max_tokens: Optional[int] = None) -> ChatGoogleGenerativeAI:
    """统一构造 Gemini Chat 模型"""
    return ChatGoogleGenerativeAI(
        model=model or settings.LLM_MODEL,
        google_api_key=settings.GOOGLE_API_KEY,
        temperature=settings.LLM_TEMPERATURE if temperature is None else temperature,
        max_output_tokens=max_tokens or settings.LLM_MAX_TOKENS,
    )


def build_embeddings(model: Optional[str] = None) -> GoogleGenerativeAIEmbeddings:
    """统一构造 Gemini Embedding 模型"""
    name = model or settings.EMBEDDING_MODEL
    # Gemini embedding 模型名需要带 "models/" 前缀
    if not name.startswith("models/"):
        name = f"models/{name}"
    return GoogleGenerativeAIEmbeddings(
        model=name,
        google_api_key=settings.GOOGLE_API_KEY,
    )


class LLM:
    """大模型封装类（Gemini）"""

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        self.model = model or settings.LLM_MODEL
        self.temperature = temperature if temperature is not None else settings.LLM_TEMPERATURE
        self.max_tokens = max_tokens or settings.LLM_MAX_TOKENS
        self.llm = build_chat_model(self.temperature, self.model, self.max_tokens)

    def chat(self, messages: List[BaseMessage]) -> str:
        """发送消息并获取回复"""
        response = self.llm.invoke(messages)
        return response.content

    def chat_with_history(
        self,
        user_input: str,
        history: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
    ) -> str:
        """带历史记录的对话"""
        messages: List[BaseMessage] = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        for msg in history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
        messages.append(HumanMessage(content=user_input))
        return self.chat(messages)

    def get_llm(self) -> ChatGoogleGenerativeAI:
        """获取底层 LLM 实例"""
        return self.llm
