"""
LLM 封装 - 统一大模型调用接口（主：Gemini；备：DeepSeek，429 时自动切换）
"""
from typing import List, Dict, Optional, Any
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage

import sys
sys.path.append("../..")
from config.settings import settings


# 触发 fallback 的错误特征︌命中任一即切备用模型
_QUOTA_ERROR_MARKERS = (
    "429",
    "RESOURCE_EXHAUSTED",
    "quota",
    "exceeded",
    "rate limit",
    "ratelimit",
)


def _is_quota_error(err: Exception) -> bool:
    """判断异常是否为配额/限频类错误"""
    msg = str(err).lower()
    return any(m.lower() in msg for m in _QUOTA_ERROR_MARKERS)


class FallbackChatModel:
    """主备双模型包装器：主模型 429/配额错误时自动切换到备用模型

    兼容 LangChain 的 .invoke(messages) 接口（项目内统一用法）。
    切换后设置熍断标志 _use_fallback=True，后续直接走备用，直到进程重启。
    """

    def __init__(self, primary, fallback):
        self.primary = primary
        self.fallback = fallback
        self._use_fallback = False

    def _active(self):
        return self.fallback if (self._use_fallback and self.fallback is not None) else self.primary

    def invoke(self, messages, **kwargs):
        # 一旦熍断直接走备用
        if self._use_fallback and self.fallback is not None:
            return self.fallback.invoke(messages, **kwargs)

        try:
            return self.primary.invoke(messages, **kwargs)
        except Exception as e:
            if self.fallback is not None and _is_quota_error(e):
                # print(
                #     # f"[LLM] 主模型触发配额/限频错误，自动切换到 DeepSeek：{e}"
                #     f"[LLM] 主模型触发配额/限频错误，自动切换到 DeepSeek"
                # )
                self._use_fallback = True
                return self.fallback.invoke(messages, **kwargs)
            raise

    # 代理其他属性（如 .model_name / .temperature 等）到当前活跃模型
    def __getattr__(self, name: str) -> Any:
        if name in ("primary", "fallback", "_use_fallback"):
            raise AttributeError(name)
        return getattr(self._active(), name)


def _build_gemini(temperature, model, max_tokens) -> Optional[ChatGoogleGenerativeAI]:
    if not settings.GOOGLE_API_KEY:
        return None
    return ChatGoogleGenerativeAI(
        model=model or settings.LLM_MODEL,
        google_api_key=settings.GOOGLE_API_KEY,
        temperature=settings.LLM_TEMPERATURE if temperature is None else temperature,
        max_output_tokens=max_tokens or settings.LLM_MAX_TOKENS,
    )


def _build_deepseek(temperature, max_tokens):
    """构造 DeepSeek 模型（OpenAI 兼容接口），未配置时返回 None"""
    if not settings.DEEPSEEK_API_KEY:
        return None
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        print("[LLM] 未安装 langchain-openai，无法启用 DeepSeek fallback。请：pip install langchain-openai")
        return None
    return ChatOpenAI(
        model=settings.DEEPSEEK_MODEL,
        api_key=settings.DEEPSEEK_API_KEY,
        base_url=settings.DEEPSEEK_BASE_URL,
        temperature=settings.LLM_TEMPERATURE if temperature is None else temperature,
        max_tokens=max_tokens or settings.LLM_MAX_TOKENS,
    )


def build_chat_model(temperature: Optional[float] = None,
                     model: Optional[str] = None,
                     max_tokens: Optional[int] = None):
    """统一构造 Chat 模型（主 Gemini，备 DeepSeek）

    行为：
    - 两者都配置 → 返回 FallbackChatModel，遇 429/quota 错误自动切换。
    - 只配 Gemini → 返回原生 Gemini 模型。
    - 只配 DeepSeek → 返回原生 DeepSeek 模型。
    - 两者都未配 → 抛出明确异常。
    """
    primary = _build_gemini(temperature, model, max_tokens)
    fallback = _build_deepseek(temperature, max_tokens)

    if primary is None and fallback is None:
        raise RuntimeError(
            "未配置任何 LLM API Key：请在 .env 中设置 GOOGLE_API_KEY 或 DEEPSEEK_API_KEY。"
        )
    if primary is None:
        return fallback
    if fallback is None:
        return primary
    return FallbackChatModel(primary, fallback)


def build_embeddings(model: Optional[str] = None) -> GoogleGenerativeAIEmbeddings:
    """统一构造 Gemini Embedding 模型（DeepSeek 暂不提供 embedding）"""
    name = model or settings.EMBEDDING_MODEL
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
