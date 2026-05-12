"""
配置文件 - 集中管理所有配置项
"""
import os
from dotenv import load_dotenv

# 加载环境变量（override=True 防止被 shell 里旧值劫持）
load_dotenv(override=True)


class Settings:
    """应用配置类"""

    # LLM 配置（Gemini）
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gemini-2.0-flash")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "2048"))

    # DeepSeek 配置（作为 Gemini 额度耗尽时的 fallback）
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_MODEL: str = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro")
    DEEPSEEK_BASE_URL: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")

    # 兼容保留（已切换到 Gemini，这两项不再使用）
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    # Tavily 搜索配置
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")

    # RAG 配置
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-004")
    VECTOR_STORE_PATH: str = os.getenv("VECTOR_STORE_PATH", "./data/vector_store")
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    TOP_K: int = int(os.getenv("TOP_K", "3"))
    
    # Memory 配置
    MAX_HISTORY_LENGTH: int = int(os.getenv("MAX_HISTORY_LENGTH", "10"))

    # ChromaDB 配置
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
    CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "company_docs")

    # SQLite 配置
    SQLITE_DB_PATH: str = os.getenv("SQLITE_DB_PATH", "./data/sqlite/memory.db")

    # 邮件服务配置
    EMAIL_ADDRESS: str = os.getenv("EMAIL_ADDRESS", "")
    EMAIL_PASSWORD: str = os.getenv("EMAIL_PASSWORD", "")
    IMAP_SERVER: str = os.getenv("IMAP_SERVER", "imap.qq.com")
    SMTP_SERVER: str = os.getenv("SMTP_SERVER", "smtp.qq.com")
    IMAP_PORT: int = int(os.getenv("IMAP_PORT", "993"))
    SMTP_PORT: int = int(os.getenv("SMTP_PORT", "465"))

    # 文档存储目录
    DOCUMENTS_DIR: str = os.getenv("DOCUMENTS_DIR", "./data/documents")


settings = Settings()


def is_deepseek_configured() -> bool:
    """DeepSeek 是否配置就绪（允许作为 fallback）"""
    return bool(settings.DEEPSEEK_API_KEY)


settings.is_deepseek_configured = is_deepseek_configured

# 模块级便捷导出
SQLITE_DB_PATH = settings.SQLITE_DB_PATH
