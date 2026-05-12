"""持久化记忆管理模块 - 基于SQLite"""
from typing import List, Dict, Optional
from datetime import datetime

from app.database.sqlite_db import SQLiteDatabase


class PersistentMemory:
    """基于SQLite的持久化对话记忆
    
    支持：
    - 按session_id隔离不同会话
    - 按agent_name隔离不同Agent的对话
    - 获取完整会话历史（所有Agent）
    """
    
    def __init__(self, db: SQLiteDatabase = None):
        self.db = db or SQLiteDatabase()
    
    def add_message(self, session_id: str, agent_name: str, role: str, content: str):
        """添加一条对话消息
        
        Args:
            session_id: 会话ID
            agent_name: Agent名称
            role: 角色 (user/assistant/system)
            content: 消息内容
        """
        self.db.execute(
            "INSERT INTO conversations (session_id, agent_name, role, content) VALUES (?, ?, ?, ?)",
            (session_id, agent_name, role, content)
        )
    
    def get_history(self, session_id: str, agent_name: Optional[str] = None, limit: int = 20) -> List[Dict]:
        """获取对话历史
        
        Args:
            session_id: 会话ID
            agent_name: Agent名称（为None时获取所有Agent的记录）
            limit: 返回最近N条
            
        Returns:
            消息列表 [{"role": ..., "content": ..., "agent_name": ..., "timestamp": ...}]
        """
        if agent_name:
            return self.db.fetch_all(
                """SELECT role, content, agent_name, timestamp 
                   FROM conversations 
                   WHERE session_id = ? AND agent_name = ? 
                   ORDER BY timestamp DESC LIMIT ?""",
                (session_id, agent_name, limit)
            )[::-1]  # 反转为时间正序
        else:
            return self.db.fetch_all(
                """SELECT role, content, agent_name, timestamp 
                   FROM conversations 
                   WHERE session_id = ? 
                   ORDER BY timestamp DESC LIMIT ?""",
                (session_id, limit)
            )[::-1]
    
    def get_full_session_history(self, session_id: str) -> List[Dict]:
        """获取完整会话历史（所有Agent）"""
        return self.db.fetch_all(
            """SELECT role, content, agent_name, timestamp 
               FROM conversations 
               WHERE session_id = ? 
               ORDER BY timestamp ASC""",
            (session_id,)
        )
    
    def get_recent_context(self, session_id: str, limit: int = 5) -> str:
        """获取最近的上下文摘要（用于路由Agent）
        
        Returns:
            格式化的最近对话字符串
        """
        history = self.get_history(session_id, limit=limit)
        if not history:
            return "暂无对话历史"
        
        lines = []
        for msg in history:
            agent = msg.get('agent_name', 'unknown')
            role = msg['role']
            content = msg['content'][:200]  # 截断过长内容
            lines.append(f"[{agent}] {role}: {content}")
        
        return "\n".join(lines)
    
    def clear_session(self, session_id: str):
        """清除指定会话的所有记录"""
        self.db.execute(
            "DELETE FROM conversations WHERE session_id = ?",
            (session_id,)
        )
    
    def get_session_stats(self, session_id: str) -> Dict:
        """获取会话统计信息"""
        result = self.db.fetch_one(
            """SELECT COUNT(*) as total, 
                      COUNT(DISTINCT agent_name) as agents_used
               FROM conversations WHERE session_id = ?""",
            (session_id,)
        )
        return result or {"total": 0, "agents_used": 0}
