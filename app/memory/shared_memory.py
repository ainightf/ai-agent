"""跨Agent共享记忆模块"""
from typing import Dict, Optional, List
import json

from app.database.sqlite_db import SQLiteDatabase


class SharedMemory:
    """跨Agent共享上下文管理器
    
    设计理念：
    - 任何Agent可以写入关键信息到共享池
    - 其他Agent可以读取这些信息
    - 按Session隔离
    - 支持key-value存储，value可以是复杂结构（JSON序列化）
    """
    
    def __init__(self, db: SQLiteDatabase = None):
        self.db = db or SQLiteDatabase()
    
    def set_context(self, session_id: str, key: str, value, agent_name: str):
        """设置共享上下文
        
        Args:
            session_id: 会话ID
            key: 上下文键名
            value: 上下文值（支持字符串、字典、列表等）
            agent_name: 写入Agent的名称
        """
        # 序列化value
        if isinstance(value, (dict, list)):
            value_str = json.dumps(value, ensure_ascii=False)
        else:
            value_str = str(value)
        
        # 检查是否已存在相同key，存在则更新
        existing = self.db.fetch_one(
            "SELECT id FROM shared_context WHERE session_id = ? AND key = ?",
            (session_id, key)
        )
        
        if existing:
            self.db.execute(
                "UPDATE shared_context SET value = ?, agent_name = ?, timestamp = CURRENT_TIMESTAMP WHERE session_id = ? AND key = ?",
                (value_str, agent_name, session_id, key)
            )
        else:
            self.db.execute(
                "INSERT INTO shared_context (session_id, key, value, agent_name) VALUES (?, ?, ?, ?)",
                (session_id, key, value_str, agent_name)
            )
    
    def get_context(self, session_id: str, key: Optional[str] = None) -> Optional[Dict]:
        """获取共享上下文
        
        Args:
            session_id: 会话ID
            key: 上下文键名（为None时返回所有上下文）
            
        Returns:
            单个key时返回 {"key": ..., "value": ..., "agent_name": ..., "timestamp": ...}
            所有key时返回 {key: {"value": ..., "agent_name": ..., "timestamp": ...}, ...}
        """
        if key:
            result = self.db.fetch_one(
                "SELECT key, value, agent_name, timestamp FROM shared_context WHERE session_id = ? AND key = ?",
                (session_id, key)
            )
            if result:
                result['value'] = self._deserialize_value(result['value'])
            return result
        else:
            results = self.db.fetch_all(
                "SELECT key, value, agent_name, timestamp FROM shared_context WHERE session_id = ? ORDER BY timestamp DESC",
                (session_id,)
            )
            context = {}
            for r in results:
                context[r['key']] = {
                    "value": self._deserialize_value(r['value']),
                    "agent_name": r['agent_name'],
                    "timestamp": r['timestamp']
                }
            return context
    
    def get_agent_summary(self, session_id: str) -> str:
        """获取当前会话的共享上下文摘要
        
        Returns:
            格式化的上下文摘要字符串，适合作为Agent的背景知识
        """
        all_context = self.get_context(session_id)
        
        if not all_context:
            return "当前会话暂无共享上下文。"
        
        lines = ["=== 共享上下文 ==="]
        for key, info in all_context.items():
            value = info['value']
            if isinstance(value, (dict, list)):
                value_str = json.dumps(value, ensure_ascii=False, indent=2)[:300]
            else:
                value_str = str(value)[:300]
            lines.append(f"- {key} (by {info['agent_name']}): {value_str}")
        
        return "\n".join(lines)
    
    def delete_context(self, session_id: str, key: str):
        """删除指定上下文"""
        self.db.execute(
            "DELETE FROM shared_context WHERE session_id = ? AND key = ?",
            (session_id, key)
        )
    
    def clear_session(self, session_id: str):
        """清除会话的所有共享上下文"""
        self.db.execute(
            "DELETE FROM shared_context WHERE session_id = ?",
            (session_id,)
        )
    
    def _deserialize_value(self, value_str: str):
        """反序列化value"""
        try:
            return json.loads(value_str)
        except (json.JSONDecodeError, TypeError):
            return value_str
