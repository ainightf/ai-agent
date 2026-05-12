import sqlite3
import os
from datetime import datetime
from config.settings import SQLITE_DB_PATH


class SQLiteDatabase:
    """SQLite数据库管理器"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or SQLITE_DB_PATH
        self._ensure_dir()
        self._init_tables()
    
    def _ensure_dir(self):
        """确保数据库目录存在"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
    
    def _get_connection(self):
        """获取数据库连接"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def _init_tables(self):
        """初始化数据表"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            # 对话历史表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    agent_name TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            # 共享上下文表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS shared_context (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    agent_name TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            # 创建索引
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_conversations_session 
                ON conversations(session_id, agent_name)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_shared_context_session 
                ON shared_context(session_id, key)
            ''')
            conn.commit()
        finally:
            conn.close()
    
    def execute(self, sql: str, params: tuple = ()):
        """执行SQL语句"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            conn.commit()
            return cursor
        finally:
            conn.close()
    
    def fetch_all(self, sql: str, params: tuple = ()):
        """查询所有结果"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()
    
    def fetch_one(self, sql: str, params: tuple = ()):
        """查询单条结果"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            row = cursor.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()
