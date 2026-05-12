"""品牌商标审核Agent - 使用天眼查API查询商标信息"""
from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import settings
from app.llm.llm import build_chat_model
from app.tools.brand_tool import BrandVerificationTool
from app.memory.memory import PersistentMemory
from app.memory.shared_memory import SharedMemory


class BrandAgent:
    """品牌商标审核Agent"""
    
    def __init__(self):
        self.llm = build_chat_model(temperature=0)
        self.brand_tool = BrandVerificationTool()
        self.memory = PersistentMemory()
        self.shared_memory = SharedMemory()
    
    def run(self, query: str, session_id: str) -> str:
        """执行品牌商标查询
        
        Args:
            query: 用户查询（需要从中提取品牌名称）
            session_id: 会话ID
            
        Returns:
            查询结果和分析
        """
        try:
            # 1. 提取品牌名称
            brand_name = self._extract_brand_name(query)
            
            # 2. 调用天眼查API
            tool_result = self.brand_tool._run(brand_name)
            
            # 3. LLM整合分析
            system_prompt = """你是一个品牌商标审核专家。根据天眼查API返回的商标查询结果，为用户提供专业的分析和建议。

规则：
1. 清晰说明商标的注册状态
2. 如果有多个结果，帮助用户区分
3. 给出实用的商标建议（如：是否可以使用、风险提示等）
4. 语言简洁专业"""

            user_prompt = f"""用户查询：{query}
提取的品牌名称：{brand_name}

天眼查API查询结果：
{tool_result}

请为用户提供专业分析："""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm.invoke(messages)
            answer = response.content
            
            # 4. 保存到共享上下文
            self.shared_memory.set_context(
                session_id,
                "last_brand_query",
                {"brand": brand_name, "result_preview": answer[:100]},
                "brand"
            )
            
            return answer
            
        except Exception as e:
            return f"品牌商标查询出错：{str(e)}"
    
    def _extract_brand_name(self, query: str) -> str:
        """从用户输入中提取品牌名称"""
        system_prompt = "从用户的输入中提取品牌/商标名称。只返回品牌名称，不要其他内容。如果无法提取，返回用户输入中最可能是品牌的词。"
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query)
        ]
        
        try:
            response = self.llm.invoke(messages)
            return response.content.strip()
        except:
            # 降级：直接使用输入中的关键词
            # 移除常见询问词
            for word in ["查询", "查一下", "商标", "品牌", "的", "注册", "了吗", "帮我", "查下"]:
                query = query.replace(word, "")
            return query.strip() or query
