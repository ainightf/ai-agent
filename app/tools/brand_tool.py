"""天眼查品牌商标审核工具"""
import httpx
from typing import Optional
from langchain.tools import BaseTool
from pydantic import Field

from config.settings import settings


class BrandVerificationTool(BaseTool):
    """品牌商标查询工具 - 基于天眼查API
    
    API: GET http://open.api.tianyancha.com/services/open/ipr/tm/2.0
    """
    
    name: str = "brand_verification"
    description: str = (
        "查询品牌商标注册信息。输入品牌名称，返回商标注册状态、申请人、商标类别等信息。"
        "适用于：查询某品牌是否已注册商标、商标审核状态、商标归属等。"
    )
    
    base_url: str = Field(default_factory=lambda: settings.TIANYANCHA_BASE_URL)
    token: str = Field(default_factory=lambda: settings.TIANYANCHA_TOKEN)
    
    def _run(self, brand_name: str) -> str:
        """执行品牌商标查询
        
        Args:
            brand_name: 品牌/商标名称
            
        Returns:
            格式化的查询结果字符串
        """
        try:
            url = f"{self.base_url}/services/open/ipr/tm/2.0"
            headers = {
                "Authorization": self.token,
                "Content-Type": "application/json"
            }
            params = {
                "keyword": brand_name,
                "pageSize": 5,
                "pageNum": 1
            }
            
            with httpx.Client(timeout=30.0) as client:
                response = client.get(url, headers=headers, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    return self._format_response(data, brand_name)
                elif response.status_code == 401:
                    return f"认证失败：Token无效或已过期。请检查天眼查API Token配置。"
                elif response.status_code == 403:
                    return f"权限不足：当前Token无权访问商标查询接口。"
                else:
                    return f"查询失败：HTTP {response.status_code} - {response.text[:200]}"
                    
        except httpx.TimeoutException:
            return f"查询超时：天眼查API响应超时，请稍后重试。"
        except httpx.ConnectError:
            return f"连接失败：无法连接天眼查API服务器 ({self.base_url})。请检查网络连接。"
        except Exception as e:
            return f"查询异常：{str(e)}"
    
    async def _arun(self, brand_name: str) -> str:
        """异步执行品牌商标查询"""
        try:
            url = f"{self.base_url}/services/open/ipr/tm/2.0"
            headers = {
                "Authorization": self.token,
                "Content-Type": "application/json"
            }
            params = {
                "keyword": brand_name,
                "pageSize": 5,
                "pageNum": 1
            }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, headers=headers, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    return self._format_response(data, brand_name)
                elif response.status_code == 401:
                    return f"认证失败：Token无效或已过期。"
                else:
                    return f"查询失败：HTTP {response.status_code}"
                    
        except Exception as e:
            return f"查询异常：{str(e)}"
    
    def _format_response(self, data: dict, brand_name: str) -> str:
        """格式化API响应
        
        天眼查商标API典型响应结构：
        {
            "error_code": 0,
            "reason": "ok",
            "result": {
                "total": 10,
                "items": [
                    {
                        "appDate": "2020-01-01",
                        "tmName": "品牌名",
                        "tmClass": "第35类",
                        "applicantCn": "申请人",
                        "status": "已注册",
                        "regNo": "注册号"
                    }
                ]
            }
        }
        """
        if not data:
            return f"未查询到\u201c{brand_name}\u201d相关商标信息。"
        
        error_code = data.get("error_code", data.get("errorCode", -1))
        
        if error_code != 0:
            reason = data.get("reason", data.get("message", "未知错误"))
            return f"API返回错误：{reason}"
        
        result = data.get("result", {})
        items = result.get("items", result.get("resultList", []))
        total = result.get("total", result.get("totalCount", 0))
        
        if not items:
            return f"未查询到\u201c{brand_name}\u201d相关商标注册信息。"
        
        # 格式化输出
        output_lines = [
            f"=== 品牌商标查询结果 ===",
            f"查询关键词：{brand_name}",
            f"共找到 {total} 条相关记录，显示前 {len(items)} 条：",
            ""
        ]
        
        for i, item in enumerate(items, 1):
            tm_name = item.get("tmName", item.get("name", "未知"))
            status = item.get("status", item.get("tmStatus", "未知"))
            applicant = item.get("applicantCn", item.get("applicant", "未知"))
            tm_class = item.get("tmClass", item.get("intCls", "未知"))
            app_date = item.get("appDate", item.get("applicationDate", "未知"))
            reg_no = item.get("regNo", item.get("regNum", "未知"))
            
            output_lines.extend([
                f"--- 记录 {i} ---",
                f"  商标名称：{tm_name}",
                f"  注册号：{reg_no}",
                f"  商标状态：{status}",
                f"  申请人：{applicant}",
                f"  商标类别：{tm_class}",
                f"  申请日期：{app_date}",
                ""
            ])
        
        return "\n".join(output_lines)
