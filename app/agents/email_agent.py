"""邮件Agent - AI邮件助手，智能阅读邮件摘要，管理收件箱，支持通过自然语言发送邮件"""
import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

from app.llm.llm import build_chat_model
from app.tools.email_tool import EmailSenderTool, fetch_emails_structured
from app.memory.memory import PersistentMemory
from app.memory.shared_memory import SharedMemory
from config.settings import settings


# 中文序号 -> 数字映射，用于解析“第三封”
_CN_NUM_MAP = {
    "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
}

# 邮件搜索关键词提取正则
# 匹配模式：“看hsbc的邮件”、“搜索nike相关邮件”、“看hsbc的电子结单”
_GENERIC_SUFFIXES = {"邮件", "mail", "email", "mails", "emails"}  # 泛指，不携带搜索信息
_EMAIL_LIKE_SUFFIXES = r"(?:邮件|电子?结单|账单|通知|信件|来信|报告|mail|email|statement|bill|newsletter)"

_SEARCH_PATTERNS = [
    # "看/查/找/搜索 + [下/一下] + 关键词 + 的 + 邮件类后缀"
    # 注意：第 2 个 group 捕获后缀，用于判断是否携带有效信息
    re.compile(rf"(?:帮我看|帮我查|帮我找|我要看|我想看|我要查|我要|我想|看|查|找|搜索|搜)(?:一下|下)?\s*(.+?)\s*(?:的|相关的?)\s*(?:最近的?)?\s*({_EMAIL_LIKE_SUFFIXES})", re.IGNORECASE),
    # "关键词 + 的最近/最新 + 邮件类后缀"
    re.compile(rf"(.+?)\s*(?:的)\s*(?:最近|最新|recent)?\s*({_EMAIL_LIKE_SUFFIXES})", re.IGNORECASE),
    # "search xxx emails / emails from xxx"
    re.compile(r"(?:search|find|look for)\s+(.+?)\s+((?:emails?|mails?))", re.IGNORECASE),
    re.compile(r"((?:emails?|mails?))\s+(?:from|about|regarding)\s+(.+)", re.IGNORECASE),
]

# 通用 fallback：剥离常见动作前缀，用于正则全部未命中时的最后尝试
_ACTION_PREFIX = re.compile(r"^(?:\u5e2e\u6211|\u6211\u8981|\u6211\u60f3|\u8bf7)?(?:\u770b|\u67e5\u770b|\u67e5|\u627e|\u641c\u7d22|\u641c|\u8bfb)?(?:\u4e00\u4e0b|\u4e0b|\u4e86)?\s*")


def _parse_email_index(query: str):
    """从用户输入中解析邮件序号，命中返回 int，否则 None"""
    # 1. 阿拉伯数字：#3、第3封、序号3、展号 3
    m = re.search(r"(?:#|第|序号|no\.?|\bnum\b|号)\s*(\d+)", query, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    # 2. 纯数字开头："3 封"、"3."、"第 3"
    m = re.search(r"(\d+)\s*封", query)
    if m:
        return int(m.group(1))
    # 3. 中文：第三封、三封
    m = re.search(r"第?([一二两三四五六七八九十]+)封", query)
    if m:
        ch = m.group(1)
        if ch in _CN_NUM_MAP:
            return _CN_NUM_MAP[ch]
        # 十一、十二等
        if ch.startswith("十") and len(ch) == 2:
            return 10 + _CN_NUM_MAP.get(ch[1], 0)
    return None


# 搜索意图不应误触发的停用词（避免 "帮我看下邮件" 被匹配成 search）
_SEARCH_STOP_WORDS = {
    "下", "一下", "收件箱", "最新", "最近", "我的", "邮件列表", "",
    "邮件", "最新邮件", "最近邮件", "看最新邮件", "看邮件",
    "查看邮件", "查邮件", "看下邮件", "看最近邮件",
    "未读邮件", "查看未读邮件",
}

# 常见中文语气词 / 助词前缀，清洗提取结果时剥离
_KEYWORD_NOISE_PREFIX = re.compile(r"^(?:下|一下|了|看|查|找|帮我)")


def _extract_search_keyword(query: str):
    """从用户输入中提取邮件搜索关键词，没命中返回 None

    智能处理：
    - "看hsbc的邮件" → "hsbc"（后缀是泛指"邮件"，不携带）
    - "看hsbc的电子结单" → "hsbc的电子结单"（后缀是具体内容，拼回搜索词）
    """
    # 第一级：正则匹配
    for pattern in _SEARCH_PATTERNS:
        m = pattern.search(query)
        if m:
            keyword = m.group(1).strip()
            # 清洗：剥离开头的语气词/助词
            keyword = _KEYWORD_NOISE_PREFIX.sub("", keyword).strip()

            # 判断后缀是否携带有效搜索信息（非泛指）
            suffix = m.group(2).strip() if m.lastindex >= 2 else ""
            if suffix and suffix.lower() not in _GENERIC_SUFFIXES:
                # 后缀是具体内容（如"电子结单"），拼回搜索词
                keyword = f"{keyword}的{suffix}"

            # 过滤停用词
            if keyword and keyword not in _SEARCH_STOP_WORDS and len(keyword) >= 2:
                return keyword

    # 第二级：通用 fallback——剥离动作前缀，留下核心内容
    cleaned = _ACTION_PREFIX.sub("", query).strip()
    if cleaned and cleaned != query.strip() and len(cleaned) >= 2:
        if cleaned not in _SEARCH_STOP_WORDS:
            return cleaned
    return None


class EmailAgent:
    """AI邮件助手 - 智能阅读邮件摘要，管理收件箱，支持通过自然语言发送邮件"""

    # === 最小硬规则：只管“死明确”的 case ===
    _SEND_KEYWORDS = (
        "发邮件", "写邮件", "发一封", "回复邮件", "发给", "发送到", "send email", "mail to",
    )

    def __init__(self):
        self.llm_precise = build_chat_model(temperature=0)       # 意图判断 & 信息提取
        self.llm_summary = build_chat_model(temperature=0.3)     # 邮件摘要生成
        self.email_sender = EmailSenderTool()
        self.memory = PersistentMemory()
        self.shared_memory = SharedMemory()
        # 按 session 缓存最近一次邮件列表结果，可用序号取单封
        self._email_cache: dict = {}
        # LLM 意图缓存（query -> {intent, keyword}）
        self._intent_cache: dict = {}
        self._intent_cache_max = 128

    def run(self, query: str, session_id: str) -> str:
        """主入口：判断用户意图并分发到对应方法

        Args:
            query: 用户输入
            session_id: 会话ID

        Returns:
            处理结果字符串
        """
        try:
            # 保存用户消息到持久化记忆
            self.memory.add_message(session_id, "email", "user", query)

            # 配置检查（如 settings 提供了检测方法）
            if hasattr(settings, "is_email_configured") and not settings.is_email_configured():
                missing = "、".join(settings.email_missing_fields()) if hasattr(settings, "email_missing_fields") else ""
                result = (
                    "⚠️ 邮件服务尚未配置完成，无法为您执行邮件操作。\n"
                    + (f"缺少字段：{missing}\n" if missing else "")
                    + "请在 .env 中补齐 EMAIL_ADDRESS / EMAIL_PASSWORD 后重启。"
                )
                self.memory.add_message(session_id, "email", "assistant", result)
                return result

            # 意图识别：硬规则只管“序号→detail”和“发邮件→send”，其余交给 LLM
            intent, keyword = self._smart_classify(query)

            if intent == "list":
                result = self.list_inbox(session_id)
            elif intent == "search":
                # LLM 返回的 keyword 优先，正则提取作 fallback
                search_kw = keyword or _extract_search_keyword(query) or query.strip()
                result = self.search_inbox(search_kw, session_id)
            elif intent == "detail":
                result = self.summarize_one(query, session_id)
            elif intent == "send":
                result = self.send_email(query, session_id)
            else:
                result = "抱歉，我无法理解您的邮件操作意图。您可以说「查看收件箱」、「看第3封」、「看hsbc的邮件」或「发邮件给XX」。"

            # 保存助手回复到持久化记忆
            self.memory.add_message(session_id, "email", "assistant", result)

            return result

        except Exception as e:
            error_msg = f"邮件处理出错：{str(e)}"
            self.memory.add_message(session_id, "email", "assistant", error_msg)
            return error_msg

    # ============ 智能意图识别 ============

    def _smart_classify(self, query: str) -> tuple:
        """意图识别主入口：返回 (intent, keyword)

        策略：
        1. 硬规则：序号→detail，发邮件→send（0 次 LLM）
        2. 其余全部交给 LLM（一次调用同时返回意图 + 搜索关键词）
        3. 缓存结果避免重复调用
        4. LLM 失败时 fallback 到 list
        """
        q = query.lower().strip()

        # --- 硬规则：只管“死明确”的 ---
        if _parse_email_index(query) is not None:
            if any(k in q for k in self._SEND_KEYWORDS):
                return "send", None
            return "detail", None
        if any(k in q for k in self._SEND_KEYWORDS):
            return "send", None

        # --- 缓存 ---
        cache_key = q
        if cache_key in self._intent_cache:
            cached = self._intent_cache[cache_key]
            return cached["intent"], cached.get("keyword")

        # --- LLM 识别 ---
        intent, keyword = self._llm_classify(query)

        # 写入缓存
        if len(self._intent_cache) >= self._intent_cache_max:
            try:
                self._intent_cache.pop(next(iter(self._intent_cache)))
            except StopIteration:
                pass
        self._intent_cache[cache_key] = {"intent": intent, "keyword": keyword}

        return intent, keyword

    def _llm_classify(self, query: str) -> tuple:
        """用 LLM 一次性识别意图 + 提取搜索关键词，返回 (intent, keyword)"""
        system_prompt = """\u4f60\u662f\u4e00\u4e2a\u90ae\u4ef6\u610f\u56fe\u5206\u7c7b\u5668\u3002\u6839\u636e\u7528\u6237\u8f93\u5165\uff0c\u8fd4\u56de JSON\u3002

## \u610f\u56fe\u5206\u7c7b\u89c4\u5219\uff1a
- **list**: \u67e5\u770b\u6536\u4ef6\u7bb1\u3001\u67e5\u770b\u6700\u65b0/\u672a\u8bfb\u90ae\u4ef6\u3001\u90ae\u4ef6\u5217\u8868\uff08\u6ca1\u6709\u6307\u5b9a\u5177\u4f53\u53d1\u4ef6\u4eba\u6216\u4e3b\u9898\u7684\u6cdb\u6cdb\u67e5\u770b\uff09
- **search**: \u641c\u7d22\u7279\u5b9a\u53d1\u4ef6\u4eba\u3001\u7279\u5b9a\u4e3b\u9898\u3001\u7279\u5b9a\u5185\u5bb9\u7684\u90ae\u4ef6\uff08\u7528\u6237\u660e\u786e\u6307\u5b9a\u4e86\u8981\u627e\u8c01/\u4ec0\u4e48\uff09
- **detail**: \u67e5\u770b\u67d0\u5c01\u5177\u4f53\u90ae\u4ef6\u7684\u5185\u5bb9/\u6458\u8981\uff08\u901a\u5e38\u5305\u542b\u5e8f\u53f7\uff09
- **send**: \u53d1\u9001\u6216\u56de\u590d\u90ae\u4ef6

## \u5173\u952e\u533a\u5206\uff08list vs search\uff09\uff1a
- \u201c\u67e5\u770b\u672a\u8bfb\u90ae\u4ef6\u201d \u2192 list\uff08\u6cdb\u6cdb\u67e5\u770b\uff0c\u6ca1\u6307\u5b9a\u641c\u7d22\u5bf9\u8c61\uff09
- \u201c\u5e2e\u6211\u770b\u4e0b\u6536\u4ef6\u7bb1\u201d \u2192 list
- \u201c\u770bhsbc\u7684\u90ae\u4ef6\u201d \u2192 search\uff0ckeyword=\u201chsbc\u201d\uff08\u6307\u5b9a\u4e86\u53d1\u4ef6\u4eba\uff09
- \u201c\u6211\u8981\u770bhsbc\u7684\u7535\u5b50\u7ed3\u5355\u201d \u2192 search\uff0ckeyword=\u201chsbc\u7684\u7535\u5b50\u7ed3\u5355\u201d
- \u201c\u641c\u7d22\u6765\u81ea\u82f9\u679c\u7684\u901a\u77e5\u201d \u2192 search\uff0ckeyword=\u201c\u82f9\u679c\u7684\u901a\u77e5\u201d

## \u8f93\u51fa\u683c\u5f0f\uff08\u4e25\u683c JSON\uff0c\u4e0d\u8981\u5176\u4ed6\u5185\u5bb9\uff09\uff1a
{"intent": "list|search|detail|send", "keyword": "\u641c\u7d22\u5173\u952e\u8bcd\u6216null"}"""

        user_prompt = f"\u7528\u6237\u8f93\u5165\uff1a{query}\n\u8bf7\u8fd4\u56de JSON\uff1a"

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]

        try:
            response = self.llm_precise.invoke(messages)
            text = response.content.strip()
            # 解析 JSON
            # 处理 LLM 可能返回 markdown 包裹的 json
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            result = json.loads(text)
            intent = result.get("intent", "list").lower()
            keyword = result.get("keyword")
            # 验证 intent 合法性
            if intent not in ("list", "search", "detail", "send"):
                intent = "list"
            # keyword 为 "null" 字符串时转 None
            if keyword in (None, "null", "", "None"):
                keyword = None
            return intent, keyword
        except Exception as e:
            print(f"[EmailAgent] LLM 意图识别失败，fallback 到 list：{e}")
            return "list", None

    # ============ 业务方法 ============

    def list_inbox(self, session_id: str, num: int = 10) -> str:
        """列出收件箱（0 次 LLM 调用），每行带序号，用户后续可说「看第N封」"""
        emails, err = fetch_emails_structured(num_emails=num)
        if err:
            return f"读取邮件失败：{err}"
        if not emails:
            return "收件箱为空，没有邮件。"

        # 方案C：缓存结构化数据用于后续序号定位
        self._email_cache[session_id] = emails

        lines = [f"📬 最新 {len(emails)} 封邮件（输入「看第N封」查看具体摘要）："]
        for e in emails:
            subject = (e["subject"] or "无主题").strip()
            sender = (e["sender"] or "未知发件人").strip()
            if len(sender) > 40:
                sender = sender[:40] + "..."
            if len(subject) > 50:
                subject = subject[:50] + "..."
            lines.append(
                f"\n[{e['index']}] {subject}\n    发件人：{sender}\n    日期：{e['date']}"
            )

        self.shared_memory.set_context(
            session_id,
            "last_email_list",
            {"count": len(emails), "latest_subject": emails[0]["subject"]},
            "email",
        )
        return "\n".join(lines)

    def search_inbox(self, keyword: str, session_id: str, num: int = 10) -> str:
        """按关键词搜索邮件（0 次 LLM 调用），服务器端按发件人/主题过滤"""
        emails, err = fetch_emails_structured(num_emails=num, search_query=keyword)
        if err:
            return f"搜索邮件失败：{err}"
        if not emails:
            return f"未找到与「{keyword}」相关的邮件。"

        # 缓存搜索结果，用户后续可以说“看第3封”
        self._email_cache[session_id] = emails

        lines = [f"🔍 找到 {len(emails)} 封与「{keyword}」相关的邮件（输入「看第N封」查看具体内容）："]
        for e in emails:
            subject = (e["subject"] or "无主题").strip()
            sender = (e["sender"] or "未知发件人").strip()
            if len(sender) > 40:
                sender = sender[:40] + "..."
            if len(subject) > 50:
                subject = subject[:50] + "..."
            lines.append(
                f"\n[{e['index']}] {subject}\n    发件人：{sender}\n    日期：{e['date']}"
            )

        self.shared_memory.set_context(
            session_id,
            "last_email_search",
            {"keyword": keyword, "count": len(emails)},
            "email",
        )
        return "\n".join(lines)

    def summarize_one(self, query: str, session_id: str) -> str:
        """根据用户指定的序号，对单封邮件生成摘要（1 次 LLM 调用）"""
        idx = _parse_email_index(query)
        if idx is None:
            return "请指定邮件序号，例如「看第3封」或「#3 摘要一下」。"

        emails = self._email_cache.get(session_id)
        if not emails:
            # 缓存不存在（比如用户直接说「看第3封」），先拉一次
            emails, err = fetch_emails_structured(num_emails=10)
            if err:
                return f"读取邮件失败：{err}"
            if not emails:
                return f"收件箱为空，无法定位第 {idx} 封邮件。"
            self._email_cache[session_id] = emails

        if idx < 1 or idx > len(emails):
            return f"序号 {idx} 超出范围，当前共 {len(emails)} 封邮件。"

        target = emails[idx - 1]
        body = target["body"] or "（无正文）"
        if len(body) > 4000:
            body = body[:4000] + "...（正文过长已截断）"

        single_prompt = (
            "请对以下单封邮件生成结构化摘要，要求覆盖：\n"
            "1. 一句话核心内容\n"
            "2. 关键信息（3-5 条，包含数字/日期/链接等）\n"
            "3. 是否需要回复以及建议回复要点\n\n"
            f"发件人：{target['sender']}\n"
            f"主题：{target['subject']}\n"
            f"日期：{target['date']}\n"
            f"正文：\n{body}"
        )
        messages = [
            SystemMessage(content="你是专业的邮件摘要助手，用中文输出，结构清晰。"),
            HumanMessage(content=single_prompt),
        ]
        try:
            response = self.llm_summary.invoke(messages)
            summary = response.content.strip()
        except Exception as e:
            return f"生成摘要失败：{str(e)}"

        self.shared_memory.set_context(
            session_id,
            "last_email_detail",
            {"index": idx, "subject": target["subject"], "sender": target["sender"]},
            "email",
        )
        header = (
            f"📧 第 {idx} 封邮件摘要\n"
            f"主题：{target['subject']}\n"
            f"发件人：{target['sender']}\n"
            f"日期：{target['date']}\n\n"
        )
        return header + summary

    def send_email(self, query: str, session_id: str) -> str:
        """从自然语言中提取信息并发送邮件

        Args:
            query: 用户自然语言发送请求
            session_id: 会话ID

        Returns:
            发送结果字符串
        """
        try:
            # 0. 用正则直接从原文提取邮箱（不依赖 LLM，避免地址被篡改）
            email_regex = re.findall(r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}', query)
            raw_email_addr = email_regex[0] if email_regex else None

            # 1. 使用 LLM 提取主题和正文
            extract_prompt = """从用户的自然语言中提取邮件发送信息。

用户输入：{query}

请严格以JSON格式输出，不要包含其他内容：
{{"to": "收件人邮箱地址", "subject": "邮件主题", "body": "邮件正文"}}

规则：
1. to 字段必须原样保留用户给出的邮箱地址，不要修改任何字符
2. 如果用户没有明确指定主题，根据正文内容生成一个简短主题
3. 如果用户没有明确指定正文，根据上下文生成合适的正文
4. 如果无法提取收件人邮箱，to字段填空字符串
5. 正文应该保持用户原文，不要擅自修改"""

            messages = [
                SystemMessage(content="你是一个邮件信息提取助手，负责从自然语言中精确提取邮件要素。只输出JSON，不要输出其他内容。邮箱地址必须原样保留，不能修改。"),
                HumanMessage(content=extract_prompt.format(query=query))
            ]

            response = self.llm_precise.invoke(messages)
            extracted_text = response.content.strip()

            # 清理可能的markdown代码块标记
            if extracted_text.startswith("```"):
                extracted_text = extracted_text.split("\n", 1)[-1]
                if extracted_text.endswith("```"):
                    extracted_text = extracted_text[:-3]
                extracted_text = extracted_text.strip()

            # 解析提取结果
            try:
                email_data = json.loads(extracted_text)
            except json.JSONDecodeError:
                return "抱歉，无法从您的描述中提取邮件信息。请提供收件人邮箱、主题和正文内容。"

            # 关键：用正则提取的邮箱覆盖 LLM 输出（防止 LLM 篡改地址）
            if raw_email_addr:
                email_data["to"] = raw_email_addr

            # 验证必填字段
            if not email_data.get("to"):
                return "抱歉，无法识别收件人邮箱地址。请明确指定收件人邮箱。"

            # 2. 调用 EmailSenderTool 发送
            send_input = json.dumps(email_data, ensure_ascii=False)
            result = self.email_sender._run(input=send_input)

            # 3. 保存操作记录到 SharedMemory
            self.shared_memory.set_context(
                session_id,
                "last_email_sent",
                {
                    "to": email_data.get("to"),
                    "subject": email_data.get("subject"),
                    "status": "成功" if "成功" in result else "失败",
                    "result": result[:200]
                },
                "email"
            )

            return result

        except Exception as e:
            return f"发送邮件出错：{str(e)}"

    def _classify_intent(self, query: str) -> str:
        """兼容旧调用（router 等可能调用此方法）"""
        intent, _ = self._smart_classify(query)
        return intent
