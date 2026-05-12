"""
邮件工具 - 邮件读取与发送能力
"""
import os
import imaplib
import smtplib
import email
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import decode_header
from typing import Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

import sys
sys.path.append("../..")
from config.settings import settings


def _get_email_config():
    """获取邮件配置，优先从 settings 读取，否则从环境变量获取"""
    return {
        "email_address": getattr(settings, "EMAIL_ADDRESS", "") or os.getenv("EMAIL_ADDRESS", ""),
        "email_password": getattr(settings, "EMAIL_PASSWORD", "") or os.getenv("EMAIL_PASSWORD", ""),
        "imap_server": getattr(settings, "IMAP_SERVER", "") or os.getenv("IMAP_SERVER", ""),
        "imap_port": int(getattr(settings, "IMAP_PORT", 993) or os.getenv("IMAP_PORT", "993")),
        "smtp_server": getattr(settings, "SMTP_SERVER", "") or os.getenv("SMTP_SERVER", ""),
        "smtp_port": int(getattr(settings, "SMTP_PORT", 465) or os.getenv("SMTP_PORT", "465")),
    }


def _decode_mime_header(header_value: str) -> str:
    """解码 MIME 邮件头"""
    if not header_value:
        return ""
    decoded_parts = decode_header(header_value)
    result = []
    for part, charset in decoded_parts:
        if isinstance(part, bytes):
            charset = charset or "utf-8"
            try:
                result.append(part.decode(charset))
            except (UnicodeDecodeError, LookupError):
                try:
                    result.append(part.decode("gbk"))
                except (UnicodeDecodeError, LookupError):
                    result.append(part.decode("utf-8", errors="replace"))
        else:
            result.append(part)
    return "".join(result)


def _extract_body(msg: email.message.Message) -> str:
    """从邮件消息中提取正文内容"""
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition", ""))
            if content_type == "text/plain" and "attachment" not in content_disposition:
                charset = part.get_content_charset() or "utf-8"
                try:
                    payload = part.get_payload(decode=True)
                    if payload:
                        body = payload.decode(charset)
                except (UnicodeDecodeError, LookupError):
                    try:
                        body = payload.decode("gbk")
                    except (UnicodeDecodeError, LookupError):
                        body = payload.decode("utf-8", errors="replace")
                break
    else:
        charset = msg.get_content_charset() or "utf-8"
        try:
            payload = msg.get_payload(decode=True)
            if payload:
                body = payload.decode(charset)
        except (UnicodeDecodeError, LookupError):
            try:
                body = payload.decode("gbk")
            except (UnicodeDecodeError, LookupError):
                body = payload.decode("utf-8", errors="replace")
    return body


class EmailReaderInput(BaseModel):
    """邮件读取工具输入"""
    num_emails: int = Field(default=10, description="读取最新邮件数量，默认10封")


def fetch_emails_structured(num_emails: int = 10, search_query: str = None):
    """获取收件箱结构化邮件列表

    Args:
        num_emails: 返回最多邮件数量
        search_query: 搜索关键词（按发件人/主题过滤），None 表示不过滤

    Returns:
        (list[dict], str) - 列表每一项包含 {index, sender, subject, date, body}；
        第二返回值为错误消息，仅在失败时非空。
    """
    config = _get_email_config()

    if not config["email_address"] or not config["email_password"]:
        return [], "错误：邮件账号或密码未配置，请设置 EMAIL_ADDRESS 和 EMAIL_PASSWORD。"
    if not config["imap_server"]:
        return [], "错误：IMAP 服务器未配置，请设置 IMAP_SERVER。"

    mail = None
    try:
        mail = imaplib.IMAP4_SSL(config["imap_server"], config["imap_port"], timeout=30)
    except Exception as e:
        return [], f"IMAP 连接异常：{str(e)}"

    try:
        mail.login(config["email_address"], config["email_password"])
    except Exception as e:
        try:
            mail.logout()
        except Exception:
            pass
        return [], f"IMAP 认证失败：{str(e)}（请确认使用邮箱授权码）"

    # QQ/163 邮箱要求的 ID 命令
    try:
        imaplib.Commands["ID"] = ("AUTH",)
        id_args = (
            '("name" "langchain-email-agent" '
            '"version" "1.0" '
            '"vendor" "myclient" '
            '"contact" "admin@example.com")'
        )
        mail._simple_command("ID", id_args)
    except Exception:
        pass

    results = []
    try:
        mail.select("INBOX")

        # 构建 IMAP 搜索条件
        if search_query:
            mail_ids = _imap_search_with_fallback(mail, search_query, num_emails)
        else:
            status, messages = mail.search(None, "ALL")
            if status != "OK":
                return [], "无法搜索收件箱邮件。"
            mail_ids = messages[0].split()
            if mail_ids:
                mail_ids = mail_ids[-num_emails:]  # 取最新 N 封
                mail_ids.reverse()

        if not mail_ids:
            if search_query:
                return [], ""
            return [], ""

        for i, mail_id in enumerate(mail_ids, 1):
            status, msg_data = mail.fetch(mail_id, "(RFC822)")
            if status != "OK":
                continue
            raw_email = msg_data[0][1]
            msg = email.message_from_bytes(raw_email)
            sender = _decode_mime_header(msg.get("From", "未知发件人"))
            subject = _decode_mime_header(msg.get("Subject", "无主题"))
            date = msg.get("Date", "未知日期")
            body = _extract_body(msg) or ""
            results.append({
                "index": i,
                "sender": sender,
                "subject": subject,
                "date": date,
                "body": body.strip(),
            })
    except Exception as e:
        return [], f"读取邮件异常：{str(e)}"
    finally:
        try:
            mail.logout()
        except Exception:
            pass

    return results, ""


def _imap_search_with_fallback(mail, query: str, num_emails: int):
    """用 IMAP SEARCH 按发件人/主题搜索，支持复合关键词（如 "hsbc的电子结单"）

    复合关键词含"的"时拆分为 sender + subject 搜索。

    Returns:
        list - 符合条件的 mail_id 列表（最新的在前）
    """
    # 解析复合关键词："hsbc的电子结单" → sender_kw="hsbc", subject_kw="电子结单"
    sender_kw = None
    subject_kw = None
    if "的" in query:
        parts = query.split("的", 1)
        if len(parts) == 2 and len(parts[0].strip()) >= 2 and len(parts[1].strip()) >= 2:
            sender_kw = parts[0].strip()
            subject_kw = parts[1].strip()

    # 方式1：IMAP SEARCH 服务器端过滤
    try:
        if sender_kw and subject_kw:
            # 复合搜索：FROM + SUBJECT
            search_criteria = f'(OR (FROM "{sender_kw}") (SUBJECT "{sender_kw}")) (SUBJECT "{subject_kw}")'
        else:
            search_criteria = f'(OR (FROM "{query}") (SUBJECT "{query}"))'
        status, messages = mail.search(None, search_criteria)
        if status == "OK" and messages[0].strip():
            ids = messages[0].split()
            ids = ids[-num_emails:]
            ids.reverse()
            return ids
    except Exception:
        pass

    # 方式1b：复合搜索失败时，退化为只用 sender_kw 搜索
    if sender_kw:
        try:
            search_criteria = f'(OR (FROM "{sender_kw}") (SUBJECT "{sender_kw}"))'
            status, messages = mail.search(None, search_criteria)
            if status == "OK" and messages[0].strip():
                ids = messages[0].split()
                ids = ids[-num_emails:]
                ids.reverse()
                return ids
        except Exception:
            pass

    # 方式2：回退为拉取最新 50 封本地过滤
    try:
        status, messages = mail.search(None, "ALL")
        if status != "OK":
            return []
        all_ids = messages[0].split()
        if not all_ids:
            return []

        candidate_ids = all_ids[-50:]
        candidate_ids.reverse()
        matched = []

        # 本地匹配：复合关键词时两个部分都要命中，单关键词时命中任一
        kw_parts = [sender_kw.lower(), subject_kw.lower()] if (sender_kw and subject_kw) else [query.lower()]

        for mail_id in candidate_ids:
            status, header_data = mail.fetch(mail_id, "(BODY[HEADER.FIELDS (FROM SUBJECT)])")
            if status != "OK":
                continue
            header_text = header_data[0][1].decode("utf-8", errors="replace").lower()
            if all(kw in header_text for kw in kw_parts):
                matched.append(mail_id)
                if len(matched) >= num_emails:
                    break
        return matched
    except Exception:
        return []


class EmailReaderTool(BaseTool):
    """读取收件箱邮件工具"""

    name: str = "email_reader"
    description: str = "读取收件箱中的最新邮件"
    args_schema: Type[BaseModel] = EmailReaderInput

    def _run(self, num_emails: int = 10) -> str:
        """
        读取收件箱最新 N 封邮件

        Args:
            num_emails: 读取邮件数量，默认10封

        Returns:
            格式化的邮件列表字符串
        """
        config = _get_email_config()

        if not config["email_address"] or not config["email_password"]:
            return "错误：邮件账号或密码未配置，请设置 EMAIL_ADDRESS 和 EMAIL_PASSWORD。"

        if not config["imap_server"]:
            return "错误：IMAP 服务器未配置，请设置 IMAP_SERVER。"

        mail = None
        try:
            # 连接 IMAP 服务器（SSL）
            mail = imaplib.IMAP4_SSL(
                config["imap_server"],
                config["imap_port"],
                timeout=30
            )
        except imaplib.IMAP4.error as e:
            return f"IMAP 连接失败：{str(e)}"
        except TimeoutError:
            return "IMAP 连接超时，请检查服务器地址和端口配置。"
        except Exception as e:
            return f"IMAP 连接异常：{str(e)}"

        try:
            # 登录认证
            mail.login(config["email_address"], config["email_password"])
        except imaplib.IMAP4.error as e:
            mail.logout()
            return (
                f"IMAP 认证失败：{str(e)}。\n"
                "请确认：1) 邮箱地址正确；2) 密码必须是邮箱后台生成的【授权码】（QQ/163/Gmail 均如此），不是登录密码。"
            )
        except Exception as e:
            mail.logout()
            return f"IMAP 登录异常：{str(e)}"

        # QQ/163 等国内邮箱 IMAP 要求登录后发送 ID 命令声明客户端身份，
        # 否则服务器会在下一条命令时直接断开连接（表现为 socket error: EOF）。
        try:
            imaplib.Commands["ID"] = ("AUTH",)
            id_args = (
                '("name" "langchain-email-agent" '
                '"version" "1.0" '
                '"vendor" "myclient" '
                '"contact" "admin@example.com")'
            )
            mail._simple_command("ID", id_args)
        except Exception:
            # 非 QQ/163 邮箱可能不支持 ID 命令，忽略即可
            pass

        try:
            # 选择收件箱
            mail.select("INBOX")

            # 搜索所有邮件
            status, messages = mail.search(None, "ALL")
            if status != "OK":
                return "无法搜索收件箱邮件。"

            mail_ids = messages[0].split()
            if not mail_ids:
                return "收件箱为空，没有邮件。"

            # 获取最新 N 封邮件
            latest_ids = mail_ids[-num_emails:]
            latest_ids.reverse()  # 最新的排前面

            results = []
            for i, mail_id in enumerate(latest_ids, 1):
                status, msg_data = mail.fetch(mail_id, "(RFC822)")
                if status != "OK":
                    continue

                raw_email = msg_data[0][1]
                msg = email.message_from_bytes(raw_email)

                # 解析邮件信息
                sender = _decode_mime_header(msg.get("From", "未知发件人"))
                subject = _decode_mime_header(msg.get("Subject", "无主题"))
                date = msg.get("Date", "未知日期")
                body = _extract_body(msg)
                body_summary = body.strip()[:200] if body else "（无正文内容）"

                results.append(
                    f"--- 邮件 {i} ---\n"
                    f"发件人：{sender}\n"
                    f"主题：{subject}\n"
                    f"日期：{date}\n"
                    f"正文摘要：{body_summary}\n"
                )

            mail.logout()

            if not results:
                return "未能成功读取任何邮件。"

            return f"共读取 {len(results)} 封邮件：\n\n" + "\n".join(results)

        except imaplib.IMAP4.error as e:
            return f"IMAP 操作错误：{str(e)}"
        except TimeoutError:
            return "IMAP 操作超时，请稍后重试。"
        except Exception as e:
            return f"读取邮件异常：{str(e)}"
        finally:
            try:
                mail.logout()
            except Exception:
                pass

    async def _arun(self, num_emails: int = 10) -> str:
        """异步版本暂不支持"""
        return self._run(num_emails=num_emails)


class EmailSenderInput(BaseModel):
    """邮件发送工具输入"""
    input: str = Field(description='JSON字符串，格式：{"to": "收件人", "subject": "主题", "body": "正文"}')


class EmailSenderTool(BaseTool):
    """发送邮件工具"""

    name: str = "email_sender"
    description: str = "发送邮件。输入为JSON字符串，格式：{\"to\": \"收件人邮箱\", \"subject\": \"邮件主题\", \"body\": \"邮件正文\"}"
    args_schema: Type[BaseModel] = EmailSenderInput

    def _run(self, input: str) -> str:
        """
        发送邮件

        Args:
            input: JSON字符串，包含 to、subject、body 字段

        Returns:
            发送结果消息
        """
        # 解析输入参数
        try:
            params = json.loads(input)
        except json.JSONDecodeError:
            return "错误：输入格式无效，请提供有效的 JSON 字符串，格式：{\"to\": \"收件人\", \"subject\": \"主题\", \"body\": \"正文\"}"

        to_addr = params.get("to", "").strip()
        subject = params.get("subject", "").strip()
        body = params.get("body", "").strip()

        if not to_addr:
            return "错误：收件人地址不能为空。"
        if not subject:
            return "错误：邮件主题不能为空。"
        if not body:
            return "错误：邮件正文不能为空。"

        config = _get_email_config()

        if not config["email_address"] or not config["email_password"]:
            return "错误：邮件账号或密码未配置，请设置 EMAIL_ADDRESS 和 EMAIL_PASSWORD。"

        if not config["smtp_server"]:
            return "错误：SMTP 服务器未配置，请设置 SMTP_SERVER。"

        # 构建邮件
        msg = MIMEMultipart()
        msg["From"] = config["email_address"]
        msg["To"] = to_addr
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain", "utf-8"))

        try:
            # 连接 SMTP 服务器（SSL）
            server = smtplib.SMTP_SSL(
                config["smtp_server"],
                config["smtp_port"],
                timeout=30
            )
        except smtplib.SMTPException as e:
            return f"SMTP 连接失败：{str(e)}"
        except TimeoutError:
            return "SMTP 连接超时，请检查服务器地址和端口配置。"
        except Exception as e:
            return f"SMTP 连接异常：{str(e)}"

        try:
            # 登录认证
            server.login(config["email_address"], config["email_password"])
        except smtplib.SMTPAuthenticationError as e:
            server.quit()
            return (
                f"SMTP 认证失败：{str(e)}。\n"
                "请确认：1) 邮箱地址正确；2) 密码必须是邮箱后台生成的【授权码】（QQ/163/Gmail 均如此），不是登录密码。"
            )
        except smtplib.SMTPException as e:
            server.quit()
            return f"SMTP 登录异常：{str(e)}"

        try:
            # 发送邮件
            server.sendmail(
                config["email_address"],
                to_addr,
                msg.as_string()
            )
            server.quit()
            return f"邮件发送成功！\n收件人：{to_addr}\n主题：{subject}"
        except smtplib.SMTPRecipientsRefused as e:
            return f"收件人地址被拒绝：{str(e)}"
        except smtplib.SMTPException as e:
            return f"邮件发送失败：{str(e)}"
        except Exception as e:
            return f"发送邮件异常：{str(e)}"
        finally:
            try:
                server.quit()
            except Exception:
                pass

    async def _arun(self, input: str) -> str:
        """异步版本暂不支持"""
        return self._run(input=input)
