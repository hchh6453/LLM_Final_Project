import re
from dataclasses import dataclass, asdict

@dataclass
class EmailSchema:
    """MCP Schema：郵件的標準化結構"""
    raw_email: str
    sender_domain: str
    subject: str
    email_body: str
    metadata: dict

def parse_email_to_mcp(raw_text: str) -> EmailSchema:
    """
    將純文字郵件解析為結構化 MCP 物件
    模擬真實 MCP 協議的 Schema 映射
    """
    # 解析寄件者
    sender_match = re.search(r'From:\s*(.+)', raw_text, re.IGNORECASE)
    sender = sender_match.group(1).strip() if sender_match else "unknown@unknown.com"
    
    # 提取 domain
    domain_match = re.search(r'@([\w\.-]+)', sender)
    if domain_match:
        sender_domain = domain_match.group(1)
    else:
        # ✅ 改成 "not_extracted" 而不是 "unknown.com"
        # 避免 LLM 把解析失敗誤判為可疑域名
        sender_domain = "not_extracted"
    
    # 解析主旨
    subject_match = re.search(r'Subject:\s*(.+)', raw_text, re.IGNORECASE)
    subject = subject_match.group(1).strip() if subject_match else ""
    
    # 解析信件本文（From/Subject 以下的部分）
    body_match = re.split(r'Subject:.+\n', raw_text, flags=re.IGNORECASE)
    email_body = body_match[-1].strip() if len(body_match) > 1 else raw_text
    
    # metadata
    metadata = {
        "has_urgency_keywords": any(
            word in raw_text for word in ["立刻", "馬上", "緊急", "immediately", "urgent", "ASAP"]
        ),
        "has_money_request": any(
            word in raw_text for word in ["匯款", "轉帳", "付款", "wire transfer", "payment"]
        ),
        "char_count": len(raw_text)
    }
    
    return EmailSchema(
        raw_email=raw_text,
        sender_domain=sender_domain,
        subject=subject,
        email_body=email_body,
        metadata=metadata
    )