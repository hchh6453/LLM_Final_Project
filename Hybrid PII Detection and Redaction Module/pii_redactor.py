import json
import os
import re
from pathlib import Path

import requests
from dotenv import load_dotenv

# ==========================================================
# Week 15 - Subsystem B Hybrid PII Redaction Module
# Regex-based structured PII detection + local Ollama semantic PII detection
# ==========================================================

_MODULE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _MODULE_DIR.parent

load_dotenv(_PROJECT_ROOT / ".env")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3.2:3b")


PII_PATTERNS = {
    "EMAIL": r"\b[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\b",
    "PHONE": r"\b09\d{2}[-\s]?\d{3}[-\s]?\d{3}\b",
    "TW_ID": r"(?<![A-Za-z0-9])[A-Z][12]\d{8}(?![A-Za-z0-9])",
    "CARD": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    "IPV4": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
    "IPV6": r"\b(?:[0-9a-fA-F]{1,4}:){2,7}[0-9a-fA-F]{1,4}\b",
    "URL": r"https?://[^\s\)]+",
    "IBAN": r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b",
    "BIC": r"\b[A-Z]{6}[A-Z0-9]{2}([A-Z0-9]{3})?\b",
    "ETH_ADDRESS": r"\b0x[a-fA-F0-9]{40}\b",
    "BTC_ADDRESS": r"\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b"
}


REDACTION_ORDER = [
    "ETH_ADDRESS",
    "BTC_ADDRESS",
    "IBAN",
    "BIC",
    "EMAIL",
    "URL",
    "IPV6",
    "IPV4",
    "CARD",
    "PHONE",
    "TW_ID"
]


def regex_detect_and_mask(text):
    """
    Detect and mask structured PII using deterministic Regex rules.
    """
    masked_text = text
    detected_pii = []

    for pii_type in REDACTION_ORDER:
        pattern = PII_PATTERNS[pii_type]
        matches = re.findall(pattern, masked_text)

        if matches:
            clean_matches = []
            for match in matches:
                if isinstance(match, tuple):
                    clean_matches.append("".join(match))
                else:
                    clean_matches.append(match)

            masked_text = re.sub(pattern, f"[MASKED_{pii_type}]", masked_text)

            detected_pii.append({
                "type": pii_type,
                "method": "Regex",
                "count": len(clean_matches),
                "masked_as": f"[MASKED_{pii_type}]"
            })

    return masked_text, detected_pii


def extract_json_from_text(text):
    """
    Ollama may return additional text. This function extracts the JSON array.
    """
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        return []
    except Exception:
        pass

    start = text.find("[")
    end = text.rfind("]")

    if start != -1 and end != -1 and end > start:
        json_part = text[start:end + 1]
        try:
            parsed = json.loads(json_part)
            if isinstance(parsed, list):
                return parsed
            return []
        except Exception:
            return []

    return []


def ollama_semantic_detect(text):
    """
    Use local Ollama to detect semantic PII.
    Semantic PII includes names, addresses, job titles, internal project names,
    and company names.

    Regex-handled PII should not be returned here.
    """
    prompt = f"""
You are a PII detection assistant.

Identify semantic PII from the email text below.

Only detect these types:
- NAME: personal names
- ADDRESS: physical addresses or location details
- JOB_TITLE: job titles or organizational roles
- PROJECT: internal project names or confidential project codes
- COMPANY: company names if they identify a business entity

Do not include emails, phone numbers, credit cards, Taiwan IDs, URLs, IP addresses, IBANs, BICs, or crypto wallet addresses because those are already handled by Regex.

Return only a valid JSON array.
Do not include explanation.
Do not use markdown.

Format:
[
  {{"type": "NAME", "value": "Wang Xiao-Ming"}},
  {{"type": "PROJECT", "value": "Project Falcon"}}
]

Email text:
{text}
"""

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=90)
        response.raise_for_status()
        content = response.json()["message"]["content"]
        return extract_json_from_text(content)

    except Exception as e:
        # Do not crash the system if Ollama fails.
        return []


def semantic_mask(text, semantic_items):
    """
    Mask semantic PII detected by Ollama.
    """
    masked_text = text
    detected_pii = []
    valid_items = []

    for item in semantic_items:
        if not isinstance(item, dict):
            continue

        pii_type = item.get("type")
        value = item.get("value")

        if not pii_type or not value:
            continue

        pii_type = str(pii_type).upper().strip()
        value = str(value).strip()

        if value and value in masked_text:
            valid_items.append((pii_type, value))

    # Longer terms should be replaced first to avoid partial replacement.
    valid_items = sorted(valid_items, key=lambda x: len(x[1]), reverse=True)

    for pii_type, value in valid_items:
        count = masked_text.count(value)
        masked_text = masked_text.replace(value, f"[MASKED_{pii_type}]")

        detected_pii.append({
            "type": pii_type,
            "method": "Ollama LLM",
            "count": count,
            "masked_as": f"[MASKED_{pii_type}]"
        })

    return masked_text, detected_pii


def final_check(text):
    """
    Final Check Node:
    After redaction, scan the output again to verify whether structured PII remains.
    """
    remaining = []

    for pii_type, pattern in PII_PATTERNS.items():
        matches = re.findall(pattern, text)

        if matches:
            remaining.append({
                "type": pii_type,
                "count": len(matches)
            })

    if remaining:
        return "FAIL", remaining

    return "PASS", []


def redact_email(email_text):
    """
    Main function for Member C integration.

    Input:
        email_text: str

    Output:
        Dictionary containing:
        - original_email
        - masked_email
        - detected_pii
        - semantic_items_from_ollama
        - final_check_status
        - remaining_pii
    """
    regex_masked_email, regex_pii = regex_detect_and_mask(email_text)

    semantic_items = ollama_semantic_detect(regex_masked_email)
    semantic_masked_email, semantic_pii = semantic_mask(
        regex_masked_email, semantic_items)

    final_status, remaining_pii = final_check(semantic_masked_email)

    return {
        "original_email": email_text,
        "masked_email": semantic_masked_email,
        "detected_pii": regex_pii + semantic_pii,
        "semantic_items_from_ollama": semantic_items,
        "final_check_status": final_status,
        "remaining_pii": remaining_pii
    }
