import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai

_MODULE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _MODULE_DIR.parent

load_dotenv(_PROJECT_ROOT / ".env")
load_dotenv(_MODULE_DIR / ".env")

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

REQUIRED_FIELDS = {'is_phishing', 'risk_score', 'mitre_technique',
                   'why_malicious_context', 'high_risk_tokens'}

MAX_RATE_LIMIT_RETRIES = 2
MAX_RATE_LIMIT_WAIT_SEC = 8


def _interruptible_sleep(seconds: float) -> None:
    """Sleep in short chunks so Ctrl+C and job cancellation can interrupt sooner."""
    remaining = max(0.0, seconds)
    while remaining > 0:
        time.sleep(min(0.5, remaining))
        remaining -= 0.5

def safe_parse(result_text):
    clean = result_text.replace("```json", "").replace("```", "").strip()
    try:
        parsed = json.loads(clean)
        if not REQUIRED_FIELDS.issubset(parsed.keys()):
            raise ValueError("Missing required fields")
        return parsed
    except Exception:
        match = re.search(r'\{.*\}', clean, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
        return {"error": "parse_failed", "is_phishing": None,
                "risk_score": 0.0, "mitre_technique": "unknown",
                "why_malicious_context": "Parse failed",
                "high_risk_tokens": []}

def judge_email(email_schema, mitre_context, fewshot_examples):

    fewshot_str = "\n".join([
        f"Example: {ex['email'][:100]}...\nLabel: {ex['label']}, Reason: {ex['reason']}"
        for ex in fewshot_examples
    ])
    mitre_str = "\n".join([
        f"- {ctx['technique']}: {ctx['description'][:150]}"
        for ctx in mitre_context
    ])

    prompt = f"""You are a professional cybersecurity analyst. Analyze the following email for phishing or BEC attack intent.

## Structured Email (MCP Schema)
- Sender Domain: {email_schema.sender_domain}
- Subject: {email_schema.subject}
- Body: {email_schema.email_body}
- Contains Urgency Keywords: {email_schema.metadata['has_urgency_keywords']}
- Contains Money Request: {email_schema.metadata['has_money_request']}

IMPORTANT RULES:
- If Sender Domain is "not_extracted", do NOT treat this as a suspicious signal.
  Base your judgment solely on the email body content.
- Only classify as phishing if the body itself contains deceptive intent,
  urgency manipulation, credential requests, or malicious links/attachments.
- Legitimate emails (meeting notices, internal reports, academic CFPs) 
  should be classified as benign even if the domain is missing.
1. T1684.001 — Email impersonates executive/official (CEO/CFO) with NO link and NO attachment.
               Pure identity deception for money transfer. This is the correct technique for BEC emails.
2. T1684.002 — Email spoofing techniques that forge sender domain or headers to bypass authentication, but do NOT contain typical phishing content in the body.
3. T1534 — Email sent from a compromised internal account, appearing very legitimate, often part of an internal thread, used for lateral movement or data exfiltration.
4. T1566.001 — Phishing email with malicious attachment (e.g. invoice   attachment, macro-enabled document, password-protected zip).
5. T1566.002 — Phishing email with malicious link (e.g. "click here", "verify your account" with suspicious URL).
6. T1598.003 — Phishing email containing fake survey or credential harvesting links, often impersonating a trusted entity, with intent to steal personal data via the link.
7. T1204.001 — Phishing email that tricks user into downloading a malicious file (e.g. "download update", "access your file here" with a drive-by download link).
8. T1204.002 — Phishing email that tricks user into opening a malicious attachment (e.g. "open attachment to view", "invoice.pdf.exe", "enable macros").
9. T1078 — Phishing email indicating use of stolen credentials, account takeover, or login from unknown location, often sent from compromised personal email accounts.
10. T1114 — Phishing email requesting sensitive documents or internal data, often with pretext of audit, management request, or urgent need for information.
11. T1036 — Phishing email using brand impersonation, fake logos, or spoofed sender to appear as a legitimate company, bank, government, or HR communication.
12. T1647 — Phishing email using legal threats, lawsuit warnings, tax audit notices, court summons, regulatory penalties, or fake legal notices to create urgency and fear.
13. T1560 — Phishing email containing password-protected attachments, encrypted archives, or passwords in the email body, used to bypass security controls and deliver malware.  
14. T1499 — Phishing email containing fake login pages, verification requests, or account suspension notices that trick users into entering credentials on a malicious site. 



## Relevant MITRE ATT&CK Techniques (Tier-1 RAG)
{mitre_str}
INSTRUCTION: Select the most specific sub-technique ID available.

## Similar Historical Cases (Tier-2 Few-Shot RAG)
{fewshot_str}

## Your Task
Return ONLY this JSON, no markdown, no extra text:
{{
  "is_phishing": true or false,
  "risk_score": float 0.0 to 1.0,
  "mitre_technique": "most specific ATT&CK ID",
  "why_malicious_context": "Max 80 characters, complete reasoning, cite only observable evidence",
  "high_risk_tokens": ["exact", "phrases", "from", "email", "body"]
}}"""

    attempt = 0
    while attempt < MAX_RATE_LIMIT_RETRIES:
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            return safe_parse(response.text.strip())

        except Exception as e:
            err = str(e)
            attempt += 1

            if any(code in err for code in ['429', '503', 'RESOURCE_EXHAUSTED', 'UNAVAILABLE', 'quota']):
                if attempt >= MAX_RATE_LIMIT_RETRIES:
                    print(f"  ⚠️ Rate limit exceeded after {MAX_RATE_LIMIT_RETRIES} retries — returning safe default score.")
                    return {"error": "rate_limit_exceeded", "is_phishing": None,
                            "risk_score": 0.0, "mitre_technique": "unknown",
                            "why_malicious_context": "Gemini rate limit (429) — safe default applied",
                            "high_risk_tokens": []}

                wait_match = re.search(r'retry in (\d+)', err)
                suggested = int(wait_match.group(1)) if wait_match else 5
                wait = min(suggested + 2, MAX_RATE_LIMIT_WAIT_SEC)

                print(
                    f"  ⏳ Rate limit retry {attempt}/{MAX_RATE_LIMIT_RETRIES}, "
                    f"waiting {wait}s (cap {MAX_RATE_LIMIT_WAIT_SEC}s)…"
                )
                _interruptible_sleep(wait)
            else:
                print(f"  ❌ Non-retryable API error: {err[:100]}")
                return {"error": err[:100], "is_phishing": None,
                        "risk_score": 0.0, "mitre_technique": "unknown",
                        "why_malicious_context": "API error",
                        "high_risk_tokens": []}

    return {"error": "max_retries_exceeded", "is_phishing": None,
            "risk_score": 0.0, "mitre_technique": "unknown",
            "why_malicious_context": "Gemini max retries exceeded",
            "high_risk_tokens": []}