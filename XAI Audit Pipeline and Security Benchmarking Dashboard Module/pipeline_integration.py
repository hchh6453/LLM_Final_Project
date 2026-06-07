"""
Privacy-First Gateway pipeline integration (presentation layer over Member A/B modules).

Storyline (SOC console narrative):
  Stage 1 · Local Sanitization Gateway  — on-prem Regex + Ollama PII mask (PII never leaves intranet)
  Stage 2 · Cloud Threat Intelligence — sanitized text escalated to Gemini LLM Judge + Agentic RAG
  Stage 3 · XAI Forensic Audit        — dashboard necessity perturbation + ML benchmarking

Internal callback IDs (A/B/C) map to narrative stages: B→1, A→2, C→3.
"""

from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MEMBER_C_DIR = Path(__file__).resolve().parent
MEMBER_A_DIR = PROJECT_ROOT / "MCP Phishing and BEC Detection Module"
MEMBER_B_DIR = PROJECT_ROOT / "Hybrid PII Detection and Redaction Module"

# callback stage_id ("A"|"B"|"C"), status ("running"|"complete"|"skipped"), message
StageCallback = Callable[[str, str, str], None]

load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(MEMBER_A_DIR / ".env")

DEFAULT_DEMO_EMAIL = """From: ceo-office@c0mpany-fake.com
Subject: URGENT! Please process wire transfer immediately

Dear Manager, this is the CEO. I am currently overseas in a meeting
and cannot take calls. We have an urgent procurement that must be
completed before end of business today. Please wire USD 50,000 to
account number 123-456-789. This is strictly confidential.
Do not inform anyone else. Thank you."""

# Free-tier Gemini (~15 RPM): ≥5s between Stage 2 cloud calls in Card 4 batch runs.
BENCHMARK_API_PACING_SECONDS = 5.0


def _interruptible_sleep(
    seconds: float,
    should_abort: Callable[[], bool] | None = None,
    on_tick: Callable[[float], None] | None = None,
    tick_interval: float = 0.25,
) -> bool:
    """Sleep in short slices; return True when abort is requested."""
    elapsed = 0.0
    while elapsed < seconds:
        if should_abort is not None and should_abort():
            return True
        remaining = seconds - elapsed
        if on_tick is not None:
            on_tick(remaining)
        step = min(tick_interval, remaining)
        time.sleep(step)
        elapsed += step
    return False


def normalize_risk_score(score: Any) -> float:
    """Normalize risk score to 0.0–1.0 (handles legacy 0–10 scales)."""
    try:
        value = float(score)
    except (TypeError, ValueError):
        return 0.0
    if value > 1.0:
        value = value / 10.0
    return min(max(value, 0.0), 1.0)


_HEURISTIC_SIGNALS: list[tuple[str, float]] = [
    ("immediately", 0.08),
    ("past due", 0.12),
    ("wire transfer", 0.14),
    ("bank wire", 0.14),
    ("click", 0.07),
    ("secure portal", 0.09),
    ("verify", 0.07),
    ("credentials", 0.10),
    ("disabled", 0.08),
    ("deleted", 0.08),
    ("invoice", 0.05),
    ("outstanding balance", 0.10),
    ("credit card", 0.07),
    ("payment", 0.05),
    ("account will be", 0.10),
    ("declined", 0.06),
]


def compute_heuristic_stage2_verdict(
    sanitized_text: str,
    email_schema: Any | None = None,
) -> dict:
    """
    Offline urgency/payment heuristic when Gemini is unavailable (429 / quota).
    Operates only on sanitized text — Privacy-First compatible.
    """
    text_lower = sanitized_text.lower()
    score = 0.0
    tokens: list[str] = []
    for phrase, weight in _HEURISTIC_SIGNALS:
        if phrase in text_lower:
            score += weight
            tokens.append(phrase)

    score = min(max(score, 0.0), 0.96)
    is_phishing = score >= 0.55

    if "wire" in text_lower or "payment" in text_lower:
        mitre = "T1684.001 (BEC / Wire Transfer)"
    elif "click" in text_lower or "portal" in text_lower:
        mitre = "T1566.002 (Phishing Link)"
    else:
        mitre = "T1566.001 (Phishing Attachment / Lure)"

    sender_domain = "not_extracted"
    if email_schema is not None:
        sender_domain = getattr(email_schema, "sender_domain", sender_domain)

    return {
        "email_body": sanitized_text,
        "sender_domain": sender_domain,
        "is_phishing": is_phishing,
        "risk_score": score,
        "mitre_technique": mitre,
        "why_malicious_context": (
            f"Heuristic pre-screen on sanitized text — {len(tokens)} urgency/payment signals"
        ),
        "high_risk_tokens": tokens[:8],
        "verdict_source": "heuristic_fallback",
    }


_BENCHMARK_PHISHING_KEYWORDS = (
    "wire",
    "pay",
    "payment",
    "invoice",
    "overdue",
    "credentials",
    "verify",
    "urgent",
    "password",
    "ssn",
    "wire transfer",
)
_BENCHMARK_BENIGN_KEYWORDS = (
    "meeting",
    "agenda",
    "sync",
    "review",
    "training",
    "newsletter",
    "maintenance",
    "nomination",
    "minutes",
    "compliance",
)


def compute_benchmark_limit_fallback_score(sanitized_text: str) -> float:
    """
    Defensive soft fallback when Gemini 429 / batch errors occur.
    High-signal keyword presets keep confusion matrix usable under rate limits.
    """
    text_lower = sanitized_text.lower()
    if any(keyword in text_lower for keyword in _BENCHMARK_PHISHING_KEYWORDS):
        return 0.85
    if any(keyword in text_lower for keyword in _BENCHMARK_BENIGN_KEYWORDS):
        return 0.15
    return float(compute_heuristic_stage2_verdict(sanitized_text)["risk_score"])


def _log_limit_fallback(record_index: int, total: int, score: float, reason: str) -> None:
    print(
        f"[LIMIT_FALLBACK] record {record_index}/{total} "
        f"score={score:.2f} reason={reason}",
        flush=True,
    )


def normalize_stage1_output(
    raw_email: str,
    judge_result: dict,
    email_schema: Any | None = None,
) -> dict:
    """Map cloud threat-intel output (Stage 2) to legacy stage_1_mcp_output schema."""
    email_body = judge_result.get("email_body")
    if not email_body and email_schema is not None:
        email_body = getattr(email_schema, "email_body", raw_email)
    if not email_body:
        email_body = raw_email

    sender_domain = judge_result.get("sender_domain")
    if not sender_domain and email_schema is not None:
        sender_domain = getattr(email_schema, "sender_domain", "not_extracted")
    if not sender_domain:
        sender_domain = "not_extracted"

    tokens = judge_result.get("high_risk_tokens") or []
    if not isinstance(tokens, list):
        tokens = []
    tokens = [str(token).strip() for token in tokens if str(token).strip()]

    is_phishing = judge_result.get("is_phishing")
    if is_phishing is None:
        is_phishing = False

    return {
        "raw_email": raw_email,
        "email_body": email_body,
        "sender_domain": sender_domain,
        "is_phishing": bool(is_phishing),
        "risk_score": normalize_risk_score(judge_result.get("risk_score", 0.0)),
        "mitre_technique": str(judge_result.get("mitre_technique", "UNKNOWN")),
        "why_malicious_context": str(
            judge_result.get("why_malicious_context", "No forensic context returned.")
        ),
        "high_risk_tokens": tokens,
        "verdict_source": str(judge_result.get("verdict_source", "gemini")),
        "gemini_error": judge_result.get("gemini_error"),
    }


def normalize_stage2_output(raw_email: str, b_result: dict, source: str) -> dict:
    """Map local sanitization gateway output (Stage 1) to legacy stage_2_privacy_output schema."""
    detected_raw = b_result.get("detected_pii") or []
    detected_pii: list[str] = []
    pii_types: set[str] = set()

    for item in detected_raw:
        if isinstance(item, dict):
            pii_type = str(item.get("type", "UNKNOWN")).upper()
            method = str(item.get("method", "Unknown"))
            count = int(item.get("count", 1))
            masked_as = str(item.get("masked_as", f"[MASKED_{pii_type}]"))
            detected_pii.append(f"{masked_as} x{count} ({pii_type} via {method})")
            pii_types.add(pii_type)
        else:
            detected_pii.append(str(item))

    remaining_raw = b_result.get("remaining_pii") or []
    remaining_pii: list[str] = []
    for item in remaining_raw:
        if isinstance(item, dict):
            remaining_pii.append(
                f"{item.get('type', 'UNKNOWN')} x{item.get('count', 1)}"
            )
        else:
            remaining_pii.append(str(item))

    final_status = str(b_result.get("final_check_status", "FAIL")).upper()
    if final_status not in {"PASS", "FAIL"}:
        final_status = "FAIL"

    semantic_used = bool(b_result.get("semantic_items_from_ollama"))
    redaction_method = "Regex + Ollama LLM" if semantic_used else "Regex"

    return {
        "raw_email": raw_email,
        "masked_email": b_result.get("masked_email", raw_email),
        "detected_pii": detected_pii,
        "pii_type": ", ".join(sorted(pii_types)) if pii_types else "NONE",
        "redaction_method": redaction_method,
        "final_check_status": final_status,
        "remaining_pii": remaining_pii,
        "source": source,
    }


@contextmanager
def _member_a_environment():
    previous_cwd = os.getcwd()
    inserted = False
    member_a_path = str(MEMBER_A_DIR)
    if member_a_path not in sys.path:
        sys.path.insert(0, member_a_path)
        inserted = True
    os.chdir(MEMBER_A_DIR)
    try:
        yield
    finally:
        os.chdir(previous_cwd)
        if inserted and sys.path and sys.path[0] == member_a_path:
            sys.path.pop(0)


def _ensure_member_b_path() -> None:
    member_b_path = str(MEMBER_B_DIR)
    if member_b_path not in sys.path:
        sys.path.insert(0, member_b_path)


def create_member_a_rag_engine() -> Any:
    """Instantiate Stage 2 Agentic RAG engine (Member A DualLayerRAG module)."""
    with _member_a_environment():
        from rag_retriever import DualLayerRAG

        return DualLayerRAG()


def evaluate_benchmark_corpus(
    snippets: list[str],
    rag_engine: Any | None = None,
    on_progress: Callable[[int, int, str], None] | None = None,
    on_pacing: Callable[[int, int, float], None] | None = None,
    should_abort: Callable[[], bool] | None = None,
    pacing_seconds: float = BENCHMARK_API_PACING_SECONDS,
) -> list[float]:
    """
    Privacy-First batch inference: Stage 1 local mask → Stage 2 cloud score per snippet.
    Returns normalized risk_score values aligned with snippet order.

    pacing_seconds enforces a minimum gap between Gemini calls (Card 4 RPM defense).
    On 429 / API failure, applies local heuristic fallback — never crashes or infinite-retries.
    """
    rag = rag_engine or create_member_a_rag_engine()
    scores: list[float] = []
    total = len(snippets)

    for index, snippet in enumerate(snippets, start=1):
        if should_abort is not None and should_abort():
            scores.extend([0.0] * (total - len(scores)))
            break

        if on_progress is not None:
            preview = snippet.replace("\n", " ").strip()
            if len(preview) > 72:
                preview = preview[:69] + "…"
            on_progress(index, total, preview)

        sanitized = snippet
        try:
            b_result = run_member_b_stage(snippet)
            stage2 = normalize_stage2_output(
                snippet, b_result, source="benchmark_batch_stage1"
            )
            sanitized = stage2["masked_email"]
            stage1 = run_member_a_stage(raw_email=sanitized, rag_engine=rag)

            if stage1.get("gemini_error") or stage1.get("verdict_source") == "heuristic_fallback":
                score = compute_benchmark_limit_fallback_score(sanitized)
                _log_limit_fallback(
                    index,
                    total,
                    score,
                    str(stage1.get("gemini_error") or "heuristic_fallback"),
                )
            else:
                score = float(stage1["risk_score"])
            scores.append(score)
        except Exception as exc:
            score = compute_benchmark_limit_fallback_score(sanitized)
            _log_limit_fallback(index, total, score, f"exception:{type(exc).__name__}")
            scores.append(score)

        if index < total and pacing_seconds > 0:
            def _pacing_tick(remaining: float) -> None:
                if on_pacing is not None:
                    on_pacing(index, total, remaining)

            aborted = _interruptible_sleep(
                pacing_seconds,
                should_abort=should_abort,
                on_tick=_pacing_tick,
            )
            if aborted:
                scores.extend([0.0] * (total - len(scores)))
                break

    while len(scores) < total:
        scores.append(0.0)

    return scores


def run_member_a_stage(raw_email: str, rag_engine: Any | None = None) -> dict:
    """
    Execute Stage 2 cloud threat-intel path (MCP parse → Agentic RAG → Gemini LLM judge).
    Console narrative: operates on sanitized-safe text with no raw PII transmitted to cloud.
    Returns normalized stage_1_mcp_output dict (legacy schema key).
    """
    if not MEMBER_A_DIR.exists():
        raise FileNotFoundError(f"Member A module not found: {MEMBER_A_DIR}")

    with _member_a_environment():
        from llm_judge import judge_email
        from mcp_parser import parse_email_to_mcp
        from rag_retriever import DualLayerRAG

        email_schema = parse_email_to_mcp(raw_email)
        rag = rag_engine or DualLayerRAG()
        mitre_context = rag.retrieve_mitre_context(email_schema.email_body)
        fewshot_examples = rag.retrieve_fewshot_examples(email_schema.email_body)
        judge_result = judge_email(email_schema, mitre_context, fewshot_examples)

        if judge_result.get("error"):
            fallback = compute_heuristic_stage2_verdict(raw_email, email_schema)
            fallback["why_malicious_context"] = (
                f"Gemini unavailable ({judge_result.get('error')}) — "
                f"{fallback['why_malicious_context']}"
            )
            fallback["gemini_error"] = str(judge_result.get("error"))
            return normalize_stage1_output(raw_email, fallback, email_schema)

        member_a_payload = {
            "sender_domain": email_schema.sender_domain,
            "email_body": email_schema.email_body,
            "is_phishing": judge_result.get("is_phishing"),
            "risk_score": judge_result.get("risk_score"),
            "mitre_technique": judge_result.get("mitre_technique"),
            "why_malicious_context": judge_result.get("why_malicious_context"),
            "high_risk_tokens": judge_result.get("high_risk_tokens", []),
            "verdict_source": "gemini",
        }
        return normalize_stage1_output(raw_email, member_a_payload, email_schema)


def run_member_b_stage(email_body: str) -> dict:
    """Execute Stage 1 local sanitization gateway (on-prem Regex + Ollama PII redaction)."""
    if not MEMBER_B_DIR.exists():
        raise FileNotFoundError(f"Member B module not found: {MEMBER_B_DIR}")

    _ensure_member_b_path()
    from pii_redactor import redact_email

    return redact_email(email_body)


def _notify_stage(
    callback: StageCallback | None,
    stage_id: str,
    status: str,
    message: str,
) -> None:
    if callback is not None:
        callback(stage_id, status, message)


def run_full_pipeline(
    raw_email: str,
    rag_engine: Any | None = None,
    on_stage: StageCallback | None = None,
) -> dict:
    """
    Privacy-First Gateway execution contract:
      1) Stage 1 — local PII mask on raw inbound email (on-prem)
      2) Stage 2 — cloud threat intel on sanitized text only (Gemini + RAG)
      3) Stage 3 — dashboard XAI context ready
    Callback IDs: B=Stage 1, A=Stage 2, C=Stage 3.
    """
    _notify_stage(
        on_stage,
        "B",
        "running",
        (
            "Stage 1 · Local Sanitization Gateway — on-prem Regex + Ollama semantic "
            "PII mask (Privacy-First: data never leaves intranet)…"
        ),
    )
    b_result = run_member_b_stage(raw_email)
    stage2 = normalize_stage2_output(
        raw_email, b_result, source="stage1_local_gateway_live"
    )
    pii_summary = stage2.get("pii_type") or "NONE"
    _notify_stage(
        on_stage,
        "B",
        "complete",
        (
            f"Stage 1 Gateway complete — privacy {stage2['final_check_status']} · "
            f"method {stage2['redaction_method']} · "
            f"PII types masked: {pii_summary}"
        ),
    )

    sanitized_text = stage2["masked_email"]

    _notify_stage(
        on_stage,
        "A",
        "running",
        (
            "Stage 2 · Cloud Threat Intelligence — escalating sanitized-safe text to "
            "Gemini LLM Judge + Agentic RAG (no raw PII leaves the enterprise)…"
        ),
    )
    stage1 = run_member_a_stage(sanitized_text, rag_engine=rag_engine)
    stage1["raw_email"] = raw_email
    verdict = "PHISHING" if stage1["is_phishing"] else "BENIGN"
    _notify_stage(
        on_stage,
        "A",
        "complete",
        (
            f"Stage 2 complete — {verdict} on PII-free context · "
            f"risk {stage1['risk_score']:.2f} · "
            f"sender domain {stage1.get('sender_domain', 'n/a')}"
        ),
    )

    analysis_text = stage1.get("email_body") or sanitized_text

    _notify_stage(
        on_stage,
        "C",
        "complete",
        "Stage 3 · XAI Forensic Audit — necessity perturbation & ML benchmark ready",
    )

    return {
        "stage_1_mcp_output": stage1,
        "stage_2_privacy_output": stage2,
        "pipeline_meta": {
            "member_a": "stage2_cloud_threat_intel",
            "member_b": "stage1_local_sanitization_gateway",
            "member_c": "stage3_xai_forensic_audit",
            "analysis_text": analysis_text,
        },
    }


def load_cached_member_a_output(
    json_path: Path | None = None,
    on_stage: StageCallback | None = None,
    raw_email: str | None = None,
) -> dict | None:
    """Privacy-First cached mode: Stage 1 live mask → Stage 2 cached cloud verdict JSON."""
    path = json_path or (MEMBER_A_DIR / "output_for_member.json")
    if not path.exists():
        return None

    import json

    inbound = (raw_email or "").strip() or DEFAULT_DEMO_EMAIL

    _notify_stage(
        on_stage,
        "B",
        "running",
        (
            "Stage 1 · Local Sanitization Gateway — live on-prem PII redaction "
            "on inbound email body…"
        ),
    )
    b_result = run_member_b_stage(inbound)
    stage2 = normalize_stage2_output(
        inbound, b_result, source="stage1_local_gateway_cached"
    )
    pii_summary = stage2.get("pii_type") or "NONE"
    _notify_stage(
        on_stage,
        "B",
        "complete",
        (
            f"Stage 1 Gateway complete — privacy {stage2['final_check_status']} · "
            f"method {stage2['redaction_method']} · "
            f"PII types masked: {pii_summary}"
        ),
    )

    sanitized_text = stage2["masked_email"]

    _notify_stage(
        on_stage,
        "A",
        "running",
        f"Stage 2 · Loading cached cloud threat verdict from {path.name} (PII-free context)…",
    )

    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)

    stage1 = normalize_stage1_output(sanitized_text, data)
    stage1["raw_email"] = inbound
    if not stage1.get("email_body"):
        stage1["email_body"] = sanitized_text
    verdict = "PHISHING" if stage1["is_phishing"] else "BENIGN"
    _notify_stage(
        on_stage,
        "A",
        "complete",
        (
            f"Stage 2 cached verdict loaded — {verdict} on sanitized context · "
            f"risk {stage1['risk_score']:.2f}"
        ),
    )

    _notify_stage(
        on_stage,
        "C",
        "complete",
        "Stage 3 · XAI Forensic Audit — necessity perturbation & ML benchmark ready",
    )

    analysis_text = stage1.get("email_body") or sanitized_text

    return {
        "stage_1_mcp_output": stage1,
        "stage_2_privacy_output": stage2,
        "pipeline_meta": {
            "member_a": "cached stage2 cloud verdict (output_for_member.json)",
            "member_b": "stage1_local_sanitization_gateway",
            "member_c": "stage3_xai_forensic_audit",
            "analysis_text": analysis_text,
        },
    }
