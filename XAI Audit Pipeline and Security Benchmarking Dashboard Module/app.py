"""
Phishing & BEC Detection System — Privacy-First Gateway SOC Console
Member C · XAI Forensic Audit · ML Security Benchmarking · Week 15 Engineering Alignment

Pipeline storyline (console narrative):
  Stage 1 · Local Sanitization Gateway  — on-prem Regex + Ollama (PII never leaves intranet)
  Stage 2 · Cloud Threat Intelligence — sanitized text → Gemini + Agentic RAG
  Stage 3 · XAI Forensic Audit        — necessity perturbation + benchmark dashboard
"""

import copy
import html
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

from benchmark_corpus import (
    BENCHMARK_BENIGN_COUNT,
    BENCHMARK_CORPUS_VERSION,
    BENCHMARK_PHISHING_COUNT,
    BENCHMARK_TOTAL,
    BENIGN_CORPUS,
    PHISHING_CORPUS,
)
from pipeline_integration import (
    BENCHMARK_API_PACING_SECONDS,
    DEFAULT_DEMO_EMAIL,
    StageCallback,
    create_member_a_rag_engine,
    evaluate_benchmark_corpus,
    load_cached_member_a_output,
    normalize_stage2_output,
    run_full_pipeline,
    run_member_b_stage,
)

BENCHMARK_ABORT_KEY = "benchmark_abort_requested"
BENCHMARK_RUN_LABEL = f"Run Live Benchmark ({BENCHMARK_TOTAL} Set)"

# Callback IDs (A/B/C) → Privacy-First narrative stages for console display
STAGE_NARRATIVE: dict[str, dict[str, str | int]] = {
    "B": {"order": 1, "num": "1", "title": "Local Sanitization Gateway"},
    "A": {"order": 2, "num": "2", "title": "Cloud Threat Intelligence"},
    "C": {"order": 3, "num": "3", "title": "XAI Forensic Audit"},
}
STAGE_STATUS_ORDER = {"running": 0, "skipped": 1, "complete": 2}

# ---------------------------------------------------------------------------
# 1. Cross-module Mock Context Object (Privacy-First Gateway aligned schema)
# stage_1_mcp_output = Stage 2 cloud verdict on sanitized text (legacy key name)
# stage_2_privacy_output = Stage 1 local gateway redaction output (legacy key name)
# ---------------------------------------------------------------------------
MOCK_CONTEXT = {
    "stage_1_mcp_output": {
        "raw_email": (
            "Hi Manager, this is Daniel. Please wire USD 60,000 to account 12341007 "
            "immediately or your account will be frozen."
        ),
        "email_body": (
            "Hi Manager, this is [MASKED_NAME]. Please wire [MASKED_AMOUNT] to "
            "account [BANK_ACCOUNT] immediately or your account will be [MASKED_ACTION]."
        ),
        "sender_domain": "secure-bank-update.com",
        "is_phishing": True,
        "risk_score": 0.89,
        "mitre_technique": "T1566.004 (Phishing Intent)",
        "why_malicious_context": (
            "Gemini judged BEC urgency and wire-transfer intent on PII-free sanitized text — "
            "no raw personal data was transmitted to the cloud."
        ),
        "high_risk_tokens": ["immediately", "wire", "account", "frozen"],
    },
    "stage_2_privacy_output": {
        "raw_email": (
            "Hi Manager, this is Daniel. Please wire USD 60,000 to account 12341007 "
            "immediately or your account will be frozen."
        ),
        "masked_email": (
            "Hi Manager, this is [MASKED_NAME]. Please wire [MASKED_AMOUNT] to "
            "account [BANK_ACCOUNT] immediately or your account will be [MASKED_ACTION]."
        ),
        "detected_pii": [
            "Daniel (NAME)",
            "60,000 (AMOUNT)",
            "12341007 (BANK_ACCOUNT)",
        ],
        "pii_type": "NAME, AMOUNT, BANK_ACCOUNT",
        "redaction_method": "Regex + Ollama LLM",
        "final_check_status": "PASS",
        "remaining_pii": [],
        "source": "stage1_local_gateway_demo",
    },
}

EMPTY_IDLE_CONTEXT: dict = {
    "stage_1_mcp_output": {
        "raw_email": "",
        "email_body": "",
        "sender_domain": "",
        "is_phishing": False,
        "risk_score": 0.0,
        "mitre_technique": "",
        "why_malicious_context": "",
        "high_risk_tokens": [],
    },
    "stage_2_privacy_output": {
        "raw_email": "",
        "masked_email": "",
        "detected_pii": [],
        "pii_type": "NONE",
        "redaction_method": "—",
        "final_check_status": "IDLE",
        "remaining_pii": [],
        "source": "idle",
    },
    "pipeline_meta": {"idle": True},
}

MITRE_ID_PATTERN = re.compile(r"(T\d{4}(?:\.\d{3})?)")

SOC_BG = "#0e1117"
SOC_CARD_BG = "#161b22"
SOC_BORDER = "#30363d"
SOC_CYAN = "#58a6ff"
SOC_GREEN = "#3fb950"
SOC_RED = "#ff4b4b"
SOC_AMBER = "#d29922"
SOC_TEXT = "#e6edf3"
SOC_MUTED = "#8b949e"
SOC_EMPHASIS_PASS_BG = "#132719"
SOC_EMPHASIS_PASS_FG = "#56d364"
SOC_EMPHASIS_PASS_BORDER = "#238636"
SOC_EMPHASIS_FAIL_BG = "#2d1517"
SOC_EMPHASIS_FAIL_FG = "#ff9492"
SOC_EMPHASIS_FAIL_BORDER = "#da3633"
SOC_EMPHASIS_WARN_BG = "#2a2416"
SOC_EMPHASIS_WARN_FG = "#f0c14d"
SOC_EMPHASIS_WARN_BORDER = "#9e6a03"
SOC_EMPHASIS_INFO_BG = "#131a24"
SOC_EMPHASIS_INFO_FG = "#79c0ff"
SOC_EMPHASIS_INFO_BORDER = "#1f6feb"
SOC_STATUS_HEADER_BG = "#21262d"
SOC_STATUS_HEADER_FG = "#e6edf3"
SOC_STATUS_BODY_BG = "#161b22"
SOC_STAGE_RUNNING_BG = "#0b2239"
SOC_STAGE_RUNNING_FG = "#b6e3ff"
SOC_STAGE_COMPLETE_FG = "#7ee787"
SOC_STAGE_SKIPPED_FG = "#f0c14d"

HIGHLIGHT_STYLE = (
    "background: #3d1418; "
    "color: #ff9492; "
    "border: 1px solid #f85149; "
    "border-radius: 4px; "
    "padding: 1px 6px; "
    "font-weight: 600;"
)


# ---------------------------------------------------------------------------
# 2. Core backend logic
# ---------------------------------------------------------------------------
class XAIAuditor:
    """Explainable AI auditor: necessity perturbation & adversarial prompt feedback."""

    PERTURBED_SCORE_MOCK = 0.15
    TRUSTWORTHY_DROP_THRESHOLD = 0.40

    def execute_necessity_test(
        self,
        original_text: str,
        high_risk_tokens: list[str],
        original_score: float,
    ) -> dict:
        perturbed_text = original_text
        for token in high_risk_tokens:
            perturbed_text = perturbed_text.replace(token, "[REMOVED]")

        perturbed_score = self.PERTURBED_SCORE_MOCK
        score_drop = original_score - perturbed_score
        is_trustworthy = score_drop > self.TRUSTWORTHY_DROP_THRESHOLD

        return {
            "perturbed_text": perturbed_text,
            "perturbed_score": perturbed_score,
            "score_drop": score_drop,
            "is_trustworthy": is_trustworthy,
        }

    def generate_adversarial_feedback(
        self,
        failed_tokens: list[str],
        current_mitre: str,
        raw_email: str,
    ) -> str:
        token_list = ", ".join(failed_tokens) if failed_tokens else "N/A"
        mitre_id = extract_mitre_id(current_mitre)
        email_excerpt = raw_email if len(raw_email) <= 600 else raw_email[:600] + "..."

        return f"""## Adversarial Refinement Prompt (Deterministic Template)

**Audit Status:** FAILED — Necessity perturbation score drop did not exceed threshold `{self.TRUSTWORTHY_DROP_THRESHOLD:.2f}`.

**MITRE Technique:** `{mitre_id}` (raw: `{current_mitre}`)
**Forensic Tokens Under Review:** `{token_list}`

### Instruction to Detection Model
The email below was flagged as **phishing**, but removing the listed high-risk tokens did **not** sufficiently reduce the normalized risk score (0.0–1.0). This pattern suggests **spurious correlation** or **LLM hallucination** rather than causal attribution.

**Sanitized Email Snapshot (Stage 2 cloud context — PII removed at Stage 1 gateway):**
```
{email_excerpt}
```

### Required Remediation Steps
1. Re-evaluate token attribution for: `{token_list}`
2. Cross-validate behavioral indicators against MITRE `{mitre_id}`
3. Expand the necessity perturbation token set and re-run the audit pipeline
4. **Do not** escalate to SOC quarantine until `causal_confidence >= 0.85`

### Expected JSON Response Schema
```json
{{
  "revised_tokens": ["..."],
  "revised_score": 0.0,
  "causal_confidence": 0.0,
  "mitre_validation": "{mitre_id}"
}}
```
"""


def extract_mitre_id(mitre_technique: str) -> str:
    """Safely parse MITRE ID from variable-format technique strings."""
    match = MITRE_ID_PATTERN.search(mitre_technique or "")
    return match.group(1) if match else "UNKNOWN"


@st.cache_data(show_spinner=False)
def _cached_evaluate_benchmark(
    mode_key: str,
    corpus_version: str,
    cache_token: int,
    snippets: tuple[str, ...],
) -> tuple[float, ...]:
    """
    Privacy-First Card 4 batch inference — disk-cached until Clear Results bumps cache_token.
    Alpha slider reruns read this cache only; no duplicate Gemini calls.
    """
    total = len(snippets)
    progress_bar = st.progress(
        0.0,
        text=(
            f"Auditing English Threat Profile 0/{total} — "
            "[Anti-429 Pacing Active] Initializing Privacy-First batch audit…"
        ),
    )
    status_line = st.empty()

    def on_progress(done: int, count: int, preview: str) -> None:
        if st.session_state.get(BENCHMARK_ABORT_KEY):
            return
        progress_bar.progress(
            done / count,
            text=(
                f"Auditing English Threat Profile {done}/{count} — "
                "Stage 1 mask → Stage 2 Gemini cloud verdict…"
            ),
        )
        status_line.markdown(
            f'<p class="soc-context-line"><strong>Batch audit:</strong> '
            f"{html.escape(preview)}</p>",
            unsafe_allow_html=True,
        )

    def on_pacing(done: int, count: int, seconds_remaining: float) -> None:
        if st.session_state.get(BENCHMARK_ABORT_KEY):
            return
        progress_bar.progress(
            done / count,
            text=(
                f"Auditing English Threat Profile {done}/{count} — "
                "[Anti-429 Pacing Active] Cooldown mechanism engaged…"
            ),
        )
        status_line.markdown(
            f'<p class="soc-context-line"><strong>Anti-429 Pacing Active</strong> — '
            f"Threat Profile {done}/{count} complete · "
            f"cooldown {max(seconds_remaining, 0.0):.0f}s before next Gemini call "
            f"(≥{BENCHMARK_API_PACING_SECONDS:.0f}s interval · golden set {count} emails)</p>",
            unsafe_allow_html=True,
        )

    def should_abort() -> bool:
        return bool(st.session_state.get(BENCHMARK_ABORT_KEY))

    rag = create_member_a_rag_engine()
    scores = evaluate_benchmark_corpus(
        list(snippets),
        rag_engine=rag,
        on_progress=on_progress,
        on_pacing=on_pacing,
        should_abort=should_abort,
        pacing_seconds=BENCHMARK_API_PACING_SECONDS,
    )

    if not should_abort():
        progress_bar.progress(
            1.0,
            text=(
                f"Auditing English Threat Profile {total}/{total} — "
                "golden set batch audit complete."
            ),
        )
        status_line.markdown(
            f'<p class="soc-context-line"><strong>Batch complete</strong> — '
            f"{total} English threat profiles scored · scores locked in cache "
            "(adjust α threshold without re-calling Gemini).</p>",
            unsafe_allow_html=True,
        )
    progress_bar.empty()
    status_line.empty()
    return tuple(scores)


def _stop_benchmark_jobs() -> None:
    """Signal in-flight benchmark / Gemini loops to exit and drop cached scores."""
    st.session_state[BENCHMARK_ABORT_KEY] = True
    st.session_state.pop("benchmark_batch_requested", None)
    st.session_state["benchmark_cache_token"] = (
        st.session_state.get("benchmark_cache_token", 0) + 1
    )
    _cached_evaluate_benchmark.clear()


def _clear_benchmark_cache() -> None:
    """Drop live benchmark session state and Streamlit disk cache."""
    _stop_benchmark_jobs()
    st.session_state.pop(BENCHMARK_ABORT_KEY, None)


def _request_live_benchmark_rerun() -> None:
    """Start (or restart) Card 4 live batch — bumps cache token so Gemini runs fresh."""
    st.session_state[BENCHMARK_ABORT_KEY] = False
    st.session_state["benchmark_batch_requested"] = True
    st.session_state["benchmark_cache_token"] = (
        st.session_state.get("benchmark_cache_token", 0) + 1
    )
    _cached_evaluate_benchmark.clear()


def resolve_benchmark_dataframe(mode: str) -> tuple[pd.DataFrame, str]:
    """
    Build benchmark DataFrame and resolve predicted_score source.

    Returns (dataframe, source_tag) where source_tag is:
    mock | live | live_pending
    """
    benchmark_df = build_benchmark_dataset()
    assert len(benchmark_df) == BENCHMARK_TOTAL
    assert (benchmark_df["label"] == 0).sum() == BENCHMARK_BENIGN_COUNT
    assert (benchmark_df["label"] == 1).sum() == BENCHMARK_PHISHING_COUNT

    if mode == "Demo (Mock Data)":
        return benchmark_df, "mock"

    if not st.session_state.get("benchmark_batch_requested"):
        return benchmark_df, "live_pending"

    cache_token = st.session_state.get("benchmark_cache_token", 0)
    live_scores = _cached_evaluate_benchmark(
        mode,
        BENCHMARK_CORPUS_VERSION,
        cache_token,
        tuple(benchmark_df["snippet"].tolist()),
    )

    benchmark_df = benchmark_df.copy()
    benchmark_df["predicted_score"] = list(live_scores)
    return benchmark_df, "live"


def build_benchmark_dataset() -> pd.DataFrame:
    """Build the 14-record Golden Balanced Test Set (7 Benign / 7 Phishing, EN-US)."""
    rows: list[dict] = []
    record_id = 1

    for subject, snippet, predicted_score, _locale in BENIGN_CORPUS:
        rows.append(
            {
                "id": record_id,
                "subject": subject,
                "snippet": snippet,
                "label": 0,
                "locale": "EN-US",
                "predicted_score": predicted_score,
            }
        )
        record_id += 1

    for subject, snippet, predicted_score, _locale in PHISHING_CORPUS:
        rows.append(
            {
                "id": record_id,
                "subject": subject,
                "snippet": snippet,
                "label": 1,
                "locale": "EN-US",
                "predicted_score": predicted_score,
            }
        )
        record_id += 1

    return pd.DataFrame(rows)


def predictions_from_threshold(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Binary predictions: 1 if predicted_score >= threshold, else 0."""
    return (scores >= threshold).astype(int)


def compute_benchmark_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "confusion_matrix": cm,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def format_risk_score(score: float) -> str:
    return f"{score:.2f} / 1.00"


def stage2_used_heuristic_fallback(stage1: dict) -> bool:
    return stage1.get("verdict_source") == "heuristic_fallback" or bool(
        stage1.get("gemini_error")
    )


def stage2_gemini_unavailable_message(stage1: dict) -> str:
    err = stage1.get("gemini_error") or stage1.get("why_malicious_context") or "unknown"
    return (
        "Stage 2 Gemini was unavailable (API rate limit / quota). "
        f"Offline heuristic pre-screen was applied instead — not a live LLM verdict. ({err})"
    )


def resolve_high_risk_tokens(stage1: dict) -> list[str]:
    """Return Stage 2 cloud forensic tokens only — never inject demo defaults."""
    tokens = stage1.get("high_risk_tokens")
    if not isinstance(tokens, list):
        return []
    return [str(token).strip() for token in tokens if str(token).strip()]


def tokens_present_in_text(text: str, tokens: list[str]) -> list[str]:
    """Subset of tokens that actually appear in the email body."""
    text_lower = text.lower()
    present: list[str] = []
    for token in tokens:
        if token.lower() in text_lower:
            present.append(token)
    return present


def run_xai_necessity_audit(
    stage1: dict,
    analysis_text: str,
    auditor: XAIAuditor | None = None,
) -> dict:
    """
    Run necessity perturbation only when Stage 2 cloud intel flags phishing and tokens
    exist in the email body. Benign emails skip XAI audit (not a failure).
    """
    auditor = auditor or XAIAuditor()
    is_phishing = bool(stage1.get("is_phishing"))
    risk_score = float(stage1.get("risk_score", 0.0))
    high_risk_tokens = resolve_high_risk_tokens(stage1)

    base = {
        "high_risk_tokens": high_risk_tokens,
        "tokens_in_text": [],
        "necessity": None,
        "risk_score": risk_score,
    }

    if not is_phishing:
        return {
            **base,
            "status": "skipped_benign",
            "message": (
                "Email classified BENIGN — necessity perturbation is not required. "
                "XAI audit applies only to phishing detections."
            ),
        }

    if not high_risk_tokens:
        return {
            **base,
            "status": "skipped_no_tokens",
            "message": (
                "Stage 2 flagged phishing but returned no high_risk_tokens — "
                "cannot run perturbation audit."
            ),
        }

    tokens_in_text = tokens_present_in_text(analysis_text, high_risk_tokens)
    if not tokens_in_text:
        return {
            **base,
            "status": "skipped_tokens_absent",
            "message": (
                "Stage 2 returned forensic tokens, but none appear in the email body — "
                "skipping perturbation to avoid spurious audit results."
            ),
        }

    necessity = auditor.execute_necessity_test(
        original_text=analysis_text,
        high_risk_tokens=tokens_in_text,
        original_score=risk_score,
    )
    return {
        **base,
        "status": "passed" if necessity["is_trustworthy"] else "failed",
        "tokens_in_text": tokens_in_text,
        "necessity": necessity,
        "message": "",
    }


def resolve_analysis_text(stage1: dict) -> str:
    return stage1.get("email_body") or stage1.get("raw_email") or ""


@st.cache_resource(show_spinner="Initializing Stage 2 Agentic RAG engine (first run may take a minute)...")
def get_member_a_rag_engine():
    """Cache cloud threat-intel RAG engine to avoid reloading on every Streamlit rerun."""
    from pipeline_integration import create_member_a_rag_engine

    return create_member_a_rag_engine()


def _clear_all_results() -> None:
    """Clear cached pipeline output so dashboard cards return to idle state."""
    st.session_state.pop("integrated_context", None)
    st.session_state.pop("pipeline_error", None)
    st.session_state.pop("run_pipeline", None)
    st.session_state.pop("demo_custom_context", None)
    _clear_benchmark_cache()


def render_pipeline_sidebar() -> str:
    """Render sidebar execution mode only."""
    with st.sidebar:
        st.markdown("### Pipeline Control")
        mode = st.radio(
            "Execution Mode",
            options=[
                "Demo (Mock Data)",
                "Live Pipeline (Stage 1 → 2 → 3)",
                "Cached Cloud Verdict JSON + Live Stage 1",
            ],
            index=0,
            help=(
                "Privacy-First Gateway: Stage 1 on-prem PII mask → Stage 2 cloud Gemini verdict "
                "on sanitized text → Stage 3 XAI audit. Demo uses mock Stage 2 scores."
            ),
        )

        st.button("Clear Results", width="stretch", on_click=_clear_all_results)
        st.button(
            "Stop Live API Jobs",
            width="stretch",
            on_click=_stop_benchmark_jobs,
            help="Cancel Card 4 benchmark batch and reset Gemini job flags.",
        )

        st.markdown("---")
        st.markdown("**Pipeline Stages**")
        st.markdown("- **1** Local Sanitization Gateway (Regex + Ollama)")
        st.markdown("- **2** Cloud Threat Intelligence (Gemini + Agentic RAG)")
        st.markdown("- **3** XAI Forensic Audit + ML Benchmark")

    return mode


def _load_sample_email() -> None:
    st.session_state["email_input"] = DEFAULT_DEMO_EMAIL
    _clear_all_results()


def _clear_email_input() -> None:
    st.session_state["email_input"] = ""
    _clear_all_results()


def _trigger_pipeline_run() -> None:
    st.session_state["run_pipeline"] = True


def render_main_email_console() -> str:
    """Main-page email input console (Module C entry point)."""
    if "email_input" not in st.session_state:
        st.session_state["email_input"] = ""

    with st.container(border=True):
        st.markdown(
            '<p style="color:#58a6ff;font-weight:700;font-size:1.05rem;margin-bottom:0.5rem;">'
            "✉️ Email Analysis Console · Enter Raw Email Content</p>",
            unsafe_allow_html=True,
        )
        st.caption(
            "Paste or type any email below. Click **Run Integrated Pipeline** to execute the "
            "Privacy-First Gateway (Stage 1 Local → Stage 2 Cloud → Stage 3 XAI). "
            "Demo mode applies live Stage 1 redaction with mock Stage 2 cloud scores."
        )

        btn_col1, btn_col2, btn_col3 = st.columns([2, 1, 1])
        with btn_col1:
            st.button(
                "Run Integrated Pipeline (Stage 1 → 2 → 3)",
                type="primary",
                width="stretch",
                on_click=_trigger_pipeline_run,
            )
        with btn_col2:
            st.button(
                "Load Sample Email",
                width="stretch",
                on_click=_load_sample_email,
            )
        with btn_col3:
            st.button(
                "Clear Email",
                width="stretch",
                on_click=_clear_email_input,
            )

        st.text_area(
            "raw_email",
            height=200,
            placeholder="From: sender@example.com\nSubject: ...\n\nEmail body...",
            label_visibility="collapsed",
            key="email_input",
        )

    return st.session_state.get("email_input", "")


def build_demo_context_with_custom_email(
    email_text: str,
    on_stage: StageCallback | None = None,
) -> dict:
    """Demo mode: live Stage 1 gateway redaction + mock Stage 2 cloud verdict on sanitized text."""
    ctx = copy.deepcopy(MOCK_CONTEXT)
    email = email_text.strip()
    ctx["stage_1_mcp_output"]["raw_email"] = email
    ctx["stage_1_mcp_output"]["email_body"] = email

    if on_stage is not None:
        on_stage(
            "A",
            "skipped",
            (
                "Demo mode — mock Stage 2 cloud threat scores "
                "(Gemini not invoked; sanitized-text verdict simulated)"
            ),
        )
        on_stage(
            "B",
            "running",
            (
                "Stage 1 · Local Sanitization Gateway — on-prem Regex + Ollama "
                "PII mask (Privacy-First: data never leaves intranet)…"
            ),
        )

    try:
        b_result = run_member_b_stage(email)
        ctx["stage_2_privacy_output"] = normalize_stage2_output(
            email, b_result, source="stage1_local_gateway_demo"
        )
        stage2 = ctx["stage_2_privacy_output"]
        ctx["stage_1_mcp_output"]["email_body"] = stage2["masked_email"]
        if on_stage is not None:
            pii_summary = stage2.get("pii_type") or "NONE"
            on_stage(
                "B",
                "complete",
                (
                    f"Stage 1 Gateway complete — privacy {stage2['final_check_status']} · "
                    f"method {stage2['redaction_method']} · "
                    f"PII types masked: {pii_summary}"
                ),
            )
    except Exception:
        ctx["stage_2_privacy_output"]["raw_email"] = email
        ctx["stage_2_privacy_output"]["masked_email"] = email
        ctx["stage_2_privacy_output"]["source"] = "stage1_local_gateway_demo"
        if on_stage is not None:
            on_stage(
                "B",
                "complete",
                "Stage 1 Gateway failed — showing unredacted inbound email (review required)",
            )

    if on_stage is not None:
        on_stage(
            "C",
            "complete",
            "Stage 3 · XAI Forensic Audit — necessity perturbation & ML benchmark ready",
        )

    return ctx


def _stage_status_icon(status: str) -> str:
    return {"running": "🔄", "complete": "✅", "skipped": "⏭️"}.get(status, "•")


def _stage_line_class(status: str) -> str:
    return {
        "running": "soc-stage-running",
        "complete": "soc-stage-complete",
        "skipped": "soc-stage-skipped",
    }.get(status, "soc-stage-running")


def _stage_heading(stage_id: str) -> str:
    meta = STAGE_NARRATIVE.get(stage_id)
    if not meta:
        return f"Stage {html.escape(stage_id)}"
    return f"Stage {meta['num']} · {meta['title']}"


def _stage_sort_key(stage_id: str, status: str) -> tuple[int, int]:
    meta = STAGE_NARRATIVE.get(stage_id, {})
    order = int(meta.get("order", 99))
    status_rank = STAGE_STATUS_ORDER.get(status, 9)
    return (order, status_rank)


def create_stage_callback(
    status_container,
) -> tuple[list[dict], StageCallback]:
    """Build a stage callback that writes to st.status and collects a persistent log."""
    stage_log: list[dict] = []
    stage_events: list[dict] = []

    def _render_status_lines() -> None:
        ordered = sorted(
            stage_events,
            key=lambda event: _stage_sort_key(event["stage_id"], event["status"]),
        )
        lines = []
        for event in ordered:
            icon = _stage_status_icon(event["status"])
            line_class = _stage_line_class(event["status"])
            heading = _stage_heading(event["stage_id"])
            lines.append(
                f'<div class="soc-stage-line {line_class}">'
                f"{icon} <strong>{heading}</strong> — "
                f'{html.escape(event["message"])}</div>'
            )
        status_container.markdown("".join(lines), unsafe_allow_html=True)

    def on_stage(stage_id: str, status: str, message: str) -> None:
        stage_events.append(
            {"stage_id": stage_id, "status": status, "message": message}
        )
        _render_status_lines()
        if status in {"complete", "skipped"}:
            stage_log.append(
                {"stage": stage_id, "status": status, "message": message}
            )
            heading = _stage_heading(stage_id)
            st.toast(f"{heading} finished", icon="✅")

    return stage_log, on_stage


def _attach_stage_log(context: dict, stage_log: list[dict]) -> dict:
    meta = context.setdefault("pipeline_meta", {})
    meta["stage_log"] = stage_log
    return context


def render_pipeline_stage_timeline(stage_log: list[dict]) -> None:
    """Persistent checklist of completed pipeline stages."""
    if not stage_log:
        return

    ordered = sorted(
        stage_log,
        key=lambda entry: _stage_sort_key(entry["stage"], entry["status"]),
    )
    items = "".join(
        f'<li><strong>{_stage_heading(entry["stage"])}</strong> '
        f'({_stage_status_icon(entry["status"])} {html.escape(entry["status"])}) — '
        f'{html.escape(entry["message"])}</li>'
        for entry in ordered
    )
    st.markdown(
        f"""
        <div class="soc-pipeline-log">
            <p class="soc-label" style="margin-bottom:0.45rem;">Pipeline Stage Log</p>
            <ul class="soc-pipeline-log-ul">{items}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def has_pipeline_results(mode: str) -> bool:
    """True when user has executed the pipeline and results should be shown."""
    if mode == "Demo (Mock Data)":
        return "demo_custom_context" in st.session_state
    return "integrated_context" in st.session_state


def build_runtime_context(mode: str, email_text: str | None) -> tuple[dict, str | None]:
    """Resolve dashboard context from mock data or integrated pipeline."""
    email = (email_text or "").strip()
    idle_notice = (
        "No analysis results yet. Enter an email and click "
        "Run Integrated Pipeline to populate Cards 1–3."
    )

    if mode == "Demo (Mock Data)":
        if not email:
            st.session_state.pop("demo_custom_context", None)
            return copy.deepcopy(EMPTY_IDLE_CONTEXT), idle_notice

        if st.session_state.get("run_pipeline"):
            with st.status(
                "Running Privacy-First demo pipeline (Stage 1 live → Stage 2 mock → Stage 3)…",
                expanded=True,
            ) as status:
                stage_log, on_stage = create_stage_callback(status)
                context = build_demo_context_with_custom_email(email, on_stage=on_stage)
                context = _attach_stage_log(context, stage_log)
                context["pipeline_meta"]["idle"] = False
                status.update(
                    label="Privacy-First demo complete · Stage 1 → 2 → 3",
                    state="complete",
                    expanded=False,
                )
            st.session_state["demo_custom_context"] = context
            st.session_state["run_pipeline"] = False

        if "demo_custom_context" in st.session_state:
            cached = st.session_state["demo_custom_context"]
            cached["stage_1_mcp_output"]["raw_email"] = email
            cached["stage_1_mcp_output"]["email_body"] = email
            return cached, None

        return copy.deepcopy(EMPTY_IDLE_CONTEXT), idle_notice

    if not email:
        return copy.deepcopy(EMPTY_IDLE_CONTEXT), (
            "Please enter email content in the analysis console above."
        )

    if not st.session_state.get("run_pipeline") and not has_pipeline_results(mode):
        return copy.deepcopy(EMPTY_IDLE_CONTEXT), (
            "Click Run Integrated Pipeline to execute "
            "Stage 1 Local Gateway → Stage 2 Cloud Intel → Stage 3 XAI."
        )

    if st.session_state.get("run_pipeline"):
        try:
            status_label = (
                "Running Privacy-First pipeline (cached Stage 2 → Stage 1 live → Stage 3)…"
                if mode == "Cached Cloud Verdict JSON + Live Stage 1"
                else "Running Privacy-First pipeline (Stage 1 → 2 → 3)…"
            )
            with st.status(status_label, expanded=True) as status:
                stage_log, on_stage = create_stage_callback(status)

                if mode == "Cached Cloud Verdict JSON + Live Stage 1":
                    context = load_cached_member_a_output(
                        on_stage=on_stage,
                        raw_email=email,
                    )
                    if context is None:
                        raise FileNotFoundError(
                            "Cached Stage 2 cloud verdict not found at "
                            "MCP Phishing and BEC Detection Module/output_for_member.json"
                        )
                else:
                    rag = get_member_a_rag_engine()
                    context = run_full_pipeline(
                        email, rag_engine=rag, on_stage=on_stage
                    )

                context = _attach_stage_log(context, stage_log)
                context["pipeline_meta"]["idle"] = False
                status.update(
                    label="Privacy-First pipeline complete · Stage 1 → 2 → 3",
                    state="complete",
                    expanded=False,
                )

            st.session_state["integrated_context"] = context
            st.session_state["pipeline_error"] = None
        except Exception as exc:
            st.session_state["pipeline_error"] = str(exc)
        finally:
            st.session_state["run_pipeline"] = False

    if st.session_state.get("pipeline_error"):
        return copy.deepcopy(EMPTY_IDLE_CONTEXT), st.session_state["pipeline_error"]

    if "integrated_context" in st.session_state:
        return st.session_state["integrated_context"], None

    return copy.deepcopy(EMPTY_IDLE_CONTEXT), idle_notice


# ---------------------------------------------------------------------------
# 3. UI helpers — SOC dark theme, charts
# ---------------------------------------------------------------------------
def inject_soc_dark_theme() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@400;500;700&display=swap');

        .stApp {{
            background: linear-gradient(160deg, {SOC_BG} 0%, #0a0d12 45%, #12151c 100%);
            color: {SOC_TEXT};
            font-family: 'Inter', sans-serif;
        }}

        .block-container {{ padding-top: 1.5rem; max-width: 1400px; }}

        header[data-testid="stHeader"] {{
            background: rgba(14, 17, 23, 0.85);
            border-bottom: 1px solid {SOC_BORDER};
        }}

        /* Main content typography — scoped to markdown, NOT alerts/emphasis panels */
        section[data-testid="stMain"] [data-testid="stMarkdownContainer"] p,
        section[data-testid="stMain"] [data-testid="stMarkdownContainer"] li,
        section[data-testid="stMain"] [data-testid="stMarkdownContainer"] h1,
        section[data-testid="stMain"] [data-testid="stMarkdownContainer"] h2,
        section[data-testid="stMain"] [data-testid="stMarkdownContainer"] h3,
        section[data-testid="stMain"] [data-testid="stMarkdownContainer"] h4,
        section[data-testid="stMain"] [data-testid="stMarkdownContainer"] h5,
        section[data-testid="stMain"] [data-testid="stMarkdownContainer"] h6,
        section[data-testid="stMain"] [data-testid="stCaptionContainer"] p,
        section[data-testid="stMain"] [data-testid="stMetricLabel"] p {{
            color: {SOC_TEXT};
        }}

        /* Streamlit native alerts — solid dark panels + high-contrast text */
        section[data-testid="stMain"] [data-testid="stAlert"],
        section[data-testid="stMain"] div[data-testid="stAlert"] {{
            background-color: {SOC_CARD_BG} !important;
            color: {SOC_TEXT} !important;
            border: 1px solid {SOC_BORDER} !important;
            border-radius: 10px !important;
        }}

        section[data-testid="stMain"] [data-testid="stAlert"] p,
        section[data-testid="stMain"] [data-testid="stAlert"] span,
        section[data-testid="stMain"] [data-testid="stAlert"] div,
        section[data-testid="stMain"] [data-testid="stAlert"] label {{
            color: inherit !important;
        }}

        section[data-testid="stMain"] .stSuccess,
        section[data-testid="stMain"] [data-testid="stAlert"][data-baseweb="notification"]:has(svg) {{
            background-color: {SOC_EMPHASIS_PASS_BG} !important;
            border: 1px solid {SOC_EMPHASIS_PASS_BORDER} !important;
            border-left: 4px solid {SOC_EMPHASIS_PASS_BORDER} !important;
            color: {SOC_EMPHASIS_PASS_FG} !important;
        }}

        section[data-testid="stMain"] .stSuccess p,
        section[data-testid="stMain"] .stSuccess span,
        section[data-testid="stMain"] .stSuccess div {{
            color: {SOC_EMPHASIS_PASS_FG} !important;
        }}

        section[data-testid="stMain"] .stWarning {{
            background-color: {SOC_EMPHASIS_WARN_BG} !important;
            border: 1px solid {SOC_EMPHASIS_WARN_BORDER} !important;
            border-left: 4px solid {SOC_EMPHASIS_WARN_BORDER} !important;
            color: {SOC_EMPHASIS_WARN_FG} !important;
        }}

        section[data-testid="stMain"] .stWarning p,
        section[data-testid="stMain"] .stWarning span,
        section[data-testid="stMain"] .stWarning div {{
            color: {SOC_EMPHASIS_WARN_FG} !important;
        }}

        section[data-testid="stMain"] .stInfo {{
            background-color: {SOC_EMPHASIS_INFO_BG} !important;
            border: 1px solid {SOC_EMPHASIS_INFO_BORDER} !important;
            border-left: 4px solid {SOC_EMPHASIS_INFO_BORDER} !important;
            color: {SOC_EMPHASIS_INFO_FG} !important;
        }}

        section[data-testid="stMain"] .stInfo p,
        section[data-testid="stMain"] .stInfo span,
        section[data-testid="stMain"] .stInfo div {{
            color: {SOC_EMPHASIS_INFO_FG} !important;
        }}

        /* ── st.status pipeline progress (Streamlit 1.57 → stExpander) ───── */
        section[data-testid="stMain"] [data-testid="stExpander"] {{
            background-color: {SOC_STATUS_BODY_BG} !important;
            border: 1px solid #484f58 !important;
            border-radius: 10px !important;
        }}

        section[data-testid="stMain"] [data-testid="stExpander"] details {{
            background: transparent !important;
            border: none !important;
        }}

        section[data-testid="stMain"] [data-testid="stExpander"] summary {{
            list-style: none !important;
            cursor: pointer !important;
            background-color: {SOC_STATUS_HEADER_BG} !important;
            border-bottom: 1px solid {SOC_BORDER} !important;
            padding: 0.65rem 0.85rem !important;
            border-radius: 9px 9px 0 0 !important;
            color: #e6edf3 !important;
            -webkit-text-fill-color: #e6edf3 !important;
        }}

        section[data-testid="stMain"] [data-testid="stExpander"] summary *,
        section[data-testid="stMain"] [data-testid="stExpander"] summary p,
        section[data-testid="stMain"] [data-testid="stExpander"] summary span,
        section[data-testid="stMain"] [data-testid="stExpander"] summary div {{
            color: #e6edf3 !important;
            -webkit-text-fill-color: #e6edf3 !important;
            background: transparent !important;
            opacity: 1 !important;
        }}

        section[data-testid="stMain"] [data-testid="stExpander"] summary:hover,
        section[data-testid="stMain"] [data-testid="stExpander"] details[open] > summary:hover {{
            background-color: #30363d !important;
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
            border-color: #58a6ff !important;
        }}

        section[data-testid="stMain"] [data-testid="stExpander"] summary:hover *,
        section[data-testid="stMain"] [data-testid="stExpander"] details[open] > summary:hover * {{
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
            background: transparent !important;
        }}

        section[data-testid="stMain"] [data-testid="stExpander"] details[open] > summary {{
            background-color: {SOC_STATUS_HEADER_BG} !important;
            color: #e6edf3 !important;
            -webkit-text-fill-color: #e6edf3 !important;
        }}

        section[data-testid="stMain"] [data-testid="stExpander"] details[open] > summary * {{
            color: #e6edf3 !important;
            -webkit-text-fill-color: #e6edf3 !important;
            background: transparent !important;
        }}

        section[data-testid="stMain"] [data-testid="stExpander"] summary svg,
        section[data-testid="stMain"] [data-testid="stExpander"] summary svg path {{
            color: {SOC_CYAN} !important;
            fill: {SOC_CYAN} !important;
            stroke: {SOC_CYAN} !important;
        }}

        section[data-testid="stMain"] [data-testid="stExpander"] summary:hover svg,
        section[data-testid="stMain"] [data-testid="stExpander"] summary:hover svg path,
        section[data-testid="stMain"] [data-testid="stExpander"] details[open] > summary:hover svg,
        section[data-testid="stMain"] [data-testid="stExpander"] details[open] > summary:hover svg path {{
            color: #58a6ff !important;
            fill: #58a6ff !important;
            stroke: #58a6ff !important;
        }}

        section[data-testid="stMain"] [data-testid="stExpander"] [data-testid="stExpanderDetails"],
        section[data-testid="stMain"] [data-testid="stExpander"] [data-testid="stExpanderDetails"] [data-testid="stMarkdownContainer"] {{
            background-color: {SOC_STATUS_BODY_BG} !important;
        }}

        section[data-testid="stMain"] [data-testid="stExpander"]:has([data-testid="stExpanderIconSpinner"]) {{
            border-left: 4px solid {SOC_EMPHASIS_INFO_BORDER} !important;
        }}

        section[data-testid="stMain"] [data-testid="stExpander"]:has([data-testid="stExpanderIconCheck"]) {{
            border-left: 4px solid {SOC_EMPHASIS_PASS_BORDER} !important;
        }}

        section[data-testid="stMain"] [data-testid="stExpander"]:has([data-testid="stExpanderIconCheck"]) summary,
        section[data-testid="stMain"] [data-testid="stExpander"]:has([data-testid="stExpanderIconCheck"]) summary * {{
            color: {SOC_STAGE_COMPLETE_FG} !important;
            -webkit-text-fill-color: {SOC_STAGE_COMPLETE_FG} !important;
        }}

        section[data-testid="stMain"] [data-testid="stExpander"]:has([data-testid="stExpanderIconCheck"]) summary:hover,
        section[data-testid="stMain"] [data-testid="stExpander"]:has([data-testid="stExpanderIconCheck"]) summary:hover * {{
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
        }}

        section[data-testid="stMain"] [data-testid="stExpander"]:has([data-testid="stExpanderIconCheck"]) summary svg,
        section[data-testid="stMain"] [data-testid="stExpander"]:has([data-testid="stExpanderIconCheck"]) summary svg path,
        section[data-testid="stMain"] [data-testid="stExpander"]:has([data-testid="stExpanderIconCheck"]) summary:hover svg,
        section[data-testid="stMain"] [data-testid="stExpander"]:has([data-testid="stExpanderIconCheck"]) summary:hover svg path {{
            color: #7ee787 !important;
            fill: #7ee787 !important;
            stroke: #7ee787 !important;
        }}

        section[data-testid="stMain"] [data-testid="stExpander"]:has([data-testid="stExpanderIconError"]) {{
            border-left: 4px solid {SOC_EMPHASIS_FAIL_BORDER} !important;
        }}

        .soc-stage-line {{
            font-size: 0.88rem;
            line-height: 1.55;
            padding: 0.5rem 0.75rem;
            margin: 0.35rem 0;
            border-radius: 8px;
            border: 1px solid {SOC_BORDER};
        }}

        .soc-stage-line strong {{ font-weight: 700 !important; color: inherit !important; }}

        .soc-stage-running {{
            background: {SOC_STAGE_RUNNING_BG} !important;
            color: {SOC_STAGE_RUNNING_FG} !important;
            border-color: {SOC_EMPHASIS_INFO_BORDER} !important;
        }}

        .soc-stage-complete {{
            background: {SOC_EMPHASIS_PASS_BG} !important;
            color: {SOC_STAGE_COMPLETE_FG} !important;
            border-color: {SOC_EMPHASIS_PASS_BORDER} !important;
        }}

        .soc-stage-skipped {{
            background: {SOC_EMPHASIS_WARN_BG} !important;
            color: {SOC_STAGE_SKIPPED_FG} !important;
            border-color: {SOC_EMPHASIS_WARN_BORDER} !important;
        }}

        section[data-testid="stMain"] [data-testid="stExpander"] .soc-stage-line,
        section[data-testid="stMain"] [data-testid="stExpander"] .soc-stage-line * {{
            background: transparent !important;
        }}

        section[data-testid="stMain"] [data-testid="stExpander"] .soc-stage-running,
        section[data-testid="stMain"] [data-testid="stExpander"] .soc-stage-running * {{
            color: {SOC_STAGE_RUNNING_FG} !important;
            -webkit-text-fill-color: {SOC_STAGE_RUNNING_FG} !important;
        }}

        section[data-testid="stMain"] [data-testid="stExpander"] .soc-stage-complete,
        section[data-testid="stMain"] [data-testid="stExpander"] .soc-stage-complete * {{
            color: {SOC_STAGE_COMPLETE_FG} !important;
            -webkit-text-fill-color: {SOC_STAGE_COMPLETE_FG} !important;
        }}

        section[data-testid="stMain"] [data-testid="stExpander"] .soc-stage-skipped,
        section[data-testid="stMain"] [data-testid="stExpander"] .soc-stage-skipped * {{
            color: {SOC_STAGE_SKIPPED_FG} !important;
            -webkit-text-fill-color: {SOC_STAGE_SKIPPED_FG} !important;
        }}

        /* Bordered input console */
        section[data-testid="stMain"] [data-testid="stVerticalBlockBorderWrapper"] {{
            background: {SOC_CARD_BG} !important;
            border-color: {SOC_BORDER} !important;
        }}

        /* ------------------------------------------------------------------ */
        /* Sidebar — SOC control panel overrides                               */
        /* ------------------------------------------------------------------ */
        [data-testid="stSidebar"] {{
            background-color: #0d1117 !important;
            border-right: 1px solid #30363d !important;
            box-shadow: inset -1px 0 0 rgba(88, 166, 255, 0.06);
        }}

        [data-testid="stSidebar"] > div:first-child {{
            background-color: #0d1117 !important;
            padding-top: 1.25rem;
        }}

        [data-testid="stSidebar"] [data-testid="stSidebarContent"] {{
            background-color: #0d1117 !important;
        }}

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] h4,
        [data-testid="stSidebar"] [data-testid="stMarkdown"] h3,
        [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
        [data-testid="stSidebar"] label[data-testid="stWidgetLabel"] {{
            color: {SOC_CYAN} !important;
            font-weight: 700 !important;
            letter-spacing: 0.02em;
        }}

        [data-testid="stSidebar"] [data-testid="stMarkdown"] p,
        [data-testid="stSidebar"] [data-testid="stMarkdown"] li {{
            color: #c9d1d9 !important;
            font-size: 0.88rem;
            line-height: 1.55;
        }}

        [data-testid="stSidebar"] [data-testid="stMarkdown"] strong {{
            color: {SOC_CYAN} !important;
            font-weight: 700 !important;
        }}

        [data-testid="stSidebar"] hr {{
            border: none !important;
            border-top: 1px solid #30363d !important;
            margin: 1.1rem 0 !important;
            opacity: 1 !important;
        }}

        /* Radio — Execution Mode */
        [data-testid="stSidebar"] [data-testid="stRadio"] label {{
            background: transparent !important;
            border-radius: 8px;
            padding: 0.35rem 0.25rem;
        }}

        [data-testid="stSidebar"] [data-testid="stRadio"] label p,
        [data-testid="stSidebar"] [data-testid="stRadio"] label span,
        [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label,
        [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label p,
        [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label span {{
            color: #c9d1d9 !important;
            -webkit-text-fill-color: #c9d1d9 !important;
            font-size: 0.92rem !important;
            font-weight: 500 !important;
            opacity: 1 !important;
        }}

        [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) p,
        [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) span,
        [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked),
        [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) p,
        [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) span {{
            color: #e6edf3 !important;
            -webkit-text-fill-color: #e6edf3 !important;
            font-weight: 700 !important;
        }}

        [data-testid="stSidebar"] [data-testid="stRadio"] input[type="radio"] {{
            accent-color: {SOC_CYAN};
            border-color: #30363d !important;
        }}

        /* Sidebar — Clear Results / Stop Live API Jobs buttons */
        [data-testid="stSidebar"] div.stButton button,
        [data-testid="stSidebar"] div.stButton button[data-testid="stBaseButton-secondary"] {{
            background-color: #21262d !important;
            color: {SOC_RED} !important;
            -webkit-text-fill-color: {SOC_RED} !important;
            border: 1px solid #30363d !important;
            border-radius: 8px !important;
            font-weight: 700 !important;
            font-size: 0.9rem !important;
            padding: 0.55rem 0.85rem !important;
            transition: background-color 0.18s ease, color 0.18s ease, border-color 0.18s ease;
        }}

        [data-testid="stSidebar"] div.stButton button p,
        [data-testid="stSidebar"] div.stButton button span,
        [data-testid="stSidebar"] div.stButton button div {{
            color: {SOC_RED} !important;
            -webkit-text-fill-color: {SOC_RED} !important;
            opacity: 1 !important;
            background: transparent !important;
        }}

        [data-testid="stSidebar"] div.stButton button:hover,
        [data-testid="stSidebar"] div.stButton button[data-testid="stBaseButton-secondary"]:hover {{
            background-color: #8b0000 !important;
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
            border-color: #f85149 !important;
            box-shadow: 0 0 12px rgba(248, 81, 73, 0.35);
        }}

        [data-testid="stSidebar"] div.stButton button:hover p,
        [data-testid="stSidebar"] div.stButton button:hover span {{
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
        }}

        [data-testid="stSidebar"] div.stButton button:active {{
            background-color: #6b0000 !important;
            color: #ffffff !important;
            border-color: #f85149 !important;
        }}

        /* ── Main console buttons (Streamlit 1.57: stButton → wrapper → stBaseButton-*) ── */
        section[data-testid="stMain"] div.stButton button {{
            border-radius: 8px !important;
            font-weight: 600 !important;
            font-size: 0.9rem !important;
            padding: 0.5rem 0.85rem !important;
            border: 1px solid #484f58 !important;
            background-color: #21262d !important;
            color: #e6edf3 !important;
            -webkit-text-fill-color: #e6edf3 !important;
        }}

        section[data-testid="stMain"] div.stButton button p,
        section[data-testid="stMain"] div.stButton button span,
        section[data-testid="stMain"] div.stButton button div {{
            color: #e6edf3 !important;
            -webkit-text-fill-color: #e6edf3 !important;
            opacity: 1 !important;
            background: transparent !important;
        }}

        section[data-testid="stMain"] div.stButton button[data-testid="stBaseButton-primary"] {{
            background-color: #1f6feb !important;
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
            border-color: #388bfd !important;
        }}

        section[data-testid="stMain"] div.stButton button[data-testid="stBaseButton-primary"] p,
        section[data-testid="stMain"] div.stButton button[data-testid="stBaseButton-primary"] span,
        section[data-testid="stMain"] div.stButton button[data-testid="stBaseButton-primary"] div {{
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
        }}

        section[data-testid="stMain"] div.stButton button[data-testid="stBaseButton-secondary"]:hover,
        section[data-testid="stMain"] div.stButton button[data-testid="stBaseButton-tertiary"]:hover {{
            background-color: #30363d !important;
            border-color: #58a6ff !important;
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
        }}

        section[data-testid="stMain"] div.stButton button[data-testid="stBaseButton-secondary"]:hover p,
        section[data-testid="stMain"] div.stButton button[data-testid="stBaseButton-secondary"]:hover span,
        section[data-testid="stMain"] div.stButton button[data-testid="stBaseButton-tertiary"]:hover p,
        section[data-testid="stMain"] div.stButton button[data-testid="stBaseButton-tertiary"]:hover span {{
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
        }}

        section[data-testid="stMain"] div.stButton button[data-testid="stBaseButton-primary"]:hover {{
            background-color: #388bfd !important;
            border-color: #58a6ff !important;
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
        }}

        section[data-testid="stMain"] div.stButton button[data-testid="stBaseButton-primary"]:hover p,
        section[data-testid="stMain"] div.stButton button[data-testid="stBaseButton-primary"]:hover span,
        section[data-testid="stMain"] div.stButton button[data-testid="stBaseButton-primary"]:hover div {{
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
        }}

        .soc-hero {{
            text-align: center;
            margin-bottom: 1.75rem;
            padding: 1.25rem 1rem 1.5rem;
            border: 1px solid {SOC_BORDER};
            border-radius: 16px;
            background: linear-gradient(135deg, rgba(22,27,34,0.95) 0%, rgba(14,17,23,0.98) 100%);
            box-shadow: 0 8px 32px rgba(0,0,0,0.45), inset 0 1px 0 rgba(88,166,255,0.08);
        }}

        .soc-hero h1 {{
            font-size: 1.85rem !important;
            font-weight: 700 !important;
            margin: 0 !important;
            background: linear-gradient(90deg, {SOC_CYAN}, {SOC_GREEN});
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}

        .soc-hero .subtitle {{
            color: {SOC_MUTED} !important;
            font-size: 0.95rem;
            margin-top: 0.5rem;
        }}

        .soc-hero .badge-row {{
            margin-top: 0.85rem;
            display: flex;
            justify-content: center;
            gap: 0.6rem;
            flex-wrap: wrap;
        }}

        .soc-badge {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            padding: 0.25rem 0.65rem;
            border-radius: 999px;
            border: 1px solid {SOC_EMPHASIS_INFO_BORDER};
            color: {SOC_EMPHASIS_INFO_FG} !important;
            background: {SOC_EMPHASIS_INFO_BG} !important;
        }}

        .soc-badge.danger {{
            color: {SOC_EMPHASIS_FAIL_FG} !important;
            border-color: {SOC_EMPHASIS_FAIL_BORDER};
            background: {SOC_EMPHASIS_FAIL_BG} !important;
        }}

        .soc-badge.ok {{
            color: {SOC_EMPHASIS_PASS_FG} !important;
            border-color: {SOC_EMPHASIS_PASS_BORDER};
            background: {SOC_EMPHASIS_PASS_BG} !important;
        }}

        .soc-card {{
            background: {SOC_CARD_BG};
            border: 1px solid {SOC_BORDER};
            border-radius: 14px;
            padding: 1.15rem 1.25rem;
            margin-bottom: 1rem;
            box-shadow: 0 4px 24px rgba(0, 0, 0, 0.35);
        }}

        .soc-card-header {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 1rem;
            font-weight: 700;
            color: {SOC_CYAN} !important;
            margin-bottom: 0.85rem;
            padding-bottom: 0.55rem;
            border-bottom: 1px solid {SOC_BORDER};
        }}

        .soc-label {{
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: {SOC_MUTED} !important;
            margin-bottom: 0.35rem;
        }}

        .soc-email-box {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.88rem;
            line-height: 1.75;
            color: {SOC_TEXT};
            background: #0d1117;
            border: 1px solid {SOC_BORDER};
            border-radius: 10px;
            padding: 0.9rem 1rem;
            margin: 0.4rem 0 0.75rem;
        }}

        .soc-masked-box {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.86rem;
            line-height: 1.7;
            color: {SOC_EMPHASIS_PASS_FG} !important;
            background: {SOC_EMPHASIS_PASS_BG};
            border: 1px solid {SOC_EMPHASIS_PASS_BORDER};
            border-left: 4px solid {SOC_EMPHASIS_PASS_BORDER};
            border-radius: 10px;
            padding: 0.85rem 1rem;
        }}

        .soc-meta-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.5rem 1rem;
            margin-top: 0.75rem;
            font-size: 0.84rem;
        }}

        .soc-meta-item strong {{
            color: {SOC_CYAN} !important;
            display: block;
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 0.15rem;
        }}

        .soc-meta-item span {{
            color: {SOC_TEXT} !important;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem;
        }}

        .soc-pii-ul {{
            margin: 0.35rem 0 0 1rem;
            padding: 0;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.82rem;
        }}

        .soc-pii-ul li {{ color: {SOC_AMBER} !important; margin-bottom: 0.25rem; }}

        /* ------------------------------------------------------------------ */
        /* Emphasis alert cards — parent tone + child inherit (anti gray-wash) */
        /* ------------------------------------------------------------------ */
        .soc-compliance-pass {{
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            margin-top: 0.65rem;
            padding: 0.45rem 0.85rem;
            border-radius: 8px;
            background: {SOC_EMPHASIS_PASS_BG};
            border: 1px solid {SOC_EMPHASIS_PASS_BORDER};
            border-left: 4px solid {SOC_EMPHASIS_PASS_BORDER};
            color: {SOC_EMPHASIS_PASS_FG} !important;
            font-size: 0.88rem;
            font-weight: 600;
        }}

        .soc-compliance-fail {{
            display: inline-flex;
            margin-top: 0.65rem;
            padding: 0.45rem 0.85rem;
            border-radius: 8px;
            background: {SOC_EMPHASIS_FAIL_BG};
            border: 1px solid {SOC_EMPHASIS_FAIL_BORDER};
            border-left: 4px solid {SOC_EMPHASIS_FAIL_BORDER};
            color: {SOC_EMPHASIS_FAIL_FG} !important;
            font-weight: 600;
        }}

        .soc-audit-pass {{
            background: {SOC_EMPHASIS_PASS_BG};
            border: 1px solid {SOC_EMPHASIS_PASS_BORDER};
            border-left: 4px solid {SOC_EMPHASIS_PASS_BORDER};
            border-radius: 10px;
            padding: 0.75rem 1rem;
            color: {SOC_EMPHASIS_PASS_FG} !important;
            font-size: 0.9rem;
            margin: 0.6rem 0;
        }}

        .soc-audit-fail {{
            background: {SOC_EMPHASIS_WARN_BG};
            border: 1px solid {SOC_EMPHASIS_WARN_BORDER};
            border-left: 4px solid {SOC_EMPHASIS_WARN_BORDER};
            border-radius: 10px;
            padding: 0.75rem 1rem;
            color: {SOC_EMPHASIS_WARN_FG} !important;
            font-size: 0.9rem;
            margin: 0.6rem 0;
        }}

        .soc-audit-skip {{
            background: {SOC_EMPHASIS_INFO_BG};
            border: 1px solid {SOC_EMPHASIS_INFO_BORDER};
            border-left: 4px solid {SOC_EMPHASIS_INFO_BORDER};
            border-radius: 10px;
            padding: 0.75rem 1rem;
            color: {SOC_EMPHASIS_INFO_FG} !important;
            font-size: 0.9rem;
            margin: 0.6rem 0;
        }}

        .soc-compliance-pass, .soc-compliance-pass * {{ color: {SOC_EMPHASIS_PASS_FG} !important; }}
        .soc-compliance-fail, .soc-compliance-fail * {{ color: {SOC_EMPHASIS_FAIL_FG} !important; }}
        .soc-audit-pass, .soc-audit-pass * {{ color: {SOC_EMPHASIS_PASS_FG} !important; }}
        .soc-audit-fail, .soc-audit-fail * {{ color: {SOC_EMPHASIS_WARN_FG} !important; }}
        .soc-audit-skip, .soc-audit-skip * {{ color: {SOC_EMPHASIS_INFO_FG} !important; }}

        section[data-testid="stMain"] [data-testid="stMarkdownContainer"] .soc-compliance-pass {{
            background: {SOC_EMPHASIS_PASS_BG} !important;
        }}

        section[data-testid="stMain"] [data-testid="stMarkdownContainer"] .soc-compliance-pass * {{
            background: transparent !important;
        }}

        section[data-testid="stMain"] [data-testid="stMarkdownContainer"] .soc-compliance-fail {{
            background: {SOC_EMPHASIS_FAIL_BG} !important;
        }}

        section[data-testid="stMain"] [data-testid="stMarkdownContainer"] .soc-compliance-fail * {{
            background: transparent !important;
        }}

        section[data-testid="stMain"] [data-testid="stMarkdownContainer"] .soc-audit-pass {{
            background: {SOC_EMPHASIS_PASS_BG} !important;
        }}

        section[data-testid="stMain"] [data-testid="stMarkdownContainer"] .soc-audit-pass * {{
            background: transparent !important;
        }}

        section[data-testid="stMain"] [data-testid="stMarkdownContainer"] .soc-audit-fail {{
            background: {SOC_EMPHASIS_WARN_BG} !important;
        }}

        section[data-testid="stMain"] [data-testid="stMarkdownContainer"] .soc-audit-fail * {{
            background: transparent !important;
        }}

        section[data-testid="stMain"] [data-testid="stMarkdownContainer"] .soc-audit-skip {{
            background: {SOC_EMPHASIS_INFO_BG} !important;
        }}

        section[data-testid="stMain"] [data-testid="stMarkdownContainer"] .soc-audit-skip * {{
            background: transparent !important;
        }}

        .soc-context-line {{
            font-size: 0.88rem;
            color: {SOC_MUTED} !important;
            margin-top: 0.5rem;
            line-height: 1.55;
        }}

        .soc-context-line strong {{ color: {SOC_TEXT} !important; }}

        .soc-pipeline-log {{
            background: {SOC_CARD_BG};
            border: 1px solid {SOC_BORDER};
            border-radius: 10px;
            padding: 0.75rem 1rem;
            margin: 0.75rem 0 1rem;
        }}

        .soc-pipeline-log-ul {{
            margin: 0;
            padding-left: 1.2rem;
            color: {SOC_TEXT} !important;
            font-size: 0.88rem;
            line-height: 1.55;
        }}

        .soc-pipeline-log-ul li {{
            margin-bottom: 0.35rem;
            color: {SOC_TEXT} !important;
        }}

        .soc-pipeline-log-ul li strong {{
            color: {SOC_CYAN} !important;
        }}

        .soc-score-flow {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.9rem;
            color: {SOC_MUTED} !important;
            text-align: center;
            margin: 0.35rem 0 0.6rem;
        }}

        .soc-score-flow .from {{ color: {SOC_RED} !important; font-weight: 700; }}
        .soc-score-flow .to {{ color: {SOC_GREEN} !important; font-weight: 700; }}
        .soc-score-flow .arrow {{ color: {SOC_CYAN} !important; margin: 0 0.35rem; }}

        .soc-perturbed {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.82rem;
            color: {SOC_MUTED};
            background: #0d1117;
            border: 1px dashed {SOC_BORDER};
            border-radius: 8px;
            padding: 0.65rem 0.85rem;
        }}

        .soc-adversarial-label {{
            color: {SOC_RED} !important;
            font-size: 0.88rem;
            font-weight: 700;
            margin: 0.75rem 0 0.4rem;
            letter-spacing: 0.02em;
        }}

        pre.soc-adversarial-prompt {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem;
            line-height: 1.65;
            color: #ff9492 !important;
            background: #0d1117 !important;
            border: 1px solid {SOC_BORDER};
            border-left: 4px solid {SOC_RED};
            border-radius: 10px;
            padding: 1rem 1.1rem;
            margin: 0;
            white-space: pre-wrap;
            word-wrap: break-word;
            overflow-x: auto;
            max-height: 420px;
            overflow-y: auto;
            box-shadow: inset 0 0 24px rgba(255, 75, 75, 0.04);
        }}

        section[data-testid="stMain"] [data-testid="stMarkdownContainer"] pre.soc-adversarial-prompt,
        section[data-testid="stMain"] [data-testid="stMarkdownContainer"] pre.soc-adversarial-prompt span,
        section[data-testid="stMain"] [data-testid="stMarkdownContainer"] pre.soc-adversarial-prompt code {{
            color: #ff9492 !important;
            background: transparent !important;
        }}

        [data-testid="stMetric"] {{
            background: #0d1117;
            border: 1px solid {SOC_BORDER};
            border-radius: 12px;
            padding: 0.75rem 0.5rem;
        }}

        [data-testid="stMetricLabel"] {{ color: {SOC_MUTED} !important; font-size: 0.78rem !important; }}
        [data-testid="stMetricValue"] {{
            color: {SOC_CYAN} !important;
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 1.35rem !important;
        }}

        div[data-testid="stVerticalBlockBorderWrapper"] {{
            background: {SOC_CARD_BG} !important;
            border: 1px solid {SOC_BORDER} !important;
            border-radius: 14px !important;
            box-shadow: 0 4px 24px rgba(0, 0, 0, 0.35) !important;
        }}

        #MainMenu {{ visibility: hidden; }}
        footer {{ visibility: hidden; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_soc_notice(kind: str, message: str) -> None:
    """High-contrast SOC notice (replaces st.success/warning/info on dark theme)."""
    css_class = {
        "success": "soc-audit-pass",
        "warning": "soc-audit-fail",
        "info": "soc-audit-skip",
        "error": "soc-compliance-fail",
    }.get(kind, "soc-audit-skip")
    st.markdown(
        f'<div class="{css_class}">{html.escape(message)}</div>',
        unsafe_allow_html=True,
    )


def render_card_header(title: str, icon: str = "◆") -> None:
    st.markdown(
        f'<div class="soc-card"><div class="soc-card-header">'
        f'<span class="icon">{icon}</span>{html.escape(title)}</div>',
        unsafe_allow_html=True,
    )


def render_card_footer() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def render_adversarial_prompt_block(prompt: str) -> None:
    """Render adversarial feedback with SOC dark-theme contrast (readable on dark UI)."""
    st.markdown(
        '<p class="soc-adversarial-label">'
        "Adversarial Refinement Prompt (Auto-Generated)</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<pre class="soc-adversarial-prompt">{html.escape(prompt)}</pre>',
        unsafe_allow_html=True,
    )


def highlight_high_risk_tokens(text: str, tokens: list[str]) -> str:
    escaped = html.escape(text)
    for token in sorted(tokens, key=len, reverse=True):
        escaped_token = html.escape(token)
        pattern = re.escape(escaped_token)
        replacement = f'<span style="{HIGHLIGHT_STYLE}">{escaped_token}</span>'
        escaped = re.sub(pattern, replacement, escaped, flags=re.IGNORECASE)
    return escaped


def render_privacy_metadata(stage2: dict) -> None:
    pii_items = "".join(
        f"<li>{html.escape(item)}</li>" for item in stage2["detected_pii"]
    )
    remaining = stage2["remaining_pii"]
    remaining_text = (
        "No residual PII"
        if len(remaining) == 0
        else ", ".join(html.escape(str(x)) for x in remaining)
    )

    st.markdown(
        f"""
        <div class="soc-meta-grid">
            <div class="soc-meta-item">
                <strong>detected_pii</strong>
                <ul class="soc-pii-ul">{pii_items}</ul>
            </div>
            <div class="soc-meta-item">
                <strong>pii_type</strong>
                <span>{html.escape(stage2["pii_type"])}</span>
            </div>
            <div class="soc-meta-item">
                <strong>redaction_method</strong>
                <span>{html.escape(stage2["redaction_method"])}</span>
            </div>
            <div class="soc-meta-item">
                <strong>source</strong>
                <span>{html.escape(stage2["source"])}</span>
            </div>
            <div class="soc-meta-item">
                <strong>remaining_pii</strong>
                <span>{remaining_text}</span>
            </div>
            <div class="soc-meta-item">
                <strong>final_check_status</strong>
                <span>{html.escape(stage2["final_check_status"])}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if stage2["final_check_status"] == "PASS":
        st.markdown(
            '<div class="soc-compliance-pass">'
            "● Privacy Compliance GREEN — final_check_status = PASS"
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="soc-compliance-fail">'
            "● Privacy Compliance ALERT — final_check_status = FAIL"
            "</div>",
            unsafe_allow_html=True,
        )


def _style_dark_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(SOC_CARD_BG)
    ax.tick_params(axis="x", colors=SOC_TEXT, labelsize=9)
    ax.tick_params(axis="y", colors=SOC_TEXT, labelsize=9)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(SOC_TEXT)
    ax.spines["bottom"].set_color(SOC_BORDER)
    ax.spines["left"].set_color(SOC_BORDER)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.label.set_color(SOC_MUTED)
    ax.yaxis.label.set_color(SOC_MUTED)


def render_confusion_matrix_figure(cm: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    fig.patch.set_facecolor(SOC_CARD_BG)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["0: Benign", "1: Phishing"],
    )
    disp.plot(ax=ax, cmap="Blues", colorbar=False, im_kw={"alpha": 0.88})
    _style_dark_axes(ax)

    ax.set_title(
        "Confusion Matrix",
        color=SOC_TEXT,
        fontsize=11,
        fontweight="bold",
        pad=10,
    )
    ax.set_xlabel("Predicted Label", color=SOC_MUTED, fontsize=9)
    ax.set_ylabel("True Label", color=SOC_MUTED, fontsize=9)

    for text in ax.texts:
        text.set_color(SOC_TEXT)
        text.set_fontsize(11)
        text.set_fontweight("bold")

    plt.tight_layout()
    return fig


def render_pr_curve_figure(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float,
) -> plt.Figure:
    precisions, recalls, pr_thresholds = precision_recall_curve(y_true, y_scores)

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    fig.patch.set_facecolor(SOC_CARD_BG)
    _style_dark_axes(ax)

    ax.plot(
        recalls,
        precisions,
        color=SOC_CYAN,
        linewidth=2.2,
        label="PR Curve",
    )
    ax.fill_between(recalls, precisions, alpha=0.08, color=SOC_CYAN)

    y_pred_at_t = predictions_from_threshold(y_scores, threshold)
    point_precision = precision_score(y_true, y_pred_at_t, zero_division=0)
    point_recall = recall_score(y_true, y_pred_at_t, zero_division=0)

    ax.scatter(
        [point_recall],
        [point_precision],
        s=180,
        c=SOC_RED,
        edgecolors="white",
        linewidths=1.5,
        zorder=5,
        label=f"Threshold α={threshold:.2f}",
    )

    ax.set_title(
        "Precision-Recall Curve",
        color=SOC_TEXT,
        fontsize=11,
        fontweight="bold",
        pad=10,
    )
    ax.set_xlabel("Recall", color=SOC_MUTED, fontsize=9)
    ax.set_ylabel("Precision", color=SOC_MUTED, fontsize=9)
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.0, 1.05)
    ax.legend(
        loc="lower left",
        facecolor=SOC_CARD_BG,
        edgecolor=SOC_BORDER,
        labelcolor=SOC_TEXT,
        fontsize=8,
    )
    ax.grid(True, alpha=0.15, color=SOC_MUTED)

    plt.tight_layout()
    return fig


def render_hero_header_idle(pipeline_label: str) -> None:
    st.markdown(
        f"""
        <div class="soc-hero">
            <h1>🛡️ Next-Gen SOC Operations Console (PoC)</h1>
            <p class="subtitle">
                Integrated Privacy-First Gateway · {html.escape(pipeline_label)} · Awaiting analysis run
            </p>
            <div class="badge-row">
                <span class="soc-badge">STAGE 3 XAI</span>
                <span class="soc-badge">STAGE 1 GATEWAY</span>
                <span class="soc-badge">PRIVACY IDLE</span>
                <span class="soc-badge">1 → 2 → 3</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero_header(stage2: dict, mitre_id: str, pipeline_label: str) -> None:
    privacy_badge = "ok" if stage2["final_check_status"] == "PASS" else "danger"
    privacy_label = f"STAGE 1 {stage2['final_check_status']}"
    st.markdown(
        f"""
        <div class="soc-hero">
            <h1>🛡️ Next-Gen SOC Operations Console (PoC)</h1>
            <p class="subtitle">
                Privacy-First Gateway · {html.escape(pipeline_label)} · MITRE {html.escape(mitre_id)}
            </p>
            <div class="badge-row">
                <span class="soc-badge">STAGE 3 XAI</span>
                <span class="soc-badge">STAGE 2 CLOUD</span>
                <span class="soc-badge {privacy_badge}">{privacy_label}</span>
                <span class="soc-badge">1 → 2 → 3</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# 4. Streamlit dashboard
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="Next-Gen SOC Operations Console",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_soc_dark_theme()

    pipeline_mode = render_pipeline_sidebar()
    email_input = render_main_email_console()

    if st.session_state.get("active_pipeline_mode") != pipeline_mode:
        st.session_state.pop("integrated_context", None)
        st.session_state.pop("pipeline_error", None)
        st.session_state.pop("run_pipeline", None)
        st.session_state.pop("demo_custom_context", None)
        _clear_benchmark_cache()
    st.session_state["active_pipeline_mode"] = pipeline_mode

    ctx, pipeline_notice = build_runtime_context(pipeline_mode, email_input)
    results_active = has_pipeline_results(pipeline_mode)

    stage1 = ctx["stage_1_mcp_output"]
    stage2 = ctx["stage_2_privacy_output"]
    pipeline_meta = ctx.get("pipeline_meta", {})

    mitre_id = extract_mitre_id(stage1["mitre_technique"]) if results_active else "—"
    if results_active:
        render_hero_header(stage2, mitre_id, pipeline_mode)
    else:
        render_hero_header_idle(pipeline_mode)

    if pipeline_notice:
        notice_kind = "warning" if st.session_state.get("pipeline_error") else "info"
        render_soc_notice(notice_kind, pipeline_notice)
    elif results_active and pipeline_mode == "Demo (Mock Data)":
        render_soc_notice(
            "success",
            "Privacy-First demo applied · Stage 1 live gateway redaction + "
            "mock Stage 2 cloud verdict on sanitized text.",
        )
    elif results_active and pipeline_mode != "Demo (Mock Data)":
        render_soc_notice(
            "success",
            "Privacy-First pipeline complete · "
            f"Stage 1: {pipeline_meta.get('member_b', 'n/a')} · "
            f"Stage 2: {pipeline_meta.get('member_a', 'n/a')}",
        )

    if results_active:
        stage_log = pipeline_meta.get("stage_log") or []
        render_pipeline_stage_timeline(stage_log)

    col_left, col_right = st.columns(2, gap="large")

    if results_active:
        raw_email = stage1["raw_email"]
        analysis_text = resolve_analysis_text(stage1)
        risk_score = float(stage1["risk_score"])
        auditor = XAIAuditor()
        xai_audit = run_xai_necessity_audit(stage1, analysis_text, auditor=auditor)
        highlight_tokens = xai_audit["tokens_in_text"] or xai_audit["high_risk_tokens"]
        necessity = xai_audit.get("necessity")

    with col_left:
        st.markdown(
            '<p style="color:#58a6ff;font-weight:700;font-size:1.05rem;">'
            "📡 Data Pipeline Monitor</p>",
            unsafe_allow_html=True,
        )

        if not results_active:
            render_card_header("Cards 1–2 · Awaiting Pipeline Run", "📨")
            render_soc_notice(
                "info",
                "Stage 1 local gateway redaction and Stage 2 cloud threat verdict "
                "will appear here after you run the Privacy-First pipeline.",
            )
            render_card_footer()
        else:
            render_card_header(
                "Card 1 · Stage 1 · Local Sanitization Gateway (Privacy-First On-Prem)",
                "📨",
            )
            st.markdown(
                '<p class="soc-label">inbound_raw_email (pre-gateway · contains PII)</p>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="soc-email-box">{html.escape(raw_email)}</div>',
                unsafe_allow_html=True,
            )
            if stage1.get("sender_domain"):
                st.markdown(
                    f'<p class="soc-context-line"><strong>Envelope domain (parsed for cloud escalation):</strong> '
                    f'{html.escape(stage1["sender_domain"])}</p>',
                    unsafe_allow_html=True,
                )
            if analysis_text != raw_email:
                st.markdown(
                    '<p class="soc-label" style="margin-top:0.75rem;">'
                    "sanitized_email_body (Stage 1 output · safe for cloud)</p>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="soc-email-box">{html.escape(analysis_text)}</div>',
                    unsafe_allow_html=True,
                )

            st.markdown(
                '<p class="soc-label" style="margin-top:0.75rem;">'
                "sanitized_email (Stage 1 Gateway · PII masked on-prem)</p>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="soc-masked-box">{html.escape(stage2["masked_email"])}</div>',
                unsafe_allow_html=True,
            )
            render_privacy_metadata(stage2)
            render_card_footer()

            render_card_header(
                "Card 2 · Stage 2 · Cloud Threat Intelligence (Sanitized-Text Gemini Verdict)",
                "🎯",
            )
            if stage2_used_heuristic_fallback(stage1):
                render_soc_notice("warning", stage2_gemini_unavailable_message(stage1))
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric(
                    label="Risk Score (0.0–1.0)",
                    value=format_risk_score(risk_score),
                    delta="MALICIOUS" if stage1["is_phishing"] else "BENIGN",
                    delta_color="inverse" if stage1["is_phishing"] else "off",
                )
            with m2:
                st.metric(
                    label="Detection Status",
                    value="PHISHING" if stage1["is_phishing"] else "BENIGN",
                )
            with m3:
                st.metric(label="MITRE ID (parsed)", value=mitre_id)

            st.markdown(
                f'<p class="soc-context-line"><strong>MITRE Technique (Stage 2 cloud judge):</strong> '
                f'{html.escape(stage1["mitre_technique"])}</p>'
                f'<p class="soc-context-line"><strong>Forensic Evidence (PII-free context):</strong> '
                f'{html.escape(stage1["why_malicious_context"])}</p>',
                unsafe_allow_html=True,
            )
            render_card_footer()

    with col_right:
        st.markdown(
            '<p style="color:#58a6ff;font-weight:700;font-size:1.05rem;">'
            "🔬 XAI Audit & Security Benchmarking</p>",
            unsafe_allow_html=True,
        )

        if not results_active:
            render_card_header("Card 3 · Stage 3 · XAI Forensic Audit", "⚡")
            render_soc_notice(
                "info",
                "Forensic token highlights and necessity perturbation "
                "will appear here after Stage 2 cloud threat intelligence completes.",
            )
            render_card_footer()
        else:
            render_card_header(
                "Card 3 · Stage 3 · Forensic Highlight & Necessity Perturbation",
                "⚡",
            )
            st.markdown(
                '<p class="soc-label">Threat Token Highlight (Alert Red)</p>',
                unsafe_allow_html=True,
            )
            highlighted = highlight_high_risk_tokens(analysis_text, highlight_tokens)
            st.markdown(
                f'<div class="soc-email-box">{highlighted}</div>',
                unsafe_allow_html=True,
            )
            token_summary = (
                ", ".join(xai_audit["high_risk_tokens"])
                if xai_audit["high_risk_tokens"]
                else "None (Stage 2 cloud judge returned no forensic tokens)"
            )
            st.markdown(
                f'<p class="soc-context-line"><strong>Perturbation tokens (Stage 2 cloud intel):</strong> '
                f'{html.escape(token_summary)}</p>',
                unsafe_allow_html=True,
            )
            if xai_audit["tokens_in_text"]:
                st.markdown(
                    f'<p class="soc-context-line"><strong>Tokens in email body:</strong> '
                    f'{html.escape(", ".join(xai_audit["tokens_in_text"]))}</p>',
                    unsafe_allow_html=True,
                )

            audit_status = xai_audit["status"]
            if audit_status.startswith("skipped_"):
                st.markdown(
                    '<p class="soc-label" style="margin-top:0.85rem;">'
                    "Necessity Test (Normalized 0.0–1.0)</p>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="soc-audit-skip">'
                    f"ℹ️ <strong>Audit Skipped</strong> — {html.escape(xai_audit['message'])}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            elif necessity is not None:
                st.markdown(
                    '<p class="soc-label" style="margin-top:0.85rem;">'
                    "Necessity Test (Normalized 0.0–1.0)</p>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<p class="soc-score-flow">'
                    f'<span class="from">{risk_score:.2f}</span>'
                    f'<span class="arrow">➔</span>'
                    f'<span class="to">{necessity["perturbed_score"]:.2f}</span>'
                    f'&nbsp;&nbsp;|&nbsp;&nbsp;Drop '
                    f'<span class="from">{necessity["score_drop"]:.2f}</span>'
                    f"</p>",
                    unsafe_allow_html=True,
                )

                n1, n2, n3 = st.columns(3)
                with n1:
                    st.metric("Original Score", f"{risk_score:.2f}")
                with n2:
                    st.metric("Perturbed Score", f"{necessity['perturbed_score']:.2f}")
                with n3:
                    st.metric(
                        "Score Drop",
                        f"{necessity['score_drop']:.2f}",
                        delta=f"{necessity['score_drop']:.2f}",
                        delta_color="normal" if necessity["score_drop"] >= 0 else "inverse",
                    )

                if audit_status == "passed":
                    st.markdown(
                        '<div class="soc-audit-pass">'
                        "✅ <strong>Audit Certified</strong> — Causal attribution confirmed "
                        f"(drop {necessity['score_drop']:.2f} &gt; threshold "
                        f"{XAIAuditor.TRUSTWORTHY_DROP_THRESHOLD:.2f}). "
                        f"{risk_score:.2f} ➔ {necessity['perturbed_score']:.2f} indicates "
                        "decision is not an LLM hallucination."
                        "</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<div class="soc-audit-fail">'
                        "❌ <strong>Audit Failed</strong> — Score drop did not exceed threshold "
                        f"({XAIAuditor.TRUSTWORTHY_DROP_THRESHOLD:.2f}). Manual SOC review required."
                        "</div>",
                        unsafe_allow_html=True,
                    )
                    adversarial_prompt = auditor.generate_adversarial_feedback(
                        failed_tokens=xai_audit["tokens_in_text"],
                        current_mitre=stage1["mitre_technique"],
                        raw_email=analysis_text,
                    )
                    render_adversarial_prompt_block(adversarial_prompt)

                with st.expander("Perturbed Email ([REMOVED] tokens)"):
                    st.markdown(
                        f'<div class="soc-perturbed">{html.escape(necessity["perturbed_text"])}</div>',
                        unsafe_allow_html=True,
                    )
            render_card_footer()

        render_card_header("Card 4 · Security Benchmarking Dashboard", "📊")

        if pipeline_mode != "Demo (Mock Data)":
            bm_run_col, bm_stop_col = st.columns(2)
            with bm_run_col:
                if st.button(
                    BENCHMARK_RUN_LABEL,
                    type="primary",
                    key="card4_run_benchmark",
                    width="stretch",
                    help="Privacy-First golden set: Stage 1 mask → Stage 2 Gemini per corpus email.",
                ):
                    _request_live_benchmark_rerun()
                    st.rerun()
            with bm_stop_col:
                if st.button(
                    "Stop Live Benchmark",
                    type="secondary",
                    key="card4_stop_benchmark",
                    width="stretch",
                    help="Abort in-flight benchmark and skip remaining Gemini calls.",
                ):
                    _stop_benchmark_jobs()
                    st.rerun()

        benchmark_df, benchmark_source = resolve_benchmark_dataframe(pipeline_mode)
        y_true = benchmark_df["label"].values
        y_scores = benchmark_df["predicted_score"].values.astype(float)

        if benchmark_source == "mock":
            benchmark_caption = (
                f"Corpus: {BENCHMARK_TOTAL} Pure English emails · "
                f"{BENCHMARK_BENIGN_COUNT} Benign / {BENCHMARK_PHISHING_COUNT} Phishing · "
                f"Demo mode — static mock predicted scores · Locale EN-US"
            )
        elif benchmark_source == "live_pending":
            benchmark_caption = (
                f"Corpus: {BENCHMARK_TOTAL} Pure English emails (Golden Balanced Set) · "
                f"{BENCHMARK_BENIGN_COUNT} Benign / {BENCHMARK_PHISHING_COUNT} Phishing · "
                f"Click **{BENCHMARK_RUN_LABEL}** to start Gemini scoring"
            )
        else:
            benchmark_caption = (
                f"Corpus: {BENCHMARK_TOTAL} Pure English emails · "
                f"{BENCHMARK_BENIGN_COUNT} Benign / {BENCHMARK_PHISHING_COUNT} Phishing · "
                f"Live Privacy-First batch ({pipeline_mode}) · Locale EN-US"
            )
        st.caption(benchmark_caption)

        if benchmark_source == "live_pending":
            render_soc_notice(
                "info",
                f"Live mode uses mock scores until you click **{BENCHMARK_RUN_LABEL}**. "
                f"Each Gemini call is paced by **{BENCHMARK_API_PACING_SECONDS:.0f}s** "
                f"(~{max(BENCHMARK_TOTAL - 1, 0) * int(BENCHMARK_API_PACING_SECONDS)}s cooldown + inference · "
                "Anti-429 tactical set). "
                "Use **Stop Live API Jobs** in the sidebar to cancel a stuck batch.",
            )
        elif st.session_state.get(BENCHMARK_ABORT_KEY):
            render_soc_notice(
                "warning",
                "Live benchmark was stopped. Remaining records use safe default score 0.0. "
                "Click **Run Live Benchmark** again after API quota recovers.",
            )

        threshold = st.slider(
            "Decision Threshold (Alpha)",
            min_value=0.0,
            max_value=1.0,
            value=0.87,
            step=0.01,
            help="Classify as Phishing (1) when predicted_score >= Alpha.",
            disabled=benchmark_source == "live_pending",
        )

        metrics_pending = benchmark_source == "live_pending"
        if metrics_pending:
            b1, b2, b3, b4 = st.columns(4)
            with b1:
                st.metric("Precision", "—", help="Run Live Benchmark to compute live metrics.")
            with b2:
                st.metric("Recall", "—", help="Run Live Benchmark to compute live metrics.")
            with b3:
                st.metric("F1-Score", "—", help="Run Live Benchmark to compute live metrics.")
            with b4:
                st.metric("Active Threshold α", f"{threshold:.2f}")
            st.markdown(
                '<p class="soc-context-line" style="opacity:0.85;">'
                "Confusion matrix and PR curve are hidden until live Gemini scoring completes — "
                "corpus placeholder scores are not real model output."
                "</p>",
                unsafe_allow_html=True,
            )
        else:
            y_pred = predictions_from_threshold(y_scores, threshold)
            metrics = compute_benchmark_metrics(y_true, y_pred)

            b1, b2, b3, b4 = st.columns(4)
            with b1:
                st.metric("Precision", f"{metrics['precision']:.2%}")
            with b2:
                st.metric("Recall", f"{metrics['recall']:.2%}")
            with b3:
                st.metric("F1-Score", f"{metrics['f1']:.2%}")
            with b4:
                st.metric("Active Threshold α", f"{threshold:.2f}")

            if benchmark_source == "mock":
                render_soc_notice(
                    "info",
                    "Demo mode — metrics use **static mock scores** from the corpus file, "
                    "not Gemini. They illustrate the dashboard layout only.",
                )

            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                fig_cm = render_confusion_matrix_figure(metrics["confusion_matrix"])
                st.pyplot(fig_cm, width="stretch")
                plt.close(fig_cm)
            with chart_col2:
                fig_pr = render_pr_curve_figure(y_true, y_scores, threshold)
                st.pyplot(fig_pr, width="stretch")
                plt.close(fig_pr)

        with st.expander(f"Full Benchmark Corpus ({BENCHMARK_TOTAL} records)"):
            display_df = benchmark_df.copy()
            if metrics_pending:
                display_df["predicted_score"] = "pending (live run)"
                display_df["predicted_label"] = "—"
            else:
                display_df["predicted_label"] = predictions_from_threshold(
                    display_df["predicted_score"].values, threshold
                )
            display_df = display_df[
                [
                    "id",
                    "subject",
                    "snippet",
                    "locale",
                    "label",
                    "predicted_score",
                    "predicted_label",
                ]
            ]
            display_df.columns = [
                "ID",
                "Subject",
                "Snippet",
                "Locale",
                "True Label (0/1)",
                "Predicted Score",
                f"Pred @ α={threshold:.2f}",
            ]
            st.dataframe(display_df, width="stretch", hide_index=True)
        render_card_footer()

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#58a6ff;font-weight:700;font-size:1.05rem;">'
        "👤 Human-in-the-Loop · Officer Override Console</p>",
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        render_card_header("Footer Card · Final Decision Override", "🔐")
        hitl_col1, hitl_col2 = st.columns([1, 2])
        with hitl_col1:
            officer_override = st.checkbox(
                "Security Officer Approval / Suppress Alert",
                value=False,
                key="officer_override",
            )
        with hitl_col2:
            if officer_override:
                render_soc_notice(
                    "success",
                    "Officer override applied — automated quarantine lifted. "
                    "Incident marked for post-review.",
                )
            else:
                render_soc_notice(
                    "info",
                    "Automated defense mode active · Stage 3 XAI audit and Stage 2 cloud "
                    "risk scoring (0.0–1.0) are enforcing policy. Check the box to apply "
                    "manual override.",
                )
        render_card_footer()


if __name__ == "__main__":
    main()
