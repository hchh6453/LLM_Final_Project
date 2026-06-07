# Phishing & BEC Detection and Defense System

**智慧型釣魚與商務電子郵件詐騙 (BEC) 偵測與防護系統**

Final Project · Large Language Models Course · NTUST

Integrated three-module **Privacy-First Gateway** pipeline with an English-first Next-Gen SOC Operations Console (PoC).

---

## Module Overview

| Folder | Member | Responsibility |
|--------|--------|----------------|
| [`MCP Phishing and BEC Detection Module/`](MCP%20Phishing%20and%20BEC%20Detection%20Module/) | A | MCP email parsing · Dual-layer Agentic RAG · Gemini LLM judge |
| [`Hybrid PII Detection and Redaction Module/`](Hybrid%20PII%20Detection%20and%20Redaction%20Module/) | B | Hybrid PII detection & redaction (Regex + Ollama semantic masking) |
| [`XAI Audit Pipeline and Security Benchmarking Dashboard Module/`](XAI%20Audit%20Pipeline%20and%20Security%20Benchmarking%20Dashboard%20Module/) | C | Streamlit SOC dashboard · XAI necessity audit · ML security benchmarking |

All three folders follow the same naming convention: **`[Descriptive English Name] Module`**.

---

## Privacy-First Pipeline (Runtime Order)

Raw email is processed **on-prem first**, then escalated to the cloud only after PII is masked.

```
raw_email
    │
    ▼
Stage 1 · Local Sanitization Gateway          (Member B)
Hybrid PII Detection and Redaction Module     Regex + Ollama · PII never leaves intranet
    │  stage_2_privacy_output (legacy schema key)
    ▼
Stage 2 · Cloud Threat Intelligence           (Member A)
MCP Phishing and BEC Detection Module         MCP parse → Agentic RAG → Gemini on sanitized text
    │  stage_1_mcp_output (legacy schema key)
    ▼
Stage 3 · XAI Forensic Audit                  (Member C)
XAI Audit Pipeline and Security Benchmarking Dashboard Module
    │  Token highlight · Necessity perturbation · Benchmark dashboard · HITL override
    ▼
SOC analyst decision
```

> **Note:** JSON schema keys `stage_1_mcp_output` / `stage_2_privacy_output` retain legacy names from early integration. In the SOC console narrative, **Stage 1 = privacy (B)** and **Stage 2 = cloud verdict (A)**.

---

## Project Structure

```
Final_project/
├── MCP Phishing and BEC Detection Module/       # Member A
│   ├── main.py · mcp_parser.py · rag_retriever.py · llm_judge.py
│   ├── chroma_db/ · data/
│   └── output_for_member.json · batch_output.json
├── Hybrid PII Detection and Redaction Module/   # Member B
│   ├── pii_redactor.py · app_test.py
│   └── run_member_a_output_to_b.py
├── XAI Audit Pipeline and Security Benchmarking Dashboard Module/  # Member C
│   ├── app.py                    # Streamlit SOC console
│   ├── pipeline_integration.py   # B → A → C orchestration
│   └── benchmark_corpus.py       # 40 EN-US benchmark emails
├── .env · .env.example
├── requirements.txt
├── run_dashboard.sh
├── .streamlit/config.toml
└── .venv/
```

---

## Quick Start

```bash
cd Final_project
python3 -m venv .venv          # skip if .venv already exists
source .venv/bin/activate
pip install -r requirements.txt
./run_dashboard.sh
```

Or manually:

```bash
streamlit run "XAI Audit Pipeline and Security Benchmarking Dashboard Module/app.py"
```

Open **http://localhost:8501**

---

## Dashboard Modes (Sidebar)

| Mode | Description |
|------|-------------|
| **Demo (Mock Data)** | Live Stage 1 PII redaction (Member B) + **mock** Stage 2 cloud scores. No Gemini calls for single-email pipeline. |
| **Live Pipeline (Stage 1 → 2 → 3)** | Full Privacy-First pipeline: on-prem mask → Gemini verdict on sanitized text → XAI audit. |
| **Cached Cloud Verdict JSON + Live Stage 1** | Live Stage 1 redaction + cached Stage 2 verdict from `MCP Phishing and BEC Detection Module/output_for_member.json`. |

Enter email in the **Email Analysis Console** on the main page, then click **Run Integrated Pipeline**.

### Sidebar Controls

| Button | Action |
|--------|--------|
| **Clear Results** | Reset pipeline output and Card 4 benchmark cache |
| **Stop Live API Jobs** | Abort in-flight Card 4 batch / Gemini loops |

---

## SOC Console Cards

| Card | Stage | Content |
|------|-------|---------|
| **Card 1** | Stage 1 | Inbound raw email · sanitized output · PII metadata · privacy PASS/FAIL |
| **Card 2** | Stage 2 | Risk score (0.0–1.0) · PHISHING/BENIGN · MITRE ID · forensic context |
| **Card 3** | Stage 3 | Threat token highlight · necessity perturbation · adversarial prompt (on audit fail) |
| **Card 4** | Benchmark | 40 EN-US corpus · Precision / Recall / F1 · confusion matrix · PR curve |
| **Footer** | HITL | Security officer override / alert suppression |

---

## Card 4 · Security Benchmarking

- **Corpus:** 40 pure English emails (20 Benign / 20 Phishing), locale `EN-US`
- **Live batch:** Click **Run Live Benchmark (40 EN-US)** — runs Stage 1 mask → Stage 2 Gemini per email
- **API pacing:** 4 s interval between Gemini calls (free-tier RPM defense; ~3–5 min total)
- **Cache:** Scores are cached via `@st.cache_data`; adjusting the α threshold does **not** re-call Gemini
- **Clear cache:** Use sidebar **Clear Results** or re-click **Run Live Benchmark**

Demo mode uses static mock scores for layout preview only.

---

## Environment (`.env` at project root)

```env
# Member A — Gemini (required for Live Pipeline & Card 4 batch)
GEMINI_API_KEY=your_gemini_api_key_here

# Member B — Ollama (optional; Regex-only fallback if unavailable)
OLLAMA_URL=http://localhost:11434/api/chat
OLLAMA_MODEL=llama3.2:3b
```

Copy from template: `cp .env.example .env`

---

## Prerequisites

### Member A — MCP Phishing and BEC Detection Module

```bash
cd "MCP Phishing and BEC Detection Module"
python main.py
```

- Set `GEMINI_API_KEY` in `.env`
- First run downloads embedding models and builds ChromaDB under `chroma_db/` (may take several minutes)

### Member B — Hybrid PII Detection and Redaction Module

```bash
ollama pull llama3.2:3b
ollama serve
```

```bash
cd "Hybrid PII Detection and Redaction Module"
python app_test.py
```

### Member C — Dashboard

Handled by root `requirements.txt` (streamlit, sklearn, matplotlib, pandas, etc.).

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `streamlit: command not found` | Activate `.venv` first |
| `member_c/app.py` not found | Folder was renamed — use `./run_dashboard.sh` or the full Module C path |
| Member A import error | `pip install -r requirements.txt` from project root |
| Gemini `429` / rate limit | Wait for quota reset; Card 4 uses 4 s pacing; use **Stop Live API Jobs** if stuck |
| Stage 2 shows BENIGN but evidence says `rate_limit_exceeded` | Gemini API failed — not a real verdict; retry after quota recovers |
| Ollama timeout | Start `ollama serve`; Regex-only Stage 1 fallback still works |
| `No module named 'torchvision'` | Harmless Streamlit/transformers noise; suppressed via `.streamlit/config.toml` |
| Card 4 shows `—` for metrics | Live benchmark not run yet — click **Run Live Benchmark (40 EN-US)** |
| Conda `anaconda-auth` warning | Unrelated; use project `.venv` only |

---

## Academic Use

Course final project PoC for demonstration and evaluation. Not intended for production deployment.
