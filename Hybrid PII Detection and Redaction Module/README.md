# Member B — Hybrid PII Detection & Redaction

Regex-based structured PII masking + Ollama semantic PII detection.

## Run standalone

```bash
cd "Hybrid PII Detection and Redaction Module"
pip install requests python-dotenv
python app_test.py
```

## Batch integration (A → B)

```bash
cd "Hybrid PII Detection and Redaction Module"
python run_member_a_output_to_b.py
```

Requires `MCP Phishing and BEC Detection Module/batch_output.json`.

## Environment (project root `.env`)

```
OLLAMA_URL=http://localhost:11434/api/chat
OLLAMA_MODEL=llama3.2:3b
```
