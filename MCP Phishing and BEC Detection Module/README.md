# Member A — MCP Phishing & BEC Detection

Email intent classification with MCP parsing, dual-layer RAG, and Gemini LLM judge.

## Run standalone

```bash
cd "MCP Phishing and BEC Detection Module"
pip install google-genai python-dotenv chromadb sentence-transformers
python main.py
```

## Environment

Copy project root `.env` or create `MCP Phishing and BEC Detection Module/.env`:

```
GEMINI_API_KEY=your_key_here
```

## Key outputs

- `output_for_member.json` — single-email result for Member B/C
- `batch_output.json` — batch evaluation output
