# Member C — XAI Audit Pipeline & Security Benchmarking Dashboard

Streamlit SOC console integrating Member A + B outputs with XAI auditing and a 14-record EN-US golden benchmark set.

## Run dashboard

From project root:

```bash
source .venv/bin/activate
streamlit run "XAI Audit Pipeline and Security Benchmarking Dashboard Module/app.py"
```

Or:

```bash
./run_dashboard.sh
```

## Files

| File | Role |
|------|------|
| `app.py` | Streamlit UI + XAI + benchmarking |
| `pipeline_integration.py` | Privacy-First B → A → C orchestration |
| `benchmark_corpus.py` | 14-record golden balanced test set (7/7) |
