#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
export STREAMLIT_SERVER_FILE_WATCHER_TYPE=none
streamlit run "XAI Audit Pipeline and Security Benchmarking Dashboard Module/app.py"
