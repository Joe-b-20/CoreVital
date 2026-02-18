#!/usr/bin/env bash
# Start the CoreVital dashboard for "Try without installing" (Codespaces / devcontainer).
# In Codespaces, the port 8501 is forwarded automatically; open the forwarded URL in the browser.
set -e
cd "$(dirname "$0")/.."
pip install -e ".[dashboard]" -q
echo "Starting dashboard at http://localhost:8501 (in Codespaces, use the forwarded URL for port 8501)"
exec streamlit run dashboard.py --server.headless true --server.port 8501
