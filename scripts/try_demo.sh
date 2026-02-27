#!/usr/bin/env bash
# Start the CoreVital local API for "Try without installing" (Codespaces / devcontainer).
# Open the hosted React dashboard (https://main.d2maxwaq575qed.amplifyapp.com) and click Connect
# to list and load traces from the local DB. In Codespaces, port 8000 is forwarded automatically.
set -e
cd "$(dirname "$0")/.."
pip install -e ".[serve]" -q
echo "Starting CoreVital API at http://localhost:8000 — open the dashboard and click Connect to use it."
exec corevital serve
