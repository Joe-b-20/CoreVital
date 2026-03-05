#!/usr/bin/env bash
# Push ~/experiment to S3 (bucket/region from setup: corevital-validation, us-east-1).
# Only excludes .pyc; syncs data, metadata, results, analysis, traces, logs, etc.

set -euo pipefail
BUCKET="${AWS_BUCKET:-corevital-validation}"
EXPERIMENT_DIR="${HOME:-/root}/experiment"
echo "Syncing $EXPERIMENT_DIR to s3://${BUCKET}/experiment ..."
aws s3 sync "$EXPERIMENT_DIR" "s3://${BUCKET}/experiment" \
  --exclude "*.pyc" --exclude "*__pycache__*" \
  --no-progress \
  "$@"
echo "Done."
