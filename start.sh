#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

# Load .env.local if present
if [ -f .env.local ]; then
  export $(grep -v '^#' .env.local | xargs)
fi

python3 -m uvicorn main:app --host 0.0.0.0 --port 8080 --reload
