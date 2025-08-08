#!/usr/bin/env bash
set -euo pipefail
export HOST="0.0.0.0"
export PORT="8000"
python train_model.py
exec gunicorn -w 2 -b ${HOST}:${PORT} app:app