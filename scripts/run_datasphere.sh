#!/usr/bin/env bash
set -e
CONFIG=${1:-configs/local.yaml}
echo "Using config: $CONFIG"
python -u src/train.py --config "$CONFIG" "$@"
