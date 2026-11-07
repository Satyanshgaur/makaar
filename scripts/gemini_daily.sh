#!/usr/bin/env bash
# One day's Gemini Flash gate run (free tier: 20 requests/day). The runner keeps finished pages in
# eval/preds/<name>/ and stops cleanly at the quota, so running this once a day completes the sample.
# Example crontab line (14:00 local, after the quota resets at midnight Pacific):
#   0 14 * * * cd ~/makaar && scripts/gemini_daily.sh >> ~/.cache/makaar-gemini.log 2>&1
set -euo pipefail
cd "$(dirname "$0")/.."
[ -d datasets/gate ] || (cd eval && uv run python scripts/make_gate_sample.py ../datasets/gate ../datasets/eval-omni:6 ../datasets/eval-hw:6)
NAME=gemini-gate scripts/gemini_baseline.sh "${GEMINI_MODEL:-gemini-3.5-flash}" ../datasets/gate
