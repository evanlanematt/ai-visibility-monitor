#!/usr/bin/env bash
# Runs the weekly check and rebuilds the report. Point cron or launchd at this file.
# Example crontab line (every Monday at 7:40 AM):
#   40 7 * * 1 /path/to/ai_visibility_monitor/run_weekly.sh >> /path/to/ai_visibility_monitor/monitor.log 2>&1
set -euo pipefail
cd "$(dirname "$0")"
# Load API keys from a local .env file if present (never commit it).
if [ -f .env ]; then set -a; . ./.env; set +a; fi
python3 monitor.py run
