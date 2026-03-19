#!/usr/bin/env bash
# Local fallback for the morning alert.
# Runs at 09:28 Shanghai via launchd, sends Telegram if GitHub Actions
# didn't already fire the morning check for today's watchlist.
#
# Skips sending if GitHub Actions already delivered (checks marker file).

set -euo pipefail

PROJECT_DIR="/Users/rayhuang/Documents/Python Project/dragon-pulse"
PYTHON="/opt/anaconda3/bin/python"
LOG_DIR="${PROJECT_DIR}/outputs/local_logs"
LOG_FILE="${LOG_DIR}/morning_alert_$(date +%Y-%m-%d).log"

mkdir -p "${LOG_DIR}"
exec >> "${LOG_FILE}" 2>&1

echo "=== Local morning alert: $(date) ==="

cd "${PROJECT_DIR}"

# Pull latest to get nightly outputs
git pull origin main --quiet 2>/dev/null || echo "WARN: git pull failed, using local state"

# Load .env
set -a
source "${PROJECT_DIR}/.env"
set +a

# Find the latest watchlist by filename date
LATEST_WL=$(ls outputs/*/execution_watchlist_*.json 2>/dev/null | sort -t_ -k3 -r | head -1)
if [ -z "${LATEST_WL}" ]; then
    echo "No watchlist found. Nothing to do."
    exit 0
fi

TRADE_DATE=$(basename "${LATEST_WL}" | sed 's/execution_watchlist_//;s/\.json//')
echo "Latest watchlist: ${LATEST_WL} (trade date: ${TRADE_DATE})"

# Check if GitHub Actions already sent today's alert (marker file from CI)
# The marker pattern is .telegram_sent_<run_id>_<attempt>.txt
MARKER_COUNT=$(find "outputs/${TRADE_DATE}" -name ".telegram_sent_*.txt" 2>/dev/null | wc -l | tr -d ' ')
if [ "${MARKER_COUNT}" -gt 0 ]; then
    echo "GitHub Actions already sent alert for ${TRADE_DATE}. Skipping."
    exit 0
fi

echo "No CI marker found — running morning check locally."
${PYTHON} scripts/morning_check.py --date "${TRADE_DATE}"
echo "=== Done: $(date) ==="
