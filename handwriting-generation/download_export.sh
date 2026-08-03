#!/usr/bin/env bash
#
# Download all captured handwriting recordings from the production collection
# app (your-collection-app.fly.dev) as a training-ready export.
#
# The export is a zip of recordings/u<user>/s<sentence>/rec-<id>.json files,
# each byte-compatible with the training pipeline AND carrying the transcript
# in metadata.text -- so no dependence on folder names for the text.
#
# Usage:
#   bash download_export.sh                 # prompts for password securely
#   ADMIN_PASSWORD=... bash download_export.sh   # non-interactive (avoid: lands in shell history)
#
# Re-run any time to pull the latest; it overwrites the previous export dir.
set -euo pipefail

BASE="${BASE:-https://your-collection-app.fly.dev}"
ADMIN_USER="${ADMIN_USER:-admin@example.com}"
OUT_DIR="${OUT_DIR:-$(cd "$(dirname "$0")/.." && pwd)/data/export}"

COOKIES="$(mktemp)"
trap 'rm -f "$COOKIES"' EXIT

# --- password: env var if set, else secure prompt (no echo, no history) ---
if [[ -z "${ADMIN_PASSWORD:-}" ]]; then
  read -rsp "Admin password for ${ADMIN_USER}: " ADMIN_PASSWORD
  echo
fi

# --- log in, capture session cookie ---
login_body=$(printf '{"username":%s,"password":%s}' \
  "$(printf '%s' "$ADMIN_USER" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')" \
  "$(printf '%s' "$ADMIN_PASSWORD" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')")

login_code=$(curl -sS -o /dev/null -w '%{http_code}' \
  -c "$COOKIES" \
  -H 'Content-Type: application/json' \
  -X POST "$BASE/api/login" \
  --data "$login_body")

if [[ "$login_code" != "200" ]]; then
  echo "Login failed (HTTP $login_code). Check the admin password." >&2
  exit 1
fi
echo "Logged in as $ADMIN_USER."

# --- sanity check: how many recordings will we get? ---
count=$(curl -sS -b "$COOKIES" "$BASE/api/admin/export/manifest" \
  | python3 -c 'import json,sys; print(len(json.load(sys.stdin)["recordings"]))')
echo "Recordings available for export: $count"

# --- download the zip ---
mkdir -p "$OUT_DIR"
zip_path="$OUT_DIR/handwriting-export.zip"
curl -sS -b "$COOKIES" "$BASE/api/admin/export" -o "$zip_path"
echo "Downloaded: $zip_path ($(du -h "$zip_path" | cut -f1))"

# --- unpack ---
unzip -oq "$zip_path" -d "$OUT_DIR"
rec_files=$(find "$OUT_DIR/recordings" -name '*.json' 2>/dev/null | wc -l | tr -d ' ')
echo "Unpacked $rec_files recording files into $OUT_DIR"
echo "  manifest:  $OUT_DIR/manifest.json"
echo "  sentences: $OUT_DIR/sentences.jsonl"
