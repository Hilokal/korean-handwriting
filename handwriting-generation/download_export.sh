#!/usr/bin/env bash
#
# Download all captured handwriting recordings from the production collection
# app (handwriting-collection.fly.dev) as a training-ready export.
#
# The export is a zip of recordings/u<user>/s<sentence>/rec-<id>.json files,
# each byte-compatible with the training pipeline AND carrying the transcript
# in metadata.text -- so no dependence on folder names for the text.
#
# Usage:
#   bash download_export.sh                 # creds/filter from the repo-root .env
#                                           # (ADMIN_USER/ADMIN_PASSWORD/USER_ID);
#                                           # prompts for the password if absent
#   LIST_ONLY=1 bash download_export.sh     # just print per-user counts, no download
#   USER_ID=2 bash download_export.sh       # only this worker's recordings
#   USER_ID= bash download_export.sh        # every worker (overrides the .env filter)
#
# Re-run any time to pull the latest. The recordings/ dir is cleared before
# unpacking so the export dir mirrors exactly what was pulled -- important
# with USER_ID, or a previous unfiltered pull's other-worker files would
# linger and ExportDataset (which globs the dir) would train on them anyway.
set -euo pipefail

# Fill unset vars from the repo-root .env (plain KEY=value lines: ADMIN_USER,
# ADMIN_PASSWORD, USER_ID, ...). Explicitly-set environment variables win, so
# e.g. `USER_ID= bash download_export.sh` still pulls every worker.
ENV_FILE="$(cd "$(dirname "$0")/.." && pwd)/.env"
if [[ -f "$ENV_FILE" ]]; then
  while IFS='=' read -r k v; do
    [[ "$k" =~ ^[A-Z_]+$ && -z "${!k+x}" ]] && export "$k=$v"
  done < "$ENV_FILE"
fi

BASE="${BASE:-https://handwriting-collection.fly.dev}"
ADMIN_USER="${ADMIN_USER:-jon@jonb.org}"
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

# --- per-user breakdown (manifest is cheap: metadata only, no dots) ---
curl -sS -b "$COOKIES" "$BASE/api/admin/export/manifest" | python3 -c '
import json, sys
from collections import Counter
recs = json.load(sys.stdin)["recordings"]
print(f"Recordings available for export: {len(recs)}")
for (uid, name), n in sorted(Counter((r["userId"], r["userName"]) for r in recs).items()):
    print(f"  userId {uid}: {name} ({n} recordings)")
'

if [[ -n "${LIST_ONLY:-}" ]]; then
  echo "LIST_ONLY set; not downloading. Re-run with USER_ID=<id> to pull one worker."
  exit 0
fi

# --- optional per-worker filter (the export endpoint supports ?userId=) ---
QS=""
if [[ -n "${USER_ID:-}" ]]; then
  QS="?userId=${USER_ID}"
  count=$(curl -sS -b "$COOKIES" "$BASE/api/admin/export/manifest$QS" \
    | python3 -c 'import json,sys; print(len(json.load(sys.stdin)["recordings"]))')
  echo "Filtering to userId ${USER_ID}: $count recordings"
  if [[ "$count" == "0" ]]; then
    echo "No recordings for userId ${USER_ID}; aborting before touching $OUT_DIR." >&2
    exit 1
  fi
fi

# --- download the zip ---
mkdir -p "$OUT_DIR"
zip_path="$OUT_DIR/handwriting-export.zip"
curl -sS -b "$COOKIES" "$BASE/api/admin/export$QS" -o "$zip_path"
echo "Downloaded: $zip_path ($(du -h "$zip_path" | cut -f1))"

# --- unpack (clear stale recordings first so the dir mirrors this pull) ---
rm -rf "$OUT_DIR/recordings"
unzip -oq "$zip_path" -d "$OUT_DIR"
rec_files=$(find "$OUT_DIR/recordings" -name '*.json' 2>/dev/null | wc -l | tr -d ' ')
echo "Unpacked $rec_files recording files into $OUT_DIR"
echo "  manifest:  $OUT_DIR/manifest.json"
echo "  sentences: $OUT_DIR/sentences.jsonl"
