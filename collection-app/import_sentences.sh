#!/usr/bin/env bash
#
# Import a JSONL sentence file ({"text": ..., "source": ...} per line) into
# the production collection app's sentence pool via the admin API. Duplicate
# texts are skipped server-side, so re-running is idempotent.
#
# Usage:
#   bash import_sentences.sh curated/haeyo-2026-08.jsonl
#   PRIORITY=1 bash import_sentences.sh curated/haeyo-2026-08.jsonl
#
# PRIORITY=N stamps every row (rows with their own "priority" keep it):
# priority > 0 sentences are assigned to workers before the coverage-driven
# pool. Re-posting an already-imported file with PRIORITY set promotes it.
set -euo pipefail

FILE="${1:?usage: import_sentences.sh <sentences.jsonl>}"

# Fill unset vars from the repo-root .env (plain KEY=value lines); explicitly
# set environment variables win. Skips the password prompt when present.
ENV_FILE="$(cd "$(dirname "$0")/.." && pwd)/.env"
if [[ -f "$ENV_FILE" ]]; then
  while IFS='=' read -r k v; do
    [[ "$k" =~ ^[A-Z_]+$ && -z "${!k+x}" ]] && export "$k=$v"
  done < "$ENV_FILE"
fi

BASE="${BASE:-https://handwriting-collection.fly.dev}"
ADMIN_USER="${ADMIN_USER:-jon@jonb.org}"

COOKIES="$(mktemp)"
trap 'rm -f "$COOKIES"' EXIT

if [[ -z "${ADMIN_PASSWORD:-}" ]]; then
  read -rsp "Admin password for ${ADMIN_USER}: " ADMIN_PASSWORD
  echo
fi

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

python3 -c '
import json, os, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
p = os.environ.get("PRIORITY")
if p is not None:
    for r in rows:
        r.setdefault("priority", int(p))
print(json.dumps({"sentences": rows}))
' "$FILE" | curl -sS -b "$COOKIES" \
  -H 'Content-Type: application/json' \
  -X POST "$BASE/api/admin/sentences/import" \
  --data @- | python3 -m json.tool
