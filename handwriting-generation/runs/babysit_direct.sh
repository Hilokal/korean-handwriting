#!/usr/bin/env bash
#
# Variant of babysit_pod_training.sh for when `runpodctl pod get` doesn't
# report ssh info (observed 2026-08-15: pods boot fine but the CLI shows
# "pod not ready"/uptime 0 forever). Takes the connection info directly.
#
# Usage:
#   bash babysit_direct.sh <pod-id> <ip> <port> <ssh-key> [remote-run-dir] [local-out-dir]
set -euo pipefail

POD_ID="${1:?pod id}"
IP="${2:?ip}"
PORT="${3:?port}"
KEY="${4:?ssh key path}"
REMOTE_DIR="${5:-/workspace/korean-handwriting/handwriting-generation}"
OUT_DIR="${6:-runs/$(date -u +%Y-%m-%d)-synth-retrain}"
POLL="${POLL:-120}"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
say() { echo "[$(ts)] $*"; }
SSH_OPTS=(-i "$KEY" -p "$PORT" -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new)
remote() { ssh "${SSH_OPTS[@]}" "root@$IP" "$@"; }

say "watching pod $POD_ID at $IP:$PORT, dir $REMOTE_DIR, every ${POLL}s"
say "artifacts will land in $OUT_DIR"

while true; do
  status=$(remote "cd '$REMOTE_DIR' 2>/dev/null || { echo BADDIR; exit 0; }
    log=\$(ls -t train*.log 2>/dev/null | head -1)
    if [ -n \"\$log\" ] && grep -q 'Training complete' \"\$log\"; then echo \"DONE \$log\"
    elif pgrep -f 'python.*train\.py' >/dev/null 2>&1; then echo \"RUNNING \$log\"
    else echo \"CRASHED \$log\"; fi" 2>/dev/null) || status="SSH_FAIL"

  case "$status" in
    DONE*|CRASHED*) break ;;
    RUNNING*) ;;
    BADDIR) say "remote dir $REMOTE_DIR missing -- wrong pod or wrong path?"; exit 1 ;;
    *) say "ssh unreachable, retrying in ${POLL}s" ;;
  esac
  sleep "$POLL"
done

state=${status%% *}
remote_log=${status#* }
say "run ended: $state (log: $remote_log)"

mkdir -p "$OUT_DIR"
pull_ok=true
for f in best_model.pt checkpoint.pt "$remote_log"; do
  # scp -O forces the legacy protocol (macOS scp defaults to SFTP, which the
  # RunPod image's sshd doesn't serve -- observed 2026-08-16: every pull
  # failed while ssh itself worked); fall back to ssh-cat either way.
  if scp -O "${SSH_OPTS[@]}" "root@$IP:$REMOTE_DIR/$f" "$OUT_DIR/" 2>/dev/null \
     || ssh "${SSH_OPTS[@]}" "root@$IP" "cat '$REMOTE_DIR/$f'" > "$OUT_DIR/$(basename "$f")" 2>/dev/null; then
    say "pulled $f ($(du -h "$OUT_DIR/$(basename "$f")" | cut -f1))"
  else
    say "FAILED to pull $f"
    pull_ok=false
  fi
done

if [[ "$pull_ok" != true || ! -s "$OUT_DIR/best_model.pt" ]]; then
  say "artifact pull incomplete -- LEAVING POD $POD_ID RUNNING (still billing!)"
  exit 1
fi

if [[ -n "${KEEP_POD:-}" ]]; then
  say "KEEP_POD set -- leaving pod $POD_ID running for the next run"
else
# runpodctl talks to the GraphQL API, which has 403'd this key before; fall back
# to the REST v2 endpoint (same key, Bearer auth) so the pod never outlives the run.
if runpodctl pod delete "$POD_ID" 2>/dev/null; then
  say "pod $POD_ID deleted (runpodctl)"
else
  key=$(grep -E '^apikey' ~/.runpod/config.toml | sed -E "s/.*= *['\"]?([A-Za-z0-9_]+)['\"]?.*/\1/")
  code=$(curl -sS -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $key" -X DELETE "https://api.runpod.io/v2/pods/$POD_ID")
  say "pod $POD_ID delete via REST v2: HTTP $code"
fi
fi

echo
say "=== final log tail ==="
tail -6 "$OUT_DIR/$(basename "$remote_log")"
say "state: $state | artifacts in $OUT_DIR"
if [[ "$state" == "CRASHED" ]]; then
  say "run CRASHED before completing -- checkpoint.pt can resume it on a fresh pod"
fi

if command -v osascript >/dev/null 2>&1; then
  osascript -e "display notification \"$state -- artifacts in $OUT_DIR, pod deleted\" with title \"Training run finished\"" || true
fi
