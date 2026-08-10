#!/usr/bin/env bash
#
# Watch a training run on a RunPod pod; when it finishes, pull the artifacts
# and delete the pod (so it stops billing). Designed to run unattended
# overnight on the local machine.
#
# Usage:
#   bash babysit_pod_training.sh <pod-id> [remote-run-dir] [local-out-dir]
#
#   remote-run-dir  default /workspace/korean-handwriting/handwriting-generation
#   local-out-dir   default runs/<UTC date>-<pod name>/   (never clobbers
#                   best_model.pt in the repo -- promote by hand after eyeballing
#                   renders)
#
# Detached overnight run (survives the terminal; caffeinate stops idle sleep --
# note a closed lid still sleeps the Mac, which just PAUSES the watch until
# wake, delaying the pull + pod deletion):
#   nohup caffeinate -i bash babysit_pod_training.sh <pod-id> > runs/babysit.log 2>&1 &
#
# Completion is detected from the newest train*.log in the run dir:
#   "Training complete"  -> DONE    (early stopping also prints this at the end)
#   train.py not running without the marker -> CRASHED
# In BOTH cases it pulls best_model.pt, checkpoint.pt (full resumable state),
# and the log. The pod is deleted only if the pull verifiably succeeded
# (best_model.pt exists locally and is non-empty); a crashed run's pod is also
# deleted -- checkpoint.pt can resume it on a fresh pod -- but the state is
# clearly reported. If the pull fails, the pod is LEFT RUNNING and this exits 1.
#
# Env knobs: POLL (seconds between checks, default 120).
set -euo pipefail

POD_ID="${1:?usage: babysit_pod_training.sh <pod-id> [remote-run-dir] [local-out-dir]}"
REMOTE_DIR="${2:-/workspace/korean-handwriting/handwriting-generation}"
POLL="${POLL:-120}"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
say() { echo "[$(ts)] $*"; }

# --- resolve connection info from runpodctl (key path, ip, port, name) ---
info=$(runpodctl pod get "$POD_ID")
read -r IP PORT KEY NAME < <(python3 -c '
import json, sys
p = json.load(sys.stdin)
print(p["ssh"]["ip"], p["ssh"]["port"], p["ssh"]["ssh_key"]["path"], p["name"])
' <<<"$info")
OUT_DIR="${3:-runs/$(date -u +%Y-%m-%d)-${NAME}}"
SSH_OPTS=(-i "$KEY" -p "$PORT" -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new)
remote() { ssh "${SSH_OPTS[@]}" "root@$IP" "$@"; }

say "watching pod $POD_ID ($NAME) at $IP:$PORT, dir $REMOTE_DIR, every ${POLL}s"
say "artifacts will land in $OUT_DIR"

# --- wait for the run to end (transient ssh failures just retry) ---
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

# --- pull artifacts ---
mkdir -p "$OUT_DIR"
pull_ok=true
for f in best_model.pt checkpoint.pt "$remote_log"; do
  if scp "${SSH_OPTS[@]}" "root@$IP:$REMOTE_DIR/$f" "$OUT_DIR/" 2>/dev/null; then
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

# --- pod is safe to delete: everything needed (incl. resumable state) is local ---
runpodctl pod delete "$POD_ID"
say "pod $POD_ID deleted"

echo
say "=== final log tail ==="
tail -6 "$OUT_DIR/$(basename "$remote_log")"
say "state: $state | artifacts in $OUT_DIR"
if [[ "$state" == "CRASHED" ]]; then
  say "run CRASHED before completing -- checkpoint.pt can resume it on a fresh pod"
fi

# best-effort desktop notification (macOS)
if command -v osascript >/dev/null 2>&1; then
  osascript -e "display notification \"$state -- artifacts in $OUT_DIR, pod deleted\" with title \"Training run finished\"" || true
fi
