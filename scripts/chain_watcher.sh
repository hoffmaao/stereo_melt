#!/bin/bash
# Disconnect-safe chain watcher.
#
# Usage:
#   chain_watcher.sh <watch_pid> <output_marker_file> <followup_pgrep_pattern> <followup_log> <followup_cmd>
#
# - Ignores SIGHUP (won't die when ssh disconnects).
# - Waits for watch_pid to exit.
# - Gives the original && chain 30 s to fire (so we don't double-launch).
# - If marker_file exists AND no process matches followup_pgrep_pattern, runs followup_cmd.
#
# followup_cmd is run via `bash -c "$followup_cmd"`, so quote any args inside it.
trap '' HUP
trap '' INT
set -u

WATCH_PID="$1"
MARKER_FILE="$2"
FOLLOWUP_PATTERN="$3"
FOLLOWUP_LOG="$4"
FOLLOWUP_CMD="$5"

echo "$(date -Is) watcher starting: waiting for PID $WATCH_PID" >> "$FOLLOWUP_LOG.watcher"
echo "$(date -Is)   followup_pattern=$FOLLOWUP_PATTERN" >> "$FOLLOWUP_LOG.watcher"
echo "$(date -Is)   followup_cmd=$FOLLOWUP_CMD" >> "$FOLLOWUP_LOG.watcher"

# Wait for the watched process to exit
while kill -0 "$WATCH_PID" 2>/dev/null; do
    sleep 30
done

echo "$(date -Is) PID $WATCH_PID exited" >> "$FOLLOWUP_LOG.watcher"

# Give the original && chain 30 s to fire (so we don't double-launch)
sleep 30

# Don't launch if marker file is missing (build failed)
if [ ! -f "$MARKER_FILE" ]; then
    echo "$(date -Is) marker file $MARKER_FILE missing — bailing out" >> "$FOLLOWUP_LOG.watcher"
    exit 1
fi

# Don't launch if the followup is already running (the && chain handled it)
if pgrep -f "$FOLLOWUP_PATTERN" > /dev/null; then
    echo "$(date -Is) followup '$FOLLOWUP_PATTERN' already running — exiting" >> "$FOLLOWUP_LOG.watcher"
    exit 0
fi

echo "$(date -Is) launching followup: $FOLLOWUP_CMD" >> "$FOLLOWUP_LOG.watcher"
exec setsid bash -c "$FOLLOWUP_CMD" > "$FOLLOWUP_LOG" 2>&1
