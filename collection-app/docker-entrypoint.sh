#!/bin/sh
set -e

# With Litestream configured (LITESTREAM_REPLICA_URL set), restore the DB if
# the volume is empty, then run the server under continuous replication.
if [ -n "$LITESTREAM_REPLICA_URL" ]; then
  if [ ! -f "$DATABASE_PATH" ]; then
    litestream restore -if-replica-exists -o "$DATABASE_PATH" "$LITESTREAM_REPLICA_URL" || true
  fi
  exec litestream replicate -exec "npm start" "$DATABASE_PATH" "$LITESTREAM_REPLICA_URL"
else
  exec npm start
fi
