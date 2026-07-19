#!/bin/sh
# Start the API. The app runs unprivileged as uid 1000 (see the Dockerfile), but a mounted
# volume (TELEMETRY_DIR, e.g. Railway's /data) is attached root-owned, so a non-root process
# can't write its telemetry there. When the container starts as root we hand that directory
# to the app user and then drop to uid 1000 via gosu; on a host that forces a non-root uid
# (no root, and no volume anyway) we skip straight to running the server. Either way uvicorn
# is exec'd so it stays PID 1 and takes SIGTERM directly.
set -e

DIR="${TELEMETRY_DIR:-/data}"
if [ "$(id -u)" = "0" ]; then
    if [ -d "$DIR" ]; then
        chown -R user:user "$DIR" 2>/dev/null || true
    fi
    exec gosu user uvicorn main:app --host 0.0.0.0 --port "${PORT:-7860}"
fi

exec uvicorn main:app --host 0.0.0.0 --port "${PORT:-7860}"
