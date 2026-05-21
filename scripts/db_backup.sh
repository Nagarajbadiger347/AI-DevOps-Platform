#!/bin/sh
# Postgres automated backup loop.
# Runs every BACKUP_INTERVAL_SECONDS, keeps last BACKUP_RETENTION_DAYS of dumps,
# optionally syncs the latest dump to BACKUP_S3_BUCKET.
#
# Intended to run as the entrypoint of a postgres:alpine sidecar container.
# Why pg_dump --format=custom (-Fc): smaller than plain SQL, supports
# pg_restore --jobs for parallel restore, and skips the "psql replay all
# DDL" failure modes on schema mismatch.

set -eu

BACKUP_DIR=/backups
INTERVAL=${BACKUP_INTERVAL_SECONDS:-86400}
RETENTION_DAYS=${BACKUP_RETENTION_DAYS:-14}
S3_BUCKET=${BACKUP_S3_BUCKET:-}

mkdir -p "$BACKUP_DIR"

# Install aws-cli only when S3 sync is requested. Keeps the image small for
# users who don't need off-host sync.
if [ -n "$S3_BUCKET" ]; then
  if ! command -v aws >/dev/null 2>&1; then
    echo "[db_backup] installing aws-cli for S3 sync..."
    apk add --no-cache aws-cli >/dev/null 2>&1 || {
      echo "[db_backup] WARN: aws-cli install failed; S3 sync disabled"
      S3_BUCKET=""
    }
  fi
fi

backup_once() {
  ts=$(date -u +%Y%m%d-%H%M%SZ)
  out="$BACKUP_DIR/dump-${PGDATABASE:-nexusops}-${ts}.pgc"
  echo "[db_backup] starting dump → $out"

  if pg_dump --format=custom --compress=9 --no-owner --no-privileges -f "$out"; then
    size=$(wc -c < "$out" 2>/dev/null || echo "?")
    echo "[db_backup] dump complete (${size} bytes)"
  else
    echo "[db_backup] ERROR: pg_dump failed"
    rm -f "$out"
    return 1
  fi

  if [ -n "$S3_BUCKET" ]; then
    if aws s3 cp "$out" "s3://${S3_BUCKET}/postgres/$(basename "$out")"; then
      echo "[db_backup] uploaded to s3://${S3_BUCKET}/postgres/"
    else
      echo "[db_backup] WARN: S3 upload failed; dump kept locally"
    fi
  fi

  # Prune local dumps older than retention window
  find "$BACKUP_DIR" -name "dump-*.pgc" -mtime "+${RETENTION_DAYS}" -delete 2>/dev/null || true

  # Maintain a `latest` symlink for one-step restore
  ln -sf "$(basename "$out")" "$BACKUP_DIR/latest.pgc"
}

# Take an immediate backup at startup, then loop on the interval.
backup_once || echo "[db_backup] initial dump failed; will retry on next tick"

while true; do
  sleep "$INTERVAL"
  backup_once || echo "[db_backup] dump failed; will retry on next tick"
done
