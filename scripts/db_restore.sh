#!/bin/sh
# Postgres restore helper. Run from the host:
#
#   scripts/db_restore.sh                 # restore the latest dump (latest.pgc)
#   scripts/db_restore.sh dump-foo.pgc    # restore a specific dump
#
# This drops and recreates the database, then restores from the named dump
# in the app_backups volume. Use with care — it destroys current data.

set -eu

COMPOSE=${COMPOSE:-docker compose}
DUMP=${1:-latest.pgc}

# Confirm — production restores are not something you want to fat-finger.
printf '\n*** This will DROP and RECREATE the database from %s.\n' "$DUMP"
printf '*** All current data will be lost.\n'
printf 'Type "RESTORE" to continue: '
read -r confirm
if [ "$confirm" != "RESTORE" ]; then
  echo "Aborted."
  exit 1
fi

# Verify the dump exists inside the backup volume.
if ! $COMPOSE run --rm --no-deps -v app_backups:/backups db-backup sh -c "test -f /backups/$DUMP"; then
  echo "ERROR: /backups/$DUMP not found inside app_backups volume"
  exit 1
fi

# Drop & recreate the database, then pg_restore in parallel.
$COMPOSE exec -T postgres sh -c '
  set -e
  psql -U "$POSTGRES_USER" -d postgres -c "DROP DATABASE IF EXISTS \"$POSTGRES_DB\";"
  psql -U "$POSTGRES_USER" -d postgres -c "CREATE DATABASE \"$POSTGRES_DB\" OWNER \"$POSTGRES_USER\";"
'

$COMPOSE run --rm --no-deps \
  -v app_backups:/backups \
  -e PGPASSWORD \
  db-backup sh -c "pg_restore -h postgres -U \"\$PGUSER\" -d \"\$PGDATABASE\" --no-owner --no-privileges --jobs=4 /backups/$DUMP"

echo "Restore complete. You may need to restart the nexusops container so it picks up the restored state."
