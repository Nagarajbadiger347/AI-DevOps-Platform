#!/usr/bin/env python3
"""Simple management CLI for NexusOps.

Usage:
  python manage.py migrate
  python manage.py rollback
  python manage.py status
"""

from __future__ import annotations

import argparse
import sys

from app.core.database import health_check
from app.core.schema import apply_migrations, get_applied_migrations, get_pending_migrations, rollback_migration


def migrate() -> int:
    if not health_check():
        print("Database is not reachable. Check DATABASE_URL and database state.", file=sys.stderr)
        return 1

    try:
        apply_migrations()
        print("Migrations applied successfully.")
        return 0
    except Exception as exc:
        print(f"Migration failed: {exc}", file=sys.stderr)
        return 1


def rollback() -> int:
    if not health_check():
        print("Database is not reachable. Check DATABASE_URL and database state.", file=sys.stderr)
        return 1

    try:
        rollback_migration()
        print("Migration rolled back successfully.")
        return 0
    except Exception as exc:
        print(f"Rollback failed: {exc}", file=sys.stderr)
        return 1


def status() -> int:
    try:
        reachable = health_check()
    except Exception as exc:
        print(f"Database health check failed: {exc}", file=sys.stderr)
        reachable = False

    print(f"Database reachable: {reachable}")

    if not reachable:
        return 1

    try:
        applied = get_applied_migrations()
        pending = get_pending_migrations()
        print(f"Applied migrations: {applied}")
        print(f"Pending migrations: {pending}")
        return 0
    except Exception as exc:
        print(f"Unable to read migration status: {exc}", file=sys.stderr)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="NexusOps management CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("migrate", help="Apply pending database migrations")
    subparsers.add_parser("rollback", help="Rollback the last applied migration")
    subparsers.add_parser("status", help="Show migration and database status")

    args = parser.parse_args()

    if args.command == "migrate":
        return migrate()
    if args.command == "rollback":
        return rollback()
    if args.command == "status":
        return status()

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
