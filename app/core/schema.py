"""PostgreSQL schema migration runner for NexusOps.

This module implements a lightweight migration system that tracks applied schema
changes in `schema_migrations` and discovers migrations from `app.migrations`.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from typing import Callable

from app.core.database import execute

logger = logging.getLogger(__name__)

MigrationFn = Callable[[], None]


def _ensure_migration_table() -> None:
    execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version TEXT PRIMARY KEY,
            applied_at TIMESTAMPTZ DEFAULT NOW()
        )
        """
    )


def _get_applied_migrations() -> set[str]:
    rows = execute("SELECT version FROM schema_migrations ORDER BY version")
    return {row["version"] for row in rows}


def _record_migration(version: str) -> None:
    execute(
        "INSERT INTO schema_migrations (version) VALUES (%s) ON CONFLICT DO NOTHING",
        (version,)
    )


def _remove_migration(version: str) -> None:
    execute("DELETE FROM schema_migrations WHERE version = %s", (version,))


def get_applied_migrations() -> list[str]:
    """Return the list of applied migration versions."""
    _ensure_migration_table()
    return sorted(_get_applied_migrations())


def get_pending_migrations() -> list[str]:
    """Return the list of migrations that have not yet been applied."""
    applied = set(get_applied_migrations())
    return [version for version, _, _ in MIGRATIONS if version not in applied]


def _load_migrations() -> list[tuple[str, MigrationFn, MigrationFn]]:
    import app.migrations as migrations_pkg

    migrations: list[tuple[str, MigrationFn, MigrationFn]] = []
    for _, module_name, is_package in pkgutil.iter_modules(migrations_pkg.__path__):
        if is_package or module_name.startswith("_"):
            continue

        module = importlib.import_module(f"app.migrations.{module_name}")
        version = getattr(module, "version", None)
        upgrade = getattr(module, "upgrade", None)
        downgrade = getattr(module, "downgrade", None)

        if not isinstance(version, str) or not callable(upgrade) or not callable(downgrade):
            raise RuntimeError(f"Invalid migration module: app.migrations.{module_name}")

        migrations.append((version, upgrade, downgrade))

    return sorted(migrations, key=lambda item: item[0])


MIGRATIONS: list[tuple[str, MigrationFn, MigrationFn]] = _load_migrations()


def apply_migrations() -> None:
    _ensure_migration_table()
    applied = _get_applied_migrations()

    for version, upgrade, _ in MIGRATIONS:
        if version in applied:
            continue

        logger.info("applying_schema_migration version=%s", version)
        upgrade()
        _record_migration(version)
        logger.info("schema_migration_applied version=%s", version)

    logger.info("schema_migrations_up_to_date", extra={"applied_count": len(MIGRATIONS)})


def rollback_migration(target_version: str | None = None) -> None:
    """Rollback to target_version or one step back if None."""
    _ensure_migration_table()
    applied = sorted(_get_applied_migrations())

    if not applied:
        logger.info("no_migrations_to_rollback")
        return

    if target_version is None:
        # Rollback last migration
        rollback_version = applied[-1]
    else:
        if target_version not in applied:
            raise ValueError(f"Target version {target_version} not applied")
        rollback_version = target_version

    # Find the migration
    for version, _, downgrade in MIGRATIONS:
        if version == rollback_version:
            logger.info("rolling_back_migration version=%s", version)
            downgrade()
            _remove_migration(version)
            logger.info("migration_rolled_back version=%s", version)
            return

    raise ValueError(f"Migration {rollback_version} not found")


# Backwards compatible alias for existing startup code.
def create_schema() -> None:
    apply_migrations()
