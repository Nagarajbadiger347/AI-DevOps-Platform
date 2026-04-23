from __future__ import annotations

from app.core.database import execute

version = "0001_initial"


def upgrade() -> None:
    """Create initial application tables and indexes."""
    # Enable pgvector extension for vector embeddings
    execute("CREATE EXTENSION IF NOT EXISTS vector")

    execute(
        """
        CREATE TABLE IF NOT EXISTS tenants (
            tenant_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            aws_role_arn TEXT,
            aws_region TEXT,
            slack_channel TEXT,
            llm_provider TEXT,
            active BOOLEAN DEFAULT TRUE,
            metadata JSONB DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ
        )
        """
    )

    execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL,
            username TEXT NOT NULL,
            email TEXT,
            password_hash TEXT,
            role TEXT DEFAULT 'viewer',
            invite_pending BOOLEAN DEFAULT FALSE,
            active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ
        )
        """
    )
    execute("CREATE UNIQUE INDEX IF NOT EXISTS users_tenant_username_idx ON users (tenant_id, username)")
    execute("CREATE UNIQUE INDEX IF NOT EXISTS users_tenant_email_idx ON users (tenant_id, lower(email))")

    execute(
        """
        CREATE TABLE IF NOT EXISTS approvals (
            approval_id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL,
            incident_id TEXT,
            action_type TEXT,
            description TEXT,
            requested_by TEXT,
            status TEXT,
            estimated_cost NUMERIC,
            metadata JSONB DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            resolved_at TIMESTAMPTZ,
            approved_by TEXT
        )
        """
    )
    execute("CREATE INDEX IF NOT EXISTS approvals_tenant_status_idx ON approvals (tenant_id, status)")

    execute(
        """
        CREATE TABLE IF NOT EXISTS incidents (
            incident_id TEXT NOT NULL,
            tenant_id TEXT NOT NULL,
            incident_type TEXT,
            source TEXT,
            description TEXT,
            root_cause TEXT,
            resolution TEXT,
            severity TEXT,
            actions_taken JSONB DEFAULT '[]'::jsonb,
            metadata JSONB DEFAULT '{}'::jsonb,
            embedding VECTOR(1536),
            created_at TIMESTAMPTZ DEFAULT NOW(),
            resolved_at TIMESTAMPTZ,
            PRIMARY KEY (tenant_id, incident_id)
        )
        """
    )
    execute("CREATE INDEX IF NOT EXISTS incidents_tenant_type_idx ON incidents (tenant_id, incident_type)")

    execute(
        """
        CREATE TABLE IF NOT EXISTS chat_sessions (
            session_id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            last_active TIMESTAMPTZ DEFAULT NOW(),
            context JSONB DEFAULT '{}'::jsonb
        )
        """
    )
    execute("CREATE INDEX IF NOT EXISTS chat_sessions_tenant_idx ON chat_sessions (tenant_id)")

    execute(
        """
        CREATE TABLE IF NOT EXISTS chat_messages (
            message_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            tenant_id TEXT NOT NULL,
            role TEXT,
            content TEXT,
            metadata JSONB DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
        """
    )
    execute("CREATE INDEX IF NOT EXISTS chat_messages_session_tenant_idx ON chat_messages (session_id, tenant_id)")

    execute(
        """
        CREATE TABLE IF NOT EXISTS tenant_costs (
            id SERIAL PRIMARY KEY,
            tenant_id TEXT NOT NULL,
            cost_usd NUMERIC,
            service TEXT,
            recorded_at TIMESTAMPTZ DEFAULT NOW()
        )
        """
    )
    execute("CREATE INDEX IF NOT EXISTS tenant_costs_tenant_idx ON tenant_costs (tenant_id)")


def downgrade() -> None:
    """Drop initial application tables and indexes."""
    execute("DROP TABLE IF EXISTS tenant_costs")
    execute("DROP TABLE IF EXISTS chat_messages")
    execute("DROP TABLE IF EXISTS chat_sessions")
    execute("DROP TABLE IF EXISTS incidents")
    execute("DROP TABLE IF EXISTS approvals")
    execute("DROP TABLE IF EXISTS users")
    execute("DROP TABLE IF EXISTS tenants")

    # Drop pgvector extension
    execute("DROP EXTENSION IF EXISTS vector")
