from __future__ import annotations

from app.core.database import execute

version = "0002_saas_foundation"


def upgrade() -> None:
    """Add SaaS tables: plans, workspaces, workspace_members, usage_events,
    billing_events, training_examples, action_outcomes.
    Also extends tenants with workspace_id and plan_name.
    """
    execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")  # for gen_random_uuid()

    # ── Plans ──────────────────────────────────────────────────────────────
    execute("""
        CREATE TABLE IF NOT EXISTS plans (
            id                    SERIAL      PRIMARY KEY,
            name                  VARCHAR(64) NOT NULL UNIQUE,
            display_name          VARCHAR(128) NOT NULL,
            price_monthly_cents   INT         NOT NULL DEFAULT 0,
            price_yearly_cents    INT         NOT NULL DEFAULT 0,
            max_seats             INT,
            max_actions_per_mo    INT,
            max_agent_runs_per_mo INT,
            max_llm_tokens_per_mo BIGINT,
            max_training_examples INT,
            features              JSONB       NOT NULL DEFAULT '{}',
            active                BOOLEAN     NOT NULL DEFAULT TRUE,
            created_at            TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    execute("""
        INSERT INTO plans (name, display_name, price_monthly_cents, price_yearly_cents,
                           max_seats, max_actions_per_mo, max_agent_runs_per_mo,
                           max_llm_tokens_per_mo, max_training_examples, features)
        VALUES
            ('starter',    'Starter',    4900,   49000,   5,    100,  50,   500000,    10,
             '{"chat":true,"agents":false,"training":false,"sso":false}'),
            ('pro',        'Pro',        19900,  199000,  25,   1000, 500,  5000000,   100,
             '{"chat":true,"agents":true,"training":true,"sso":false}'),
            ('enterprise', 'Enterprise', 0,      0,       NULL, NULL, NULL, NULL,      NULL,
             '{"chat":true,"agents":true,"training":true,"sso":true,"custom_integrations":true}')
        ON CONFLICT (name) DO NOTHING
    """)

    # ── Workspaces ─────────────────────────────────────────────────────────
    execute("""
        CREATE TABLE IF NOT EXISTS workspaces (
            id                     UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
            slug                   VARCHAR(64) NOT NULL UNIQUE,
            name                   VARCHAR(128) NOT NULL,
            owner_user_id          VARCHAR(128) NOT NULL,
            plan_name              VARCHAR(64) NOT NULL DEFAULT 'starter' REFERENCES plans(name),
            stripe_customer_id     VARCHAR(128),
            stripe_subscription_id VARCHAR(128),
            subscription_status    VARCHAR(32) DEFAULT 'trialing',
            trial_ends_at          TIMESTAMPTZ DEFAULT (NOW() + INTERVAL '14 days'),
            current_period_start   TIMESTAMPTZ,
            current_period_end     TIMESTAMPTZ,
            logo_url               TEXT,
            metadata               JSONB       NOT NULL DEFAULT '{}',
            active                 BOOLEAN     NOT NULL DEFAULT TRUE,
            created_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at             TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)

    # ── Workspace members ──────────────────────────────────────────────────
    execute("""
        CREATE TABLE IF NOT EXISTS workspace_members (
            id           SERIAL      PRIMARY KEY,
            workspace_id UUID        NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
            username     VARCHAR(128) NOT NULL,
            email        VARCHAR(256),
            role         VARCHAR(32) NOT NULL DEFAULT 'developer',
            invited_by   VARCHAR(128),
            joined_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            active       BOOLEAN     NOT NULL DEFAULT TRUE,
            UNIQUE(workspace_id, username)
        )
    """)

    # ── Extend tenants with workspace + plan ───────────────────────────────
    execute("ALTER TABLE tenants ADD COLUMN IF NOT EXISTS workspace_id UUID REFERENCES workspaces(id)")
    execute("ALTER TABLE tenants ADD COLUMN IF NOT EXISTS plan_name VARCHAR(64) NOT NULL DEFAULT 'starter'")

    # ── Usage events ───────────────────────────────────────────────────────
    execute("""
        CREATE TABLE IF NOT EXISTS usage_events (
            id           BIGSERIAL   PRIMARY KEY,
            tenant_id    VARCHAR(128) NOT NULL,
            workspace_id UUID,
            event_type   VARCHAR(64) NOT NULL,
            quantity     BIGINT      NOT NULL DEFAULT 1,
            meta         JSONB       NOT NULL DEFAULT '{}',
            billed_period VARCHAR(7),
            created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    execute("CREATE INDEX IF NOT EXISTS idx_usage_tenant_period ON usage_events(tenant_id, billed_period, event_type)")

    execute("""
        CREATE OR REPLACE VIEW usage_monthly AS
        SELECT
            tenant_id, workspace_id, billed_period,
            SUM(CASE WHEN event_type = 'llm_tokens'      THEN quantity ELSE 0 END) AS llm_tokens,
            SUM(CASE WHEN event_type = 'agent_run'       THEN quantity ELSE 0 END) AS agent_runs,
            SUM(CASE WHEN event_type = 'action_executed' THEN quantity ELSE 0 END) AS actions,
            SUM(CASE WHEN event_type = 'api_call'        THEN quantity ELSE 0 END) AS api_calls,
            COUNT(*) AS total_events
        FROM usage_events
        GROUP BY tenant_id, workspace_id, billed_period
    """)

    # ── Billing events ─────────────────────────────────────────────────────
    execute("""
        CREATE TABLE IF NOT EXISTS billing_events (
            id              BIGSERIAL   PRIMARY KEY,
            workspace_id    UUID        REFERENCES workspaces(id) ON DELETE SET NULL,
            stripe_event_id VARCHAR(128) UNIQUE,
            event_type      VARCHAR(64) NOT NULL,
            payload         JSONB       NOT NULL DEFAULT '{}',
            processed       BOOLEAN     NOT NULL DEFAULT FALSE,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)

    # ── Training examples ──────────────────────────────────────────────────
    execute("""
        CREATE TABLE IF NOT EXISTS training_examples (
            id              BIGSERIAL   PRIMARY KEY,
            tenant_id       VARCHAR(128) NOT NULL,
            workspace_id    UUID,
            trigger_text    TEXT        NOT NULL,
            correct_plan    JSONB       NOT NULL DEFAULT '{}',
            correct_action  VARCHAR(128),
            correct_params  JSONB       NOT NULL DEFAULT '{}',
            outcome         VARCHAR(32) DEFAULT 'resolved',
            tags            TEXT[]      NOT NULL DEFAULT '{}',
            source          VARCHAR(32) DEFAULT 'manual',
            quality_score   FLOAT       DEFAULT 1.0,
            approved        BOOLEAN     NOT NULL DEFAULT TRUE,
            embedding       vector(1536),
            created_by      VARCHAR(128),
            created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    execute("CREATE INDEX IF NOT EXISTS idx_training_tenant ON training_examples(tenant_id, approved)")

    # ── Action outcomes (auto training data collection) ────────────────────
    execute("""
        CREATE TABLE IF NOT EXISTS action_outcomes (
            id                   BIGSERIAL   PRIMARY KEY,
            tenant_id            VARCHAR(128) NOT NULL,
            workspace_id         UUID,
            session_id           VARCHAR(128),
            action               VARCHAR(128) NOT NULL,
            params               JSONB       NOT NULL DEFAULT '{}',
            success              BOOLEAN     NOT NULL,
            error_msg            TEXT,
            duration_ms          INT,
            promoted_to_training BOOLEAN     NOT NULL DEFAULT FALSE,
            created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    execute("CREATE INDEX IF NOT EXISTS idx_outcomes_tenant_action ON action_outcomes(tenant_id, action, success)")


def downgrade() -> None:
    execute("DROP TABLE IF EXISTS action_outcomes")
    execute("DROP TABLE IF EXISTS training_examples")
    execute("DROP TABLE IF EXISTS billing_events")
    execute("DROP VIEW  IF EXISTS usage_monthly")
    execute("DROP TABLE IF EXISTS usage_events")
    execute("ALTER TABLE tenants DROP COLUMN IF EXISTS workspace_id")
    execute("ALTER TABLE tenants DROP COLUMN IF EXISTS plan_name")
    execute("DROP TABLE IF EXISTS workspace_members")
    execute("DROP TABLE IF EXISTS workspaces")
    execute("DROP TABLE IF EXISTS plans")
