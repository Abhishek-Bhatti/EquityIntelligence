import aiosqlite
import os
from contextlib import asynccontextmanager

DB_PATH = os.path.join("data", "raw_data.db")

@asynccontextmanager
async def get_db_connection():
    db = await aiosqlite.connect(DB_PATH)
    await db.execute("PRAGMA journal_mode=WAL;")
    await db.execute("PRAGMA synchronous=NORMAL;")
    await db.execute("PRAGMA cache_size=-40000;")
    try:
        yield db
    finally:
        await db.close()


async def init_db():
    """Initializes the SQLite database, creates tables, and applies schema migrations."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    async with get_db_connection() as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS raw_text_staging (
                text_block_id TEXT PRIMARY KEY,
                ticker TEXT NOT NULL,
                source TEXT NOT NULL,
                period TEXT NOT NULL,
                raw_content TEXT NOT NULL,
                is_parsed INTEGER DEFAULT 0
            );
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS parsed_insights (
                insight_id TEXT PRIMARY KEY,
                ticker TEXT NOT NULL,
                source TEXT NOT NULL,
                period TEXT NOT NULL,
                revenue_growth_guidance TEXT,
                ebitda_margin_trend TEXT,
                capex_plans TEXT,
                key_risks_mentioned TEXT,
                management_sentiment TEXT,
                raw_management_quotes TEXT,
                raw_json_output TEXT
            );
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS equity_research_reports (
                ticker TEXT PRIMARY KEY,
                investment_score TEXT,
                report_payload TEXT,
                report_type TEXT DEFAULT 'FULL',
                last_processed_period TEXT,
                last_processed_period_concall TEXT,
                last_processed_period_ar TEXT,
                generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.execute("CREATE INDEX IF NOT EXISTS idx_staging_lookup ON raw_text_staging (ticker, is_parsed);")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_insights_lookup ON parsed_insights (ticker, source, period);")

        # Safe migrations for existing databases
        for migration in [
            "ALTER TABLE parsed_insights ADD COLUMN raw_management_quotes TEXT;",
            "ALTER TABLE equity_research_reports ADD COLUMN report_type TEXT DEFAULT 'FULL';",
            "ALTER TABLE equity_research_reports ADD COLUMN last_processed_period TEXT;",
            "ALTER TABLE equity_research_reports ADD COLUMN last_processed_period_concall TEXT;",
            "ALTER TABLE equity_research_reports ADD COLUMN last_processed_period_ar TEXT;",
        ]:
            try:
                await db.execute(migration)
            except Exception:
                pass  # Column already exists

        await db.commit()