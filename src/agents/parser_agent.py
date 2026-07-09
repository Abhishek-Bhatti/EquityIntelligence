import json
import asyncio
import logging
import groq
from typing import List
from groq import AsyncGroq
from pydantic import BaseModel, Field
from src.database.connection import get_db_connection
from src.utils.rate_limiter import AsyncTokenBucketLimiter

logger = logging.getLogger(__name__)


class CorporateInsightSchema(BaseModel):
    revenue_growth_guidance: str = Field(
        description="Merged revenue trajectories and volume demand with exact figures."
    )
    revenue_source_chunks: List[str] = Field(
        description="Deduplicated chunk IDs sourcing revenue data."
    )
    ebitda_margin_trend: str = Field(
        description="Merged margin and input cost analysis with exact figures."
    )
    ebitda_source_chunks: List[str] = Field(
        description="Deduplicated chunk IDs sourcing margin data."
    )
    capex_plans: str = Field(
        description="Merged capex numbers and project timelines with exact figures."
    )
    capex_source_chunks: List[str] = Field(
        description="Deduplicated chunk IDs sourcing capex data."
    )
    key_risks_mentioned: str = Field(
        description="Merged operational and financial risk factors."
    )
    risk_source_chunks: List[str] = Field(
        description="Deduplicated chunk IDs sourcing risk data."
    )
    management_sentiment: str = Field(
        description="Final BULLISH/CAUTIOUS/NEUTRAL with rationale."
    )
    raw_management_quotes: List[dict] = Field(
        description="Verbatim management quotes with source chunk IDs. Never paraphrase."
    )


class ParserAgent:
    def __init__(self, groq_client: AsyncGroq, groq_limiter: AsyncTokenBucketLimiter):
        self.groq_client = groq_client
        self.groq_limiter = groq_limiter
        self.model = "meta-llama/llama-4-scout-17b-16e-instruct"

    def _get_reduce_system_prompt(self) -> str:
        return (
            "You are a financial data merge API. You receive multiple partial JSON extractions "
            "from page windows of the same corporate document and must merge them into one schema. "
            "Rules: combine non-empty text fields by appending with ' | ' as separator; "
            "deduplicate all chunk ID arrays; for management_sentiment pick the most data-supported "
            "sentiment across all windows; for raw_management_quotes collect ALL entries from ALL "
            "windows into one deduplicated array — never summarize or paraphrase any quote. "
            "CRITICAL: Preserve all numerical values exactly as they appear. "
            "Output only valid minified JSON, no markdown."
        )

    async def _reduce_document(
        self, ticker: str, source: str, period: str, partial_jsons: list[str]):
        """Merges all map outputs for one document into one CorporateInsightSchema."""
        source_upper = source.upper()
        ticker_upper = ticker.upper()
        insight_id = f"{ticker_upper}_{source_upper}_{period}"

        async with get_db_connection() as db:
            async with db.execute(
                "SELECT 1 FROM parsed_insights WHERE insight_id = ? LIMIT 1",
                (insight_id,)
            ) as cursor:
                if await cursor.fetchone():
                    logger.info("⏭️ [Reduce Cache Hit] %s already exists. Skipping.", insight_id)
                    return

        label = "AR_REDUCE" if "AR" in source_upper else "CONCALL_REDUCE"
        system_prompt = self._get_reduce_system_prompt()
        combined_partials = "\n---WINDOW_BOUNDARY---\n".join(partial_jsons)
        user_prompt = (
            f"Merge these {len(partial_jsons)} partial window extractions for "
            f"{ticker_upper} ({source_upper} - {period}) into one final JSON object:\n"
            "{\n"
            '  "revenue_growth_guidance": "Merged revenue data with exact figures preserved",\n'
            '  "revenue_source_chunks": ["CHUNK_ID_1"],\n'
            '  "ebitda_margin_trend": "Merged margin data with exact figures preserved",\n'
            '  "ebitda_source_chunks": ["CHUNK_ID_1"],\n'
            '  "capex_plans": "Merged capex data with exact figures preserved",\n'
            '  "capex_source_chunks": ["CHUNK_ID_1"],\n'
            '  "key_risks_mentioned": "Merged risk data",\n'
            '  "risk_source_chunks": ["CHUNK_ID_1"],\n'
            '  "management_sentiment": "BULLISH/CAUTIOUS/NEUTRAL with rationale",\n'
            '  "raw_management_quotes": [{"chunk_id": "ID", "quote": "verbatim quote"}]\n'
            "}\n\n"
            f"Partial Window Extractions:\n{combined_partials}"
        )
        estimated_tokens = int((len(system_prompt) + len(user_prompt)) / 3) + 1500

        while True:
            try:
                logger.info(
                    "⏳ [%s] Reduce call for %s (%d windows)...",
                    label, insight_id, len(partial_jsons)
                )
                await self.groq_limiter.consume(estimated_tokens)
                completion = await self.groq_client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    model=self.model,
                    response_format={"type": "json_object"},
                    temperature=0.0,
                    extra_headers={"groq-prompt-caching": "true"},
                )
                actual_tokens = completion.usage.total_tokens
                await self.groq_limiter.correct_from_actual_usage(estimated_tokens, actual_tokens)
                raw_response = completion.choices[0].message.content.strip()
                validated = CorporateInsightSchema(**json.loads(raw_response))

                async with get_db_connection() as db:
                    await db.execute("""
                        INSERT OR REPLACE INTO parsed_insights
                        (insight_id, ticker, source, period, revenue_growth_guidance,
                         ebitda_margin_trend, capex_plans, key_risks_mentioned,
                         management_sentiment, raw_management_quotes, raw_json_output)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        insight_id, ticker_upper, source_upper, period,
                        validated.revenue_growth_guidance,
                        validated.ebitda_margin_trend,
                        validated.capex_plans,
                        validated.key_risks_mentioned,
                        validated.management_sentiment,
                        json.dumps(validated.raw_management_quotes),
                        raw_response
                    ))
                    await db.execute("""
                        UPDATE raw_text_staging SET is_parsed = 1
                        WHERE UPPER(ticker) = ? AND source = ? AND period = ?
                    """, (ticker_upper, source_upper, period))
                    await db.commit()
                logger.info("✅ [%s] Reduced and stored: %s.", label, insight_id)
                break
            except groq.RateLimitError as e:
                if any(x in str(e).lower() for x in ["daily", "tpd", "rate_limit_exceeded"]):
                    await self.groq_limiter.handle_groq_daily_limit_backoff(e)
                    continue
                raise e
            except Exception as e:
                logger.error("❌ [Reduce Failure] %s: %s", insight_id, e)
                break

    async def parse_unprocessed_documents(
        self, ticker: str, source_filter: str | None = None):
        """Groups staged map outputs by document and fires one reduce call per document."""
        ticker_upper = ticker.upper()
        document_map: dict[tuple, list[str]] = {}

        sql = """
            SELECT source, period, raw_content
            FROM raw_text_staging
            WHERE UPPER(ticker) = ? AND is_parsed = 0
        """
        params: list = [ticker_upper]
        if source_filter:
            sql += " AND UPPER(source) = ?"
            params.append(source_filter.upper())
        sql += " ORDER BY source ASC, period ASC, text_block_id ASC"

        async with get_db_connection() as db:
            async with db.execute(sql, params) as cursor:
                async for source, period, raw_content in cursor:
                    key = (source.upper(), period)
                    if key not in document_map:
                        document_map[key] = []
                    document_map[key].append(raw_content)

        if not document_map:
            scope = f" (source={source_filter})" if source_filter else ""
            logger.info("[ParserAgent] No unprocessed documents for %s%s.", ticker_upper, scope)
            return

        tasks = [
            self._reduce_document(ticker_upper, source, period, partial_jsons)
            for (source, period), partial_jsons in document_map.items()
        ]
        logger.info("🚀 [ParserAgent] Firing %d reduce calls (1 per document)...", len(tasks))
        await asyncio.gather(*tasks)
        logger.info("🏁 [ParserAgent] Reduce pass complete for %s.", ticker_upper)