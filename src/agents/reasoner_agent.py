import os
import json
import asyncio
import logging
import groq
from datetime import datetime
from groq import AsyncGroq
from pydantic import BaseModel, Field
from src.database.connection import get_db_connection
from src.utils.rate_limiter import AsyncTokenBucketLimiter

logger = logging.getLogger(__name__)


class FinalInvestmentReportSchema(BaseModel):
    summary_text: str = Field(description="High-level macro synthesis and investment core thesis.")
    financials_and_notes_analysis: str = Field(description="Deep dive into financial highlights with exact figures.")
    financials_referenced_chunks: list[str] = Field(description="Chunk IDs verifying the financials section.")
    management_sentiment_synthesis: str = Field(description="Detailed analysis of tone changes across quarters.")
    sentiment_referenced_chunks: list[str] = Field(description="Chunk IDs verifying the sentiment section.")
    company_moats: str = Field(description="Structural advantages, brand assets, and capacity moats.")
    moats_referenced_chunks: list[str] = Field(description="Chunk IDs verifying the moats section.")
    red_flags: str = Field(description="Deep dive on channel risks and regulatory alerts.")
    red_flags_referenced_chunks: list[str] = Field(description="Chunk IDs verifying the red flags section.")
    promise_evaluator_matrix: str = Field(description="Chronological audit of management promises vs outcomes.")
    promise_matrix_referenced_chunks: list[str] = Field(description="Chunk IDs verifying the promise matrix.")
    investment_score: str = Field(description="Score out of 100 with buy/sell logic, e.g. '72/100 - Buy'.")


class ReasonerAgent:
    def __init__(self, client: AsyncGroq, limiter: AsyncTokenBucketLimiter):
        self.client = client
        self.limiter = limiter
        self.model = "meta-llama/llama-4-scout-17b-16e-instruct"

    def _parse_period_to_datetime(self, period: str) -> datetime:
        """Converts stored period strings to datetime for comparison.
        NOTE: only meaningful for comparing periods *within the same source*.
        AR periods ("2026") and concall periods ("FEB_2026") are not on a
        comparable timeline and must never be compared against each other."""
        try:
            period_normalized = period[:3].capitalize() + period[3:]  # FEB_2026 -> Feb_2026
            return datetime.strptime(period_normalized, "%b_%Y")
        except ValueError:
            pass
        try:
            return datetime(int(period[:4]), 1, 1)
        except (ValueError, IndexError):
            return datetime(2000, 1, 1)

    async def check_local_cache(self, ticker: str) -> FinalInvestmentReportSchema | None:
        """Returns cached report if one exists, otherwise None."""
        async with get_db_connection() as db:
            async with db.execute("""
                SELECT report_payload FROM equity_research_reports WHERE ticker = ?
            """, (ticker.upper(),)) as cursor:
                row = await cursor.fetchone()

        if row:
            logger.info("🎯 [Cache Hit] Report for %s found in local storage.", ticker.upper())
            return FinalInvestmentReportSchema(**json.loads(row[0]))
        return None

    async def _save_report(
        self, ticker: str, report: FinalInvestmentReportSchema, report_type: str,
        concall_period: str | None = None, ar_period: str | None = None,):
        """Persists report with metadata to local storage.
        concall_period / ar_period are only written when provided (non-None) —
        COALESCE keeps whichever checkpoint wasn't touched by this call intact,
        so folding in an AR update can never clobber the concall checkpoint
        and vice versa."""
        async with get_db_connection() as db:
            await db.execute("""
                INSERT INTO equity_research_reports
                (ticker, investment_score, report_payload, report_type,
                 last_processed_period_concall, last_processed_period_ar)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker) DO UPDATE SET
                    investment_score = excluded.investment_score,
                    report_payload = excluded.report_payload,
                    report_type = excluded.report_type,
                    last_processed_period_concall = COALESCE(
                        excluded.last_processed_period_concall,
                        equity_research_reports.last_processed_period_concall
                    ),
                    last_processed_period_ar = COALESCE(
                        excluded.last_processed_period_ar,
                        equity_research_reports.last_processed_period_ar
                    )
            """, (
                ticker.upper(), report.investment_score, report.model_dump_json(),
                report_type, concall_period, ar_period,
            ))
            await db.commit()

        logger.info(
            "💾 [%s Report Saved] %s — concall checkpoint: %s, AR checkpoint: %s.",
            report_type, ticker.upper(), concall_period or '(unchanged)', ar_period or '(unchanged)'
        )

    async def _factual_call(self, ticker: str, source_filter: str | None = None) -> dict:
        """
        Reads parsed_insights for numerical fields and raw_management_quotes.
        source_filter scopes this to one source ('CONCALL' or 'AR') so a
        concall-only report can never silently pick up AR rows mid-flight.
        """
        ticker_upper = ticker.upper()
        sql = """
            SELECT period, source, revenue_growth_guidance, ebitda_margin_trend,
                   capex_plans, raw_management_quotes
            FROM parsed_insights
            WHERE UPPER(ticker) = ?
        """
        chunk_sql = "SELECT text_block_id FROM raw_text_staging WHERE UPPER(ticker) = ?"
        chunk_params = [ticker_upper]
        if source_filter:
            chunk_sql += " AND UPPER(source) = ?"
            chunk_params.append(source_filter.upper())
        async with get_db_connection() as db:
            async with db.execute(chunk_sql, chunk_params) as cursor:
                valid_chunk_ids = sorted(row[0] for row in await cursor.fetchall())
        params = [ticker_upper]
        if source_filter:
            sql += " AND UPPER(source) = ?"
            params.append(source_filter.upper())
        sql += " ORDER BY source ASC, period ASC"

        async with get_db_connection() as db:
            async with db.execute(sql, params) as cursor:
                rows = await cursor.fetchall()

        factual_blocks = []
        all_quotes = []
        for period, source, revenue, ebitda, capex, quotes_json in rows:
            factual_blocks.append(
                f"--- {source} | {period} ---\n"
                f"Revenue: {revenue}\n"
                f"EBITDA: {ebitda}\n"
                f"Capex: {capex}"
            )
            if quotes_json:
                try:
                    all_quotes.extend(json.loads(quotes_json))
                except (json.JSONDecodeError, TypeError):
                    pass

        combined_factual = "\n\n".join(factual_blocks)
        quotes_block = json.dumps(all_quotes, indent=2)

        system_prompt = (
            "You are a financial data synthesis API. "
            "CRITICAL: Preserve all numerical values exactly as written. Never round or approximate. "
            "For the promise matrix, use the verbatim quotes provided — do not paraphrase them. "
            "Output only valid minified JSON, no markdown."
        )
        user_prompt = (
            f"Analyze numerical financial data for {ticker_upper} across all periods.\n\n"
            f"VALID CHUNK IDS — you may ONLY cite chunk IDs from this exact list. "
            f"Never invent, infer, or reference a document title, call name, or period "
            f"that is not in this list, even if mentioned inside the source text:\n"
            f"{json.dumps(valid_chunk_ids)}\n\n"
            "Produce this exact JSON:\n"
            "{\n"
            '  "financials_and_notes_analysis": "Chronological financial trajectory with EXACT numbers. '
            'Use bullet points per quarter. Format: ### Financial Trajectory\\n* Q: [details]",\n'
            '  "financials_referenced_chunks": ["CHUNK_ID_1"],\n'
            '  "promise_evaluator_matrix": "| Quarter | Management Promise | Realized Outcome |\\n|---|---|---|\\n'
            '| PERIOD | verbatim promise from raw_management_quotes | outcome if observable, else Pending |",\n'
            '  "promise_matrix_referenced_chunks": ["CHUNK_ID_1"]\n'
            "}\n\n"
            f"Financial Data by Quarter:\n{combined_factual}\n\n"
            f"Verbatim Management Quotes (use directly in promise matrix, do not paraphrase):\n{quotes_block}"
        )
        estimated_tokens = int((len(system_prompt) + len(user_prompt)) / 3) + 1500

        while True:
            try:
                logger.info(
                    "⏳ [Factual Call] Requesting %d tokens for %s%s...",
                    estimated_tokens, ticker_upper, f" ({source_filter})" if source_filter else ""
                )
                await self.limiter.consume(estimated_tokens)
                completion = await self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    model=self.model,
                    response_format={"type": "json_object"},
                    temperature=0.0
                )
                actual_tokens = completion.usage.total_tokens
                await self.limiter.correct_from_actual_usage(estimated_tokens, actual_tokens)
                return json.loads(completion.choices[0].message.content.strip())
            except groq.RateLimitError as e:
                if any(x in str(e).lower() for x in ["daily", "tpd", "rate_limit_exceeded"]):
                    await self.limiter.handle_groq_daily_limit_backoff(e)
                    continue
                raise e

    async def _qualitative_call(self, ticker: str, factual_summary: str, source_filter: str | None = None) -> dict:
        """
        Reads parsed_insights for qualitative synthesis, optionally scoped to one source.
        """
        ticker_upper = ticker.upper()
        sql = """
            SELECT period, source, key_risks_mentioned, management_sentiment
            FROM parsed_insights
            WHERE UPPER(ticker) = ?
        """
        chunk_sql = "SELECT text_block_id FROM raw_text_staging WHERE UPPER(ticker) = ?"
        chunk_params = [ticker_upper]
        if source_filter:
            chunk_sql += " AND UPPER(source) = ?"
            chunk_params.append(source_filter.upper())
        async with get_db_connection() as db:
            async with db.execute(chunk_sql, chunk_params) as cursor:
                valid_chunk_ids = sorted(row[0] for row in await cursor.fetchall())
        params = [ticker_upper]
        if source_filter:
            sql += " AND UPPER(source) = ?"
            params.append(source_filter.upper())
        sql += " ORDER BY source ASC, period ASC"

        async with get_db_connection() as db:
            async with db.execute(sql, params) as cursor:
                rows = await cursor.fetchall()

        qualitative_blocks = []
        for period, source, risks, sentiment in rows:
            qualitative_blocks.append(
                f"--- {source} | {period} ---\n"
                f"Risks: {risks}\n"
                f"Sentiment: {sentiment}"
            )
        combined_qualitative = "\n\n".join(qualitative_blocks)

        system_prompt = (
            "You are an institutional equity research analyst. "
            "Synthesize qualitative patterns across multiple quarters of corporate data. "
            "Be specific about which quarters drove sentiment changes or introduced new risks. "
            "Output only valid minified JSON, no markdown."
        )
        user_prompt = (
            f"Analyze numerical financial data for {ticker_upper} across all periods.\n\n"
            f"VALID CHUNK IDS — you may ONLY cite chunk IDs from this exact list. "
            f"Never invent, infer, or reference a document title, call name, or period "
            f"that is not in this list, even if mentioned inside the source text:\n"
            f"{json.dumps(valid_chunk_ids)}\n\n"
            "Produce this exact JSON:\n"
            "{\n"
            '  "company_moats": "### Structural Competitive Moats\\n* [specific moat with evidence]",\n'
            '  "moats_referenced_chunks": ["CHUNK_ID_1"],\n'
            '  "red_flags": "### Operational & Regulatory Risks\\n* [specific risk with quarter context]",\n'
            '  "red_flags_referenced_chunks": ["CHUNK_ID_1"],\n'
            '  "management_sentiment_synthesis": "### Tone Progression\\n* [quarter-by-quarter sentiment arc]",\n'
            '  "sentiment_referenced_chunks": ["CHUNK_ID_1"],\n'
            '  "summary_text": "Two-paragraph investment thesis overview.",\n'
            '  "investment_score": "XX/100 - Buy/Sell/Hold with rationale referencing exact figures"\n'
            "}\n\n"
            f"Qualitative Data by Quarter:\n{combined_qualitative}\n\n"
            f"Factual Financial Context (use exact figures from here in investment_score rationale):\n{factual_summary}"
        )
        estimated_tokens = int((len(system_prompt) + len(user_prompt)) / 3) + 1500

        while True:
            try:
                logger.info(
                    "⏳ [Qualitative Call] Requesting %d tokens for %s%s...",
                    estimated_tokens, ticker_upper, f" ({source_filter})" if source_filter else ""
                )
                await self.limiter.consume(estimated_tokens)
                completion = await self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    model=self.model,
                    response_format={"type": "json_object"},
                    temperature=0.1
                )
                actual_tokens = completion.usage.total_tokens
                await self.limiter.correct_from_actual_usage(estimated_tokens, actual_tokens)
                return json.loads(completion.choices[0].message.content.strip())
            except groq.RateLimitError as e:
                if any(x in str(e).lower() for x in ["daily", "tpd", "rate_limit_exceeded"]):
                    await self.limiter.handle_groq_daily_limit_backoff(e)
                    continue
                raise e

    def _filter_valid_chunks(self, chunk_ids: list[str], valid_ids: set[str], field_name: str, ticker: str) -> list[str]:
        """Drops any cited chunk ID that doesn't correspond to a real staged chunk,
        logging each removal so fabricated citations are visible rather than silent."""
        clean = []
        for cid in chunk_ids:
            if cid in valid_ids:
                clean.append(cid)
            else:
                logger.warning(
                    "[Citation Guard] %s: dropping fabricated/unrecognized chunk_id '%s' from %s",
                    ticker, cid, field_name
                )
        return clean

    async def _get_valid_chunk_ids(self, ticker: str) -> set[str]:
        """Fetches the full set of real staged chunk IDs for this ticker —
        the ground truth against which any _referenced_chunks field is checked."""
        async with get_db_connection() as db:
            async with db.execute(
                "SELECT text_block_id FROM raw_text_staging WHERE UPPER(ticker) = ?",
                (ticker.upper(),)
            ) as cursor:
                rows = await cursor.fetchall()
        return {row[0] for row in rows}

    def _sanitize_report_chunks(
        self, report: FinalInvestmentReportSchema, valid_ids: set[str], ticker: str
    ) -> FinalInvestmentReportSchema:
        """Runs _filter_valid_chunks across every *_referenced_chunks field on a report."""
        report.financials_referenced_chunks = self._filter_valid_chunks(
            report.financials_referenced_chunks, valid_ids, "financials_referenced_chunks", ticker
        )
        report.sentiment_referenced_chunks = self._filter_valid_chunks(
            report.sentiment_referenced_chunks, valid_ids, "sentiment_referenced_chunks", ticker
        )
        report.moats_referenced_chunks = self._filter_valid_chunks(
            report.moats_referenced_chunks, valid_ids, "moats_referenced_chunks", ticker
        )
        report.red_flags_referenced_chunks = self._filter_valid_chunks(
            report.red_flags_referenced_chunks, valid_ids, "red_flags_referenced_chunks", ticker
        )
        report.promise_matrix_referenced_chunks = self._filter_valid_chunks(
            report.promise_matrix_referenced_chunks, valid_ids, "promise_matrix_referenced_chunks", ticker
        )
        return report

    async def generate_full_report(self, ticker: str, source_filter: str | None = None) -> FinalInvestmentReportSchema:
        """
        Full two-call synthesis. Pass source_filter='CONCALL' to build the
        first-pass report from concall data only (AR gets folded in later via
        apply_incremental_update). Pass None to synthesize across everything
        currently parsed for the ticker.
        """
        ticker_upper = ticker.upper()
        factual_fields = await self._factual_call(ticker_upper, source_filter=source_filter)
        qualitative_fields = await self._qualitative_call(
            ticker_upper,
            factual_summary=factual_fields.get("financials_and_notes_analysis", ""),
            source_filter=source_filter,
        )
        merged = {**factual_fields, **qualitative_fields}
        report = FinalInvestmentReportSchema(**merged)

        # Strip any fabricated/unrecognized chunk citations before saving
        valid_ids = await self._get_valid_chunk_ids(ticker_upper)
        report = self._sanitize_report_chunks(report, valid_ids, ticker_upper)

        # Checkpoint each source independently — never compare AR periods to
        # concall periods, since their label formats aren't on the same timeline.
        sql = "SELECT period, source FROM parsed_insights WHERE UPPER(ticker) = ?"
        params = [ticker_upper]
        if source_filter:
            sql += " AND UPPER(source) = ?"
            params.append(source_filter.upper())

        async with get_db_connection() as db:
            async with db.execute(sql, params) as cursor:
                rows = await cursor.fetchall()

        concall_periods = [p for p, s in rows if s.upper() == "CONCALL"]
        ar_periods = [p for p, s in rows if s.upper() == "AR"]
        latest_concall = max(concall_periods, key=self._parse_period_to_datetime) if concall_periods else None
        latest_ar = max(ar_periods, key=self._parse_period_to_datetime) if ar_periods else None

        await self._save_report(ticker_upper, report, "FULL", concall_period=latest_concall, ar_period=latest_ar)
        logger.info(
            "✅ [Reasoner] Full report generated for %s%s.",
            ticker_upper, f" ({source_filter} only)" if source_filter else ""
        )
        return report

    async def apply_incremental_update(
        self, ticker: str, existing_report: FinalInvestmentReportSchema, source_filter: str | None = None,) -> FinalInvestmentReportSchema:
        """
        Targeted update call. Reads periods newer than the checkpoint for
        their OWN source — a CONCALL row is only ever compared against the
        concall checkpoint, an AR row only against the AR checkpoint. This is
        what makes it safe to fold a slow-finishing AR document into an
        already-generated concall report: there's no shared timeline to get
        confused about.

        Pass source_filter='AR' when you specifically want to fold a freshly
        finished AR document into a concall-only report. Leave it None for a
        general "pick up anything new since last run" refresh across both
        sources (used by the cache-staleness refresh path).
        """
        ticker_upper = ticker.upper()

        async with get_db_connection() as db:
            async with db.execute("""
                SELECT last_processed_period_concall, last_processed_period_ar
                FROM equity_research_reports WHERE ticker = ?
            """, (ticker_upper,)) as cursor:
                row = await cursor.fetchone()

        last_concall = row[0] if row and row[0] else None
        last_ar = row[1] if row and row[1] else None
        last_concall_dt = self._parse_period_to_datetime(last_concall) if last_concall else datetime(2000, 1, 1)
        last_ar_dt = self._parse_period_to_datetime(last_ar) if last_ar else datetime(2000, 1, 1)

        sql = """
            SELECT period, source, revenue_growth_guidance, ebitda_margin_trend,
                   capex_plans, key_risks_mentioned, management_sentiment, raw_management_quotes
            FROM parsed_insights
            WHERE UPPER(ticker) = ?
        """
        params = [ticker_upper]
        if source_filter:
            sql += " AND UPPER(source) = ?"
            params.append(source_filter.upper())
        sql += " ORDER BY source ASC, period ASC"

        async with get_db_connection() as db:
            async with db.execute(sql, params) as cursor:
                all_rows = await cursor.fetchall()

        def _is_new(row) -> bool:
            period, source = row[0], row[1]
            checkpoint_dt = last_concall_dt if source.upper() == "CONCALL" else last_ar_dt
            return self._parse_period_to_datetime(period) > checkpoint_dt

        new_rows = [row for row in all_rows if _is_new(row)]

        if not new_rows:
            logger.warning(
                "[Reasoner] No new periods found beyond existing checkpoints (concall: %s, ar: %s). Returning existing report.",
                last_concall, last_ar
            )
            return existing_report

        delta_blocks = []
        new_quotes = []
        for period, source, revenue, ebitda, capex, risks, sentiment, quotes_json in new_rows:
            delta_blocks.append(
                f"--- NEW: {source} | {period} ---\n"
                f"Revenue: {revenue}\n"
                f"EBITDA: {ebitda}\n"
                f"Capex: {capex}\n"
                f"Risks: {risks}\n"
                f"Sentiment: {sentiment}"
            )
            if quotes_json:
                try:
                    new_quotes.extend(json.loads(quotes_json))
                except (json.JSONDecodeError, TypeError):
                    pass

        delta_text = "\n\n".join(delta_blocks)
        new_periods = [row[0] for row in new_rows]

        system_prompt = (
            "You are an equity research report update API. "
            "You receive an existing report and new quarterly data. "
            "Update ONLY fields meaningfully affected by the new data. "
            "Append new quarter data to existing sections — do not replace historical data. "
            "Preserve all exact numbers from the existing report. Add new numbers verbatim. "
            "Preserve company_moats unless new data explicitly changes the competitive picture. "
            "Output the complete updated report as valid minified JSON in the exact same schema."
        )
        user_prompt = (
            f"Update the equity research report for {ticker_upper}.\n\n"
            "Fields to update if new data warrants it: financials_and_notes_analysis, "
            "promise_evaluator_matrix, management_sentiment_synthesis, red_flags, "
            "investment_score, summary_text.\n"
            "Fields to preserve unless explicitly contradicted: company_moats.\n"
            "Append new chunk IDs to all _referenced_chunks arrays — do not replace existing IDs.\n\n"
            "Output schema must match exactly:\n"
            "{\n"
            '  "summary_text": "...", "financials_and_notes_analysis": "...",\n'
            '  "financials_referenced_chunks": [...], "management_sentiment_synthesis": "...",\n'
            '  "sentiment_referenced_chunks": [...], "company_moats": "...",\n'
            '  "moats_referenced_chunks": [...], "red_flags": "...",\n'
            '  "red_flags_referenced_chunks": [...], "promise_evaluator_matrix": "...",\n'
            '  "promise_matrix_referenced_chunks": [...], "investment_score": "..."\n'
            "}\n\n"
            f"Existing Report:\n{existing_report.model_dump_json()}\n\n"
            f"New Quarter Delta ({', '.join(new_periods)}):\n{delta_text}\n\n"
            f"New Verbatim Management Quotes (use directly in promise matrix):\n{json.dumps(new_quotes, indent=2)}"
        )
        estimated_tokens = int((len(system_prompt) + len(user_prompt)) / 3) + 1500

        while True:
            try:
                logger.info("⏳ [Incremental Update] %s — new periods: %s...", ticker_upper, new_periods)
                await self.limiter.consume(estimated_tokens)
                completion = await self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    model=self.model,
                    response_format={"type": "json_object"},
                    temperature=0.0
                )
                actual_tokens = completion.usage.total_tokens
                await self.limiter.correct_from_actual_usage(estimated_tokens, actual_tokens)
                updated_report = FinalInvestmentReportSchema(
                    **json.loads(completion.choices[0].message.content.strip())
                )

                # Strip any fabricated/unrecognized chunk citations before saving
                valid_ids = await self._get_valid_chunk_ids(ticker_upper)
                updated_report = self._sanitize_report_chunks(updated_report, valid_ids, ticker_upper)

                new_concall_periods = [r[0] for r in new_rows if r[1].upper() == "CONCALL"]
                new_ar_periods = [r[0] for r in new_rows if r[1].upper() == "AR"]
                concall_checkpoint = (
                    max(new_concall_periods, key=self._parse_period_to_datetime) if new_concall_periods else None
                )
                ar_checkpoint = (
                    max(new_ar_periods, key=self._parse_period_to_datetime) if new_ar_periods else None
                )

                await self._save_report(
                    ticker_upper, updated_report, "INCREMENTAL",
                    concall_period=concall_checkpoint, ar_period=ar_checkpoint,
                )
                logger.info(
                    "✅ [Reasoner] Incremental update applied for %s (concall through: %s, AR through: %s).",
                    ticker_upper, concall_checkpoint or 'unchanged', ar_checkpoint or 'unchanged'
                )
                return updated_report
            except groq.RateLimitError as e:
                if any(x in str(e).lower() for x in ["daily", "tpd", "rate_limit_exceeded"]):
                    await self.limiter.handle_groq_daily_limit_backoff(e)
                    continue
                raise e