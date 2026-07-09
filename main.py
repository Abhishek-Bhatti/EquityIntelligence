import asyncio
import os
import argparse
import logging
import httpx
from groq import AsyncGroq
from src.utils.report_writer import write_report_to_txt
from openai import AsyncOpenAI
from src.database.connection import init_db
from src.scrapers.screener_pipeline import ScraperPipeline
from src.agents.parser_agent import ParserAgent
from src.agents.reasoner_agent import ReasonerAgent
from src.utils.rate_limiter import AsyncTokenBucketLimiter
from src.utils.cache_manager import CacheManager
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AlphaQuant: LLM-powered equity research pipeline for Indian public companies."
    )
    parser.add_argument(
        "ticker",
        type=str,
        help="BSE/NSE ticker symbol to analyze (e.g. TATASTEEL, SULA, KIRLOSENG)"
    )
    return parser


async def run_alpha_quant_pipeline(ticker: str):
    if not os.environ.get("OPENROUTER_API_KEY"):
        logger.error("OPENROUTER_API_KEY must be set.")
        return
    if not os.environ.get("GROQ_API_KEY_PRIMARY"):
        logger.error("GROQ_API_KEY_PRIMARY must be set.")
        return

    await init_db()

    # ----------------------------------------------------------------
    # OpenRouter — all mapping (vision ToC + window map calls)
    # ----------------------------------------------------------------
    openrouter_client = AsyncOpenAI(
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )
    openrouter_limiter = AsyncTokenBucketLimiter(
        max_tokens=200_000,           # OpenRouter has no published hard TPM; generous budget
        refill_rate_per_sec=200_000 / 60.0,
        max_requests=18,              # 20 RPM hard cap; 18 gives 2 RPM safety margin
    )

    # ----------------------------------------------------------------
    # Groq — reduce (parser) and reason calls only
    # Two separate limiters because llama-4-scout (parse) and
    # llama-3.3-70b (reason) have different per-model TPM budgets.
    # ----------------------------------------------------------------
    groq_client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY_PRIMARY"))
    groq_parse_limiter = AsyncTokenBucketLimiter(
        max_tokens=29_000,
        refill_rate_per_sec=29_000 / 60.0,
        max_requests=28,
    )
    groq_reason_limiter = AsyncTokenBucketLimiter(
        max_tokens=30_000,
        refill_rate_per_sec=30_000 / 60.0,
        max_requests=28,
    )

    ticker_upper = ticker.upper()
    reasoner = ReasonerAgent(client=groq_client, limiter=groq_reason_limiter)
    cache_manager = CacheManager()

    logger.info("[Phase 1] Checking local cache for %s...", ticker_upper)
    cached_report = await reasoner.check_local_cache(ticker_upper)

    if cached_report:
        logger.info("[Phase 1] Report found. Checking web for new filings...")
        is_stale = await cache_manager.evaluate_cache_staleness(ticker_upper)
        if is_stale:
            logger.info("🔄 [Orchestrator] New filing detected. Running incremental update pipeline...")
            pipeline = ScraperPipeline(
                ticker=ticker_upper,
                openrouter_client=openrouter_client,
                openrouter_limiter=openrouter_limiter,
            )
            parser = ParserAgent(
                groq_client=groq_client,
                groq_limiter=groq_parse_limiter,
            )
            logger.info("[Phase 2] Scraping new documents for %s...", ticker_upper)
            await pipeline.run_pipeline()
            logger.info("[Phase 3] Reducing new documents...")
            await parser.parse_unprocessed_documents(ticker=ticker_upper)
            logger.info("[Phase 4] Applying incremental update to existing report...")
            report = await reasoner.apply_incremental_update(ticker_upper, cached_report)
        else:
            logger.info("✅ [Orchestrator] Cache is current. Serving from local storage.")
            report = cached_report
    else:
        logger.info("⚡ [Orchestrator] No existing report. Running full pipeline...")
        pipeline = ScraperPipeline(
            ticker=ticker_upper,
            openrouter_client=openrouter_client,
            openrouter_limiter=openrouter_limiter,
        )
        parser = ParserAgent(
            groq_client=groq_client,
            groq_limiter=groq_parse_limiter,
        )

        logger.info("[Phase 2] Scraping documents for %s...", ticker_upper)
        async with httpx.AsyncClient(headers=pipeline.headers, follow_redirects=True) as http_client:
            concall_tasks, ar_tasks = await pipeline.build_download_tasks(http_client)

            # AR download+map fires in the background — not awaited until Phase 5.
            ar_future = asyncio.ensure_future(asyncio.gather(*ar_tasks)) if ar_tasks else None

            if concall_tasks:
                logger.info("[Phase 2] Mapping %d concall transcript(s)...", len(concall_tasks))
                await asyncio.gather(*concall_tasks)
            else:
                logger.info("[Phase 2] No concall transcripts found.")

            logger.info("[Phase 3] Reducing concall documents...")
            await parser.parse_unprocessed_documents(ticker=ticker_upper, source_filter="CONCALL")

            logger.info("[Phase 4] Generating concall-based equity research report...")
            report = await reasoner.generate_full_report(ticker=ticker_upper, source_filter="CONCALL")

            if ar_future:
                logger.info("[Phase 5] Waiting for annual report mapping to finish...")
                await ar_future
                logger.info("[Phase 6] Reducing annual report...")
                await parser.parse_unprocessed_documents(ticker=ticker_upper, source_filter="AR")
                logger.info("[Phase 7] Folding AR insights into report...")
                report = await reasoner.apply_incremental_update(
                    ticker_upper, report, source_filter="AR"
                )
            else:
                logger.info("[Phase 5] No annual report found — skipping AR fold-in.")

    logger.info("═" * 90)
    logger.info("🥇 EQUITY RESEARCH DOSSIER: %s", ticker_upper)
    logger.info("═" * 90)
    write_report_to_txt(ticker_upper, report)
    logger.info("📊 INVESTMENT SCORE: %s", report.investment_score)
    logger.info("📝 FINANCIALS:\n%s", report.financials_and_notes_analysis)
    logger.info("🔗 CHUNKS: %s", ', '.join(report.financials_referenced_chunks))
    logger.info("🛡 MOATS:\n%s", report.company_moats)
    logger.info("🔗 CHUNKS: %s", ', '.join(report.moats_referenced_chunks))
    logger.info("🤝 SENTIMENT:\n%s", report.management_sentiment_synthesis)
    logger.info("🔗 CHUNKS: %s", ', '.join(report.sentiment_referenced_chunks))
    logger.info("🔮 PROMISE MATRIX:\n%s", report.promise_evaluator_matrix)
    logger.info("🔗 CHUNKS: %s", ', '.join(report.promise_matrix_referenced_chunks))
    logger.info("🚨 RED FLAGS:\n%s", report.red_flags)
    logger.info("🔗 CHUNKS: %s", ', '.join(report.red_flags_referenced_chunks))
    logger.info("📋 SUMMARY:\n%s", report.summary_text)
    logger.info("═" * 90)


if __name__ == "__main__":
    setup_logging()
    args = build_arg_parser().parse_args()
    asyncio.run(run_alpha_quant_pipeline(args.ticker))