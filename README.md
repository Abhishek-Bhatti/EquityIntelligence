---
title: EquityIntelligence
emoji: 📊
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---
# EquityIntelligence  

An LLM-powered equity research pipeline for publicly-listed Indian companies. Given a stock ticker, it scrapes the latest concall transcripts and annual report from [Screener.in](https://www.screener.in/), extracts structured financial data through a map-reduce LLM pipeline, and synthesizes a full equity research report - financial trajectory, management sentiment, competitive moats, red flags, a promise-vs-outcome matrix, and an investment score.

```
python main.py TICKER
```

---

## 1. Full Chain of Events (Anatomy of a Run)

### Step 0 - Startup

1. `main.py` parses the CLI argument (`ticker`) and validates that `OPENROUTER_API_KEY` and `GROQ_API_KEY_PRIMARY` are set in the environment. Missing either one aborts the run immediately.
2. `init_db()` creates (or migrates) the SQLite database at `data/raw_data.db` - three tables: `raw_text_staging`, `parsed_insights`, `equity_research_reports` - plus indexes and any schema migrations needed for older DB files.
3. Two API clients and their rate limiters are constructed:
   - **OpenRouter client** (`AsyncOpenAI`, base URL pointed at OpenRouter) - used exclusively for the *mapping* stage (vision table-of-contents detection + per-window financial-data extraction). Limiter: 200k tokens/min budget, 18 requests/min cap.
   - **Groq client** (`AsyncGroq`) - used for the *reduce* stage (document-level merge) and the *reasoning* stage (final report synthesis). Two independent limiters (29k TPM for parsing, 30k TPM for reasoning) because the two Groq-hosted models draw from separate per-model quotas.
4. A `ReasonerAgent` and an `CacheManager` are instantiated.

### Phase 1 - Cache Check

- `reasoner.check_local_cache(ticker)` looks up `equity_research_reports` for an existing report.
- **If found:** `cache_manager.evaluate_cache_staleness(ticker)` is called:
  - Reads the most recent period already staged locally (`raw_text_staging`).
  - Live-scrapes the ticker's Screener.in profile page and reads the newest date shown in the concall/annual-report document blocks.
  - Compares the two dates. If the web date is newer → **stale** (Phase 2 onward runs as an *incremental* pass). If the web scrape fails → defaults to **cache-valid** (fail-safe: never triggers unnecessary spend). If equal/older → **cache-valid**, and the cached report is served as-is with zero further LLM calls.
- **If not found:** the full pipeline runs from scratch (see below).

### Phase 2 - Scrape & Download

- `ScraperPipeline.build_download_tasks()` hits the ticker's Screener profile page once, parses it with BeautifulSoup, and builds (without yet running) two lists of coroutines:
  - **Concall tasks** - up to 4 most recent transcript links.
  - **AR tasks** - the single latest annual report link.
- The AR download+map coroutine is fired immediately in the background via `asyncio.ensure_future` - it runs concurrently with everything else and is only explicitly awaited much later (Phase 5), so a slow annual-report PDF never blocks the faster concall-based report from being generated.
- The concall tasks are awaited now. Each one independently:
  1. Checks `raw_text_staging` - if this exact (ticker, source, period) is already staged, skip (idempotent resume).
  2. Downloads the document (`httpx`).
  3. Extracts text windows depending on content type - PDF, HTML, or DOCX (see §2.6 for the PDF-specific logic).
  4. Fires one **map call per window**, bounded to 3 concurrent calls via a semaphore, against OpenRouter's Gemma model - each call strips boilerplate and extracts structured financial fields + verbatim management quotes from that single window.
  5. Persists every window's JSON output into `raw_text_staging`.

### Step 2a - Annual Report Targeting (background, runs in parallel with the above)

For the AR PDF specifically, the pipeline does **not** scan the whole document - annual reports can run 150+ pages, and doing so would be slow and expensive. Instead:

1. `_detect_page_offset_sync` samples header/footer page numbers across the first 60 pages to compute the offset between "PDF page index" and "printed page number."
2. `_find_toc_pages_vision` renders the first 25 pages to JPEG and sends them to a vision-capable OpenRouter model, asking it to locate the table of contents and return the printed page numbers of financially relevant sections (financials, MD&A, risks, etc.) with a confidence score. Retries up to 3× on rate-limit errors.
3. The returned printed page numbers are mapped to PDF indices via the offset, and a ±10-page buffer is applied around each hit.
4. **Fail-closed design:** if the offset can't be determined, the vision call fails or returns low confidence, or the resulting page set is suspiciously small (<10% of the document - signaling a likely offset error), the AR is **skipped entirely for this run** rather than falling back to a full scan. This is a deliberate cost/latency guard, not an oversight - a full fallback scan would defeat the purpose of targeting in the first place.
5. If targeting succeeds, only the targeted page windows go through the same per-window map-call process as concalls.

### Phase 3 - Reduce (Concall)

- `ParserAgent.parse_unprocessed_documents(source_filter="CONCALL")` groups all unparsed staged windows by (source, period) and fires **one reduce call per document** against Groq's Llama-4-Scout model.
- The reduce prompt merges all partial window extractions for that document into a single `CorporateInsightSchema`: concatenated text fields, deduplicated chunk-ID arrays, one consensus sentiment label, and the full deduplicated list of verbatim management quotes (never paraphrased).
- Results are written to `parsed_insights` and the source rows in `raw_text_staging` are flagged `is_parsed = 1`.

### Phase 4 - Reason (Concall-only report)

- `ReasonerAgent.generate_full_report(source_filter="CONCALL")` makes two sequential Groq calls:
  1. **Factual call** - reads all parsed concall insights, produces the chronological financial trajectory and the promise-evaluator matrix, citing only chunk IDs from an explicit whitelist built from the DB.
  2. **Qualitative call** - takes the factual output as context, produces company moats, red flags, management-sentiment synthesis, the investment score, and the executive summary.
- The two JSON outputs are merged into one `FinalInvestmentReportSchema`.
- **Citation guard:** every `*_referenced_chunks` field is checked against the real set of staged chunk IDs (`_get_valid_chunk_ids`); any ID the model invented is dropped and logged rather than silently trusted.
- The report is saved to `equity_research_reports`, along with independent **concall** and **AR** checkpoint periods (so the two document types are never compared on the same timeline - a concall period like `FEB_2026` and an AR period like `2026` aren't chronologically comparable strings).

### Phase 5 - Wait for AR

- The background AR download+map task started in Phase 2 is now awaited.
- If no AR was found (or targeting failed and it was skipped), this phase logs that and the pipeline proceeds straight to completion with the concall-only report.

### Phase 6 - Reduce (AR)

- Same as Phase 3, scoped to `source_filter="AR"` - one reduce call merges all of the AR's targeted-page window extractions into a single `parsed_insights` row.

### Phase 7 - Fold AR into Report

- `ReasonerAgent.apply_incremental_update(source_filter="AR")` is the *targeted update* path (also used for stale-cache refreshes): it reads only rows newer than the last-processed checkpoint for their **own** source (AR rows only ever compared to the AR checkpoint, concall rows only to the concall checkpoint), builds a "delta" of just the new data, and asks Groq to update *only* the fields the new data actually affects - appending new figures/quotes rather than replacing history, and preserving `company_moats` unless the AR explicitly contradicts it.
- The updated report goes through the same citation-guard sanitation and is re-saved with an updated AR checkpoint (the concall checkpoint is left untouched via `COALESCE` in the upsert).

### Completion

- `write_report_to_txt()` renders the final `FinalInvestmentReportSchema` into a clean, overwritten `data/reports/{TICKER}.txt`.
- The full report (all sections + their referenced chunk IDs) is logged to console/`runtimelogs.log`.

### The Incremental-Refresh Path (cache-hit-but-stale case)

When Phase 1 finds a cached report that's stale, the flow is a shorter variant of the above: scrape → download only what's new → parse it → `apply_incremental_update()` with no `source_filter` (so it picks up whatever's new across *both* sources at once) → re-render the `.txt`. This is what makes repeat runs on the same ticker cheap: only genuinely new filings ever hit an LLM.

---

## 2. File-by-File Reference

### `main.py`
Entry point and orchestrator. Owns the CLI arg parser, environment-variable validation, construction of both API clients and all three rate limiters, and the full phase-by-phase control flow described above (cache check → scrape → map → reduce → reason → fold-in AR → write report). Contains no business logic itself - it wires the other modules together and drives the sequencing, including the "fire AR in the background, await it later" concurrency trick.

### `src/utils/cache_manager.py` - `CacheManager`
Decides whether a cached report is still valid.
- `get_latest_local_document_date` - queries `raw_text_staging` for the most recent period already stored for a ticker.
- `_parse_period_string` - normalizes stored period labels (`FEB_2026`, `2026`, `FY26`) into `datetime` objects.
- `fetch_screener_latest_date` - live-scrapes the Screener.in profile page for the newest date shown in the concall/annual-report blocks.
- `_clean_scraped_date` - normalizes messy web date strings (`"14th Nov 2025"`, `"May 2026"`, etc.) across several format patterns.
- `evaluate_cache_staleness` - the actual decision function: compares local vs. web dates and returns `True` (stale, re-run) or `False` (cache valid, serve as-is). Defaults to "cache valid" if the web check itself fails, to avoid unnecessary spend on a scraping hiccup.

### `src/utils/logging_config.py`
Single `setup_logging()` call configures the root logger once for the whole process: a console handler plus a 5 MB rotating file handler (`runtimelogs.log`, 5 backups). Idempotent - safe to import/call from multiple modules without duplicating log lines. Also quiets noisy third-party `httpx`/`httpcore` request logging so it doesn't drown out pipeline logs.

### `src/utils/rate_limiter.py` - `AsyncTokenBucketLimiter`
A dual-bucket async rate limiter shared across all calls to one provider/model:
- Tracks a **token** bucket and a **requests/min** bucket independently, since both are enforced separately by the APIs.
- `consume(token_count)` blocks (async-sleeps) until both buckets have room, computing whichever wait is longer.
- `correct_from_actual_usage(estimated, actual)` - after a real API response comes back with its true token usage, retroactively deducts the difference so estimation drift doesn't silently over- or under-throttle later calls.
- `handle_groq_daily_limit_backoff` / `handle_openrouter_rate_limit` - parse the provider's own error message for an exact "try again in Xs" wait time and sleep accordingly, falling back to a fixed cool-down (35 min / 60s respectively) if the message can't be parsed.

### `src/utils/report_writer.py`
`write_report_to_txt(ticker, report)` - pure rendering function. Takes the final Pydantic report object and writes a formatted, human-readable `.txt` file to `data/reports/{TICKER}.txt`, overwriting any prior version regardless of which pipeline path (full/incremental/cache-hit) produced it. Each section is printed with its referenced chunk IDs beneath it for traceability.

### `src/scrapers/screener_pipeline.py` - `ScraperPipeline`
The largest module; owns scraping, document extraction, and the map stage.
- `build_download_tasks` - scrapes the Screener profile page once and returns unstarted coroutine lists for AR and concall documents.
- `_download_and_map_document` - per-document orchestration: skip-if-already-staged check, download, window extraction, concurrent map calls, persistence to `raw_text_staging`.
- `_extract_windows_from_pdf_async` / `_extract_windows_from_html` / `_extract_windows_from_docx` - content-type-specific text extraction, each producing character-bounded "windows" suitable for a single LLM call.
- `_extract_page_text` - per-page PDF text extraction with a `pytesseract` OCR fallback when native text yield is too low (scanned pages).
- `_detect_page_offset_sync`, `_render_pages_as_b64_sync`, `_find_toc_pages_vision`, `_get_ar_target_pages` - the annual-report ToC-targeting subsystem described in Step 2a; deliberately fails closed (skips the AR) rather than falling back to a full scan.
- `_map_window` / `_map_window_guarded` - fires the actual OpenRouter extraction call for one window, semaphore-bounded to 3 concurrent calls; returns an empty schema on unrecoverable failure so one bad window never crashes the whole document.
- `_sanitize_text`, `_parse_json_response` - small utility helpers (unicode cleanup, stripping markdown fences before JSON parsing).

### `src/database/connection.py`
`get_db_connection()` - async context-managed `aiosqlite` connection with WAL journaling and a larger page cache set per-connection. `init_db()` - creates all three tables and their indexes on first run, and applies a list of best-effort `ALTER TABLE` migrations (wrapped in try/except) so older database files pick up new columns without a manual migration step.

### `src/agents/parser_agent.py` - `ParserAgent`
Owns the **reduce** stage.
- `CorporateInsightSchema` (Pydantic) - the per-document merged-insight shape: revenue/EBITDA/capex/risk text fields with their source chunk IDs, a single sentiment label, and the full list of verbatim management quotes.
- `_reduce_document` - merges all of one document's window-level map outputs into a single `CorporateInsightSchema` via one Groq call; skips if already reduced; persists to `parsed_insights` and flags the source rows `is_parsed = 1`.
- `parse_unprocessed_documents` - groups all unparsed staged rows by (source, period) and fires one `_reduce_document` call per document, all concurrently.

### `src/agents/reasoner_agent.py` - `ReasonerAgent`
Owns the **reasoning/synthesis** stage and all report persistence logic.
- `FinalInvestmentReportSchema` (Pydantic) - the full report shape: summary, financials, sentiment, moats, red flags, promise matrix, investment score, each paired with its referenced-chunks list.
- `check_local_cache` - Phase 1 cache lookup.
- `_factual_call` / `_qualitative_call` - the two sequential Groq calls that produce the report's numeric and qualitative halves respectively, each restricted to citing only a whitelist of real chunk IDs.
- `generate_full_report` - runs both calls, merges them, sanitizes citations, computes independent concall/AR checkpoints, and saves.
- `apply_incremental_update` - the targeted-refresh path: finds only rows newer than each source's own checkpoint, asks Groq to update only the affected report fields (append, don't replace), and re-saves with updated checkpoints.
- `_filter_valid_chunks` / `_sanitize_report_chunks` / `_get_valid_chunk_ids` - the citation-guard subsystem shared by both report-generation paths.
- `_save_report` - persists via an upsert that uses `COALESCE` on the two checkpoint columns, so updating one source's checkpoint can never accidentally null out the other's.
- `_parse_period_to_datetime` - converts stored period strings to `datetime` for ordering, with an explicit warning in its docstring that concall and AR periods are never comparable to each other.


