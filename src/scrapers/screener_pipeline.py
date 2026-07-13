import asyncio
import base64
import logging
import re
import io
import json
import httpx
import fitz
import pytesseract
import openai
from collections import Counter
from PIL import Image
from docx import Document
from openai import AsyncOpenAI
from bs4 import BeautifulSoup
from src.database.connection import get_db_connection
from src.utils.rate_limiter import AsyncTokenBucketLimiter

logger = logging.getLogger(__name__)


class ScraperPipeline:
    PAGE_WINDOW_CHAR_THRESHOLD = 8000
    TWO_UP_TOLERANCE = 0.15
    MIN_PAGE_TEXT_CHARS = 100
    MIN_FINANCIAL_PAGE_CHARS = 300
    CONCALL_FETCH_LIMIT = 4

    # ToC vision targeting config
    AR_TOC_TARGETING_ENABLED = True
    AR_PAGE_HIT_BUFFER = 15
    TOC_SCAN_PAGE_LIMIT = 25
    ANCHOR_SCAN_PAGE_LIMIT = 60
    ANCHOR_MIN_SUPPORT = 3
    TOC_FALLBACK_MIN_PAGE_RATIO = 0.10
    TOC_MIN_CONFIDENCE = 0.5

    # OpenRouter model identifiers — single constant for easy swapping
    MAP_MODEL = "google/gemma-4-31b-it"
    VISION_MODEL = "google/gemma-4-31b-it"

    # Matches a standalone page number in a page header or footer line
    _PAGE_NUM_PATTERN = re.compile(
        r'(?:^|\bpage\b[:\s]*)(\d{1,3})\s*$', re.IGNORECASE
    )

    def __init__(self, ticker: str, openrouter_client: AsyncOpenAI,
                 openrouter_limiter: AsyncTokenBucketLimiter):
        self.ticker = ticker.upper().strip()
        self.base_url = f"https://www.screener.in/company/{self.ticker}/"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        self.openrouter_client = openrouter_client
        self.openrouter_limiter = openrouter_limiter
        self._map_semaphore = asyncio.Semaphore(3)

    # ----------------------------------------------------------------
    # Utility helpers
    # ----------------------------------------------------------------

    def _sanitize_text(self, text: str) -> str:
        """Replaces common PDF unicode artifacts with ASCII equivalents."""
        replacements = {
            '\u201c': '"', '\u201d': '"',
            '\u2018': "'", '\u2019': "'",
            '\u2013': '-', '\u2014': '--',
            '\u2026': '...', '\u00a0': ' ',
        }
        for char, replacement in replacements.items():
            text = text.replace(char, replacement)
        return text.encode('ascii', errors='ignore').decode('ascii')

    def _parse_json_response(self, text: str) -> str:
        """
        Strips markdown code fences from an LLM response and returns a clean
        JSON string. Validates parseability — raises json.JSONDecodeError if
        the content isn't valid JSON even after stripping, so callers can catch
        and return an empty schema instead of storing garbage.
        """
        cleaned = re.sub(
            r'^```(?:json)?\s*|\s*```$', '', text.strip(), flags=re.MULTILINE
        ).strip()
        json.loads(cleaned)  # validate only — raises on malformed JSON
        return cleaned

    # ----------------------------------------------------------------
    # Page text extraction
    # ----------------------------------------------------------------

    def _extract_page_text(self, page: fitz.Page, clip: fitz.Rect | None = None) -> str:
        """Extracts text from a PDF page; falls back to pytesseract if yield is low."""
        text = (page.get_text("text", clip=clip) if clip else page.get_text()).strip()
        if len(text) < self.MIN_PAGE_TEXT_CHARS:
            logger.debug("[OCR] Low text yield (%d chars). Triggering pytesseract fallback...", len(text))
            pix = page.get_pixmap(dpi=150, clip=clip) if clip else page.get_pixmap(dpi=150)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text = pytesseract.image_to_string(img).strip()
        return text

    def _detect_two_up_pages_sync(self, doc: fitz.Document) -> set[int]:
        """
        Flags PDF pages that are actually two logical pages side-by-side, using
        an aspect-ratio signature self-calibrated to this document's own minimum
        (assumed single-page) ratio. No gutter/whitespace check — that proved
        unreliable on dense multi-column spreads with no visible gap between
        halves, so aspect ratio alone is trusted; false positives on rare wide
        infographic pages are an accepted tradeoff.
        """
        if doc.page_count == 0:
            return set()

        aspects = [doc[i].rect.width / doc[i].rect.height for i in range(doc.page_count)]
        min_aspect = min(aspects)
        target = min_aspect * 2
        lower, upper = target * (1 - self.TWO_UP_TOLERANCE), target * (1 + self.TWO_UP_TOLERANCE)

        two_up = {i for i, a in enumerate(aspects) if lower <= a <= upper}
        if two_up:
            logger.info("[Layout] Detected %d two-up spread page(s) in this document.", len(two_up))
        return two_up

    def _build_page_layout(self, doc: fitz.Document) -> list:
        """
        Builds the reading-order list of virtual page keys for this document.
        Each entry is either an int (a normal single PDF page) or a tuple
        (pdf_index, 'L'/'R') for one half of a detected 2-up spread. Every
        downstream page-reading function iterates this list instead of assuming
        one logical page per PDF index.
        """
        two_up_indices = self._detect_two_up_pages_sync(doc)
        layout = []
        for idx in range(doc.page_count):
            if idx in two_up_indices:
                layout.append((idx, "L"))
                layout.append((idx, "R"))
            else:
                layout.append(idx)
        return layout

    def _resolve_virtual_page_text(self, doc: fitz.Document, virtual_key) -> str:
        """Resolves one virtual page key (int, or (index,'L'/'R') tuple) to its text."""
        if isinstance(virtual_key, tuple):
            idx, half = virtual_key
            page = doc[idx]
            rect = page.rect
            if half == "L":
                clip = fitz.Rect(rect.x0, rect.y0, rect.x0 + rect.width / 2, rect.y1)
            else:
                clip = fitz.Rect(rect.x0 + rect.width / 2, rect.y0, rect.x1, rect.y1)
            return self._extract_page_text(page, clip=clip)
        else:
            return self._extract_page_text(doc[virtual_key])

    def _accumulate_page_windows_sync(self, doc: fitz.Document, page_indices: list[int] | None = None) -> list[str]:
        """
        Builds the full virtual-page layout (splitting any detected 2-up spreads
        into left/right halves) first, then optionally filters down to only the
        PDF indices named in page_indices (as produced by ToC targeting) — a
        filtered spread page still yields both its halves, since ToC targeting
        only knows about PDF indices, not which half of a spread the target
        section actually falls in.
        """
        windows = []
        current_pages = []
        current_chars = 0

        layout = self._build_page_layout(doc)
        if page_indices is not None:
            allowed = set(page_indices)
            layout = [
                key for key in layout
                if (key if isinstance(key, int) else key[0]) in allowed
            ]

        for virtual_key in layout:
            page_text = self._resolve_virtual_page_text(doc, virtual_key)
            if not page_text:
                continue
            if len(page_text.strip()) < self.MIN_FINANCIAL_PAGE_CHARS:
                logger.debug("[Filter] Page %s skipped — %d chars.", virtual_key, len(page_text.strip()))
                continue
            current_pages.append(page_text)
            current_chars += len(page_text)

            if current_chars >= self.PAGE_WINDOW_CHAR_THRESHOLD:
                windows.append("\n\n".join(current_pages))
                current_pages = []
                current_chars = 0

        if current_pages:
            windows.append("\n\n".join(current_pages))

        return windows

    def _accumulate_paragraph_windows(self, paragraphs: list[str]) -> list[str]:
        """Accumulates flat paragraph lists into character-bounded windows (HTML/DOCX)."""
        windows = []
        current_window = []
        current_chars = 0
        for para in paragraphs:
            current_window.append(para)
            current_chars += len(para)
            if current_chars >= self.PAGE_WINDOW_CHAR_THRESHOLD:
                windows.append("\n\n".join(current_window))
                current_window = []
                current_chars = 0
        if current_window:
            windows.append("\n\n".join(current_window))
        return windows

    # ----------------------------------------------------------------
    # ToC vision targeting (AR only)
    #
    # IMPORTANT: AR_TOC_TARGETING is the ONLY way ARs get scanned. If it
    # fails at any stage below (offset detection, vision call, low
    # confidence, empty page set, or a suspiciously small page ratio),
    # we deliberately DO NOT fall back to scanning the entire AR. A full
    # scan defeats the purpose of targeting (it's exactly the cost/latency
    # we built this to avoid) — so failure here means the AR is skipped
    # entirely for this run, and the report is generated concall-only.
    # ----------------------------------------------------------------

    def _detect_page_offset_sync(self, doc: fitz.Document) -> tuple[int | None, dict[int, int]]:
        """
        Scans header/footer lines across the first ANCHOR_SCAN_PAGE_LIMIT pages
        and builds two things:

        1. A direct printed-page-number -> PDF-page-index map. This is what
            makes two-page spreads (a single PDF page carrying two printed
            numbers, e.g. "004 | ... | ... | 005") resolve correctly — we
            record every number found against that index rather than assuming
            exactly one number per PDF page.
        2. A consensus scalar offset (idx - num), used only as a fallback for
            printed numbers that never appear directly within the scan window.

        Returns (offset_or_None, page_number_map). offset is None if no single
        offset value clears ANCHOR_MIN_SUPPORT — that no longer means "abort,"
        since the direct map alone may still resolve every page we need.
        """
        offsets: Counter = Counter()
        page_number_map: dict[int, int] = {}
        limit = min(self.ANCHOR_SCAN_PAGE_LIMIT, doc.page_count)

        for idx in range(limit):
            lines = [l.strip() for l in doc[idx].get_text().splitlines() if l.strip()]
            if not lines:
                continue
            # Check first AND last line independently — a spread page typically
            # carries the left page's number in one corner, the right page's
            # in the other, both of which belong to this same PDF index.
            for line in (lines[0], lines[-1]):
                if len(line) > 20:
                    continue
                m = self._PAGE_NUM_PATTERN.search(line)
                if not m:
                    continue
                num = int(m.group(1))
                if num <= 0:
                    continue
                page_number_map[num] = idx
                offsets[idx - num] += 1

        if not offsets:
            return None, page_number_map

        best_offset, support = offsets.most_common(1)[0]
        resolved_offset = best_offset if support >= self.ANCHOR_MIN_SUPPORT else None
        return resolved_offset, page_number_map

    def _render_pages_as_b64_sync(self, doc: fitz.Document) -> list[str]:
        """Renders the first TOC_SCAN_PAGE_LIMIT pages to JPEG base64 strings at 150 DPI."""
        images = []
        limit = min(self.TOC_SCAN_PAGE_LIMIT, doc.page_count)
        for idx in range(limit):
            pix = doc[idx].get_pixmap(dpi=150)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            images.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
        return images

    async def _find_toc_pages_vision(self, doc: fitz.Document) -> dict | None:
        """
        Sends the first TOC_SCAN_PAGE_LIMIT page images to the OpenRouter vision
        model and asks it to return the printed page numbers of financially
        relevant sections directly — no category mapping, just the raw numbers.

        Returns {"pages": [int, ...], "confidence": float} or None on failure.
        Retries up to 3 times on rate limit errors before giving up.
        """
        page_images = await asyncio.to_thread(self._render_pages_as_b64_sync, doc)

        content: list[dict] = [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            for b64 in page_images
        ]
        content.append({
            "type": "text",
            "text": (
                f"These are the first {len(page_images)} pages of an Indian corporate "
                "annual report. Locate the table of contents page(s) and extract the "
                "printed page numbers (the numbers shown in the document itself, not "
                "the image position) for every section containing: financial statements, "
                "balance sheet, profit and loss account, notes to accounts, risk factors, "
                "management discussion and analysis, and outlook or strategy.\n\n"
                "Return ONLY valid JSON — no markdown, no code fences, no explanation:\n"
                '{"pages": [34, 88, 89, 90, 142, 143], "confidence": 0.95}\n\n'
                '"pages" must be an array of integers representing printed page numbers. '
                '"confidence" is your certainty from 0.0 to 1.0 that these numbers are '
                "correct as printed in the document."
            )
        })

        # Gemma uses fixed ~256-300 tokens per image tile
        estimated_tokens = (len(page_images) * 300) + 600

        for attempt in range(3):
            try:
                await self.openrouter_limiter.consume(estimated_tokens)
                response = await self.openrouter_client.chat.completions.create(
                    model=self.VISION_MODEL,
                    messages=[{"role": "user", "content": content}],
                    temperature=0.0,
                    extra_headers={
                        "HTTP-Referer": "https://github.com/alphaquant",
                        "X-Title": "AlphaQuant",
                    },
                )
                actual_tokens = (
                    response.usage.total_tokens if response.usage else estimated_tokens
                )
                await self.openrouter_limiter.correct_from_actual_usage(
                    estimated_tokens, actual_tokens
                )
                raw = response.choices[0].message.content.strip()
                result = json.loads(self._parse_json_response(raw))
                # Coerce pages to a clean list of ints — guards against "34" strings
                result["pages"] = [
                    int(p) for p in result.get("pages", [])
                    if str(p).lstrip('-').isdigit()
                ]
                return result
            except openai.RateLimitError as e:
                logger.warning(
                    "[%s] OpenRouter rate limit on vision ToC (attempt %d/3).",
                    self.ticker, attempt + 1
                )
                await self.openrouter_limiter.handle_openrouter_rate_limit(e)
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(
                    "[%s] Vision ToC JSON parse failure: %s. "
                    "AR full-page scan intentionally avoided — skipping AR for this run.",
                    self.ticker, e
                )
                return None
            except Exception as e:
                logger.error(
                    "[%s] Vision ToC call failed: %s. "
                    "AR full-page scan intentionally avoided — skipping AR for this run.",
                    self.ticker, e
                )
                return None
        logger.error(
            "[%s] Vision ToC exhausted 3 retries. "
            "AR full-page scan intentionally avoided — skipping AR for this run.",
            self.ticker
        )
        return None

    async def _get_ar_target_pages(self, doc: fitz.Document) -> list[int] | None:
        """
        Orchestrates vision ToC targeting for annual reports.
        Returns a sorted list of 0-indexed PDF page indices to scan,
        or None at any failure point to trigger a full-document scan.
        """
        offset, page_number_map = await asyncio.to_thread(self._detect_page_offset_sync, doc)
        if offset is None and not page_number_map:
            logger.info(
                "[%s] Could not determine a reliable page-number offset. "
                "Falling back to full scan.", self.ticker
            )
            return None

        result = await self._find_toc_pages_vision(doc)
        if not result:
            return None

        printed_pages = result.get("pages") or []
        confidence = result.get("confidence", 0.0)

        if not printed_pages or confidence < self.TOC_MIN_CONFIDENCE:
            logger.info(
                "[%s] Vision ToC confidence too low (%.2f) or no pages returned. "
                "Falling back to full scan.", self.ticker, confidence
            )
            return None

        # target_pages MUST be built before anything below references it
        def _resolve(p: int) -> int | None:
            """Direct map wins; scalar offset is the fallback for unmapped pages."""
            if p in page_number_map:
                return page_number_map[p]
            if offset is not None:
                idx = p + offset
                if 0 <= idx < doc.page_count:
                    return idx
            return None

        resolved_starts = sorted(set(
            pdf_idx
            for p in printed_pages
            if isinstance(p, int)
            for pdf_idx in (_resolve(p),)
            if pdf_idx is not None
        ))

        # Each resolved index is a section START. Span from each start to the
        # page before the next section begins — this naturally captures the full
        # section body without needing to know its length. For the final section
        # (no next start to bound it) use AR_PAGE_HIT_BUFFER as the trailing cap.
        target_pages_set: set[int] = set()
        for i, start in enumerate(resolved_starts):
            if i + 1 < len(resolved_starts):
                end = min(resolved_starts[i + 1] - 1, doc.page_count - 1)
            else:
                end = min(start + self.AR_PAGE_HIT_BUFFER, doc.page_count - 1)
            target_pages_set.update(range(start, end + 1))
        target_pages = sorted(target_pages_set)

        if not target_pages:
            logger.info(
                "[%s] No valid PDF indices after offset application. "
                "Falling back to full scan.", self.ticker
            )
            return None

        ratio = len(target_pages) / doc.page_count
        if ratio < self.TOC_FALLBACK_MIN_PAGE_RATIO:
            logger.error(
                "[%s] Targeted page set too small (%d/%d pages, %.0f%%) — likely a resolution error. "
                "Pages: %s",
                self.ticker, len(target_pages), doc.page_count, ratio * 100, target_pages
            )
            return None

        logger.info(
            "[%s] Vision ToC targeting locked on %d printed pages "
            "(confidence=%.2f) — scanning %d/%d pages.",
            self.ticker, len(printed_pages), confidence,
            len(target_pages), doc.page_count
        )
        return target_pages
    # ----------------------------------------------------------------
    # PDF / HTML / DOCX extraction entry points
    # ----------------------------------------------------------------

    async def _extract_windows_from_pdf_async(
        self, content: bytes, source_type: str = "") -> list[str]:
        """
        Opens a PDF and extracts character-bounded text windows.

        For AR documents, ToC targeting is mandatory when enabled: if it
        fails for any reason, the AR is skipped entirely (empty windows
        returned) rather than falling back to scanning the whole document.
        Non-AR documents (concalls) are always scanned in full, since they
        have no targeting step and are typically short enough not to need one.
        """
        doc = await asyncio.to_thread(fitz.open, stream=content, filetype="pdf")
        try:
            if self.AR_TOC_TARGETING_ENABLED and source_type.upper() == "AR":
                target_pages = await self._get_ar_target_pages(doc)
                if target_pages is None:
                    logger.error(
                        "[%s] AR ToC targeting failed — AR full scan avoided by design. "
                        "No AR text will be extracted for this run.",
                        self.ticker
                    )
                    return []
                return await asyncio.to_thread(
                    self._accumulate_page_windows_sync, doc, target_pages
                )
            # Non-AR (or targeting disabled): full scan is the intended behavior.
            return await asyncio.to_thread(
                self._accumulate_page_windows_sync, doc, None
            )
        finally:
            doc.close()

    def _extract_windows_from_html(self, html_text: str) -> list[str]:
        soup = BeautifulSoup(html_text, 'html.parser')
        for tag in soup(["script", "style", "nav", "header", "footer"]):
            tag.decompose()
        paragraphs = [
            p.get_text().strip()
            for p in soup.find_all(['p', 'div', 'li', 'td', 'span'])
            if p.get_text().strip()
        ]
        return self._accumulate_paragraph_windows(paragraphs)

    def _extract_windows_from_docx(self, content: bytes) -> list[str]:
        doc = Document(io.BytesIO(content))
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        return self._accumulate_paragraph_windows(paragraphs)

    async def _extract_windows_from_response(
        self, response: httpx.Response, source_type: str = "") -> list[str] | None:
        content_type = response.headers.get("Content-Type", "").lower()
        if "application/pdf" in content_type:
            return await self._extract_windows_from_pdf_async(response.content, source_type)
        elif "text/html" in content_type:
            return self._extract_windows_from_html(response.text)
        elif "wordprocessingml" in content_type or content_type.endswith(".docx"):
            return await asyncio.to_thread(self._extract_windows_from_docx, response.content)
        else:
            logger.warning("[Scraper] Unsupported content type: '%s'. Skipping.", content_type)
            return None

# ----------------------------------------------------------------
# Map calls — OpenRouter Gemma 4 31B
# ----------------------------------------------------------------

    def _get_map_system_prompt(self) -> str:
        return (
            "You are a financial data extraction API processing a window of Indian corporate filing text. "
            "Strip all boilerplate, operator introductions, greetings, legal disclaimers, and safe harbor statements. "
            "Extract only hard financial facts, numerical data, and management commentary. "
            "CRITICAL NUMERICAL FIDELITY: Preserve all numbers exactly as written in the source text. "
            "If the text says 'INR 847 crore', write 'INR 847 crore'. Never round, approximate, or paraphrase any figure. "
            "For raw_management_quotes: extract verbatim forward-looking statements, promises, or guidance "
            "made by management. These must be word-for-word from the source text. "
            "Leave fields as empty strings or empty arrays if not present in this window. "
            "Output ONLY valid minified JSON. No markdown, no code fences, no explanation."
        )

    async def _map_window_guarded(self, window_text: str, window_id: str) -> str:
        async with self._map_semaphore:
            return await self._map_window(window_text, window_id)

    async def _map_window(self, window_text: str, window_id: str) -> str:
        """
        Fires one extraction call against the OpenRouter map model for a single
        page window. Returns a partial CorporateInsightSchema as a JSON string.
        On unrecoverable failure, returns an empty schema so the pipeline continues.
        """
        clean_text = self._sanitize_text(window_text)
        system_prompt = self._get_map_system_prompt()
        user_prompt = (
            f"Extract financial data from this corporate text window. Window ID: {window_id}\n\n"
            "Return ONLY this exact JSON, no markdown:\n"
            "{\n"
            ' "revenue_growth_guidance": "Exact revenue/volume/order book data, or empty string",\n'
            f' "revenue_source_chunks": ["{window_id}"] if revenue data found, else [],\n'
            ' "ebitda_margin_trend": "Exact margin/input cost figures, or empty string",\n'
            f' "ebitda_source_chunks": ["{window_id}"] if margin data found, else [],\n'
            ' "capex_plans": "Exact capex/capacity/project figures, or empty string",\n'
            f' "capex_source_chunks": ["{window_id}"] if capex data found, else [],\n'
            ' "key_risks_mentioned": "Risk/regulatory/debt data, or empty string",\n'
            f' "risk_source_chunks": ["{window_id}"] if risk data found, else [],\n'
            ' "management_sentiment": "BULLISH/CAUTIOUS/NEUTRAL with one-sentence rationale, or empty string",\n'
            ' "raw_management_quotes": [\n'
            f'   {{"chunk_id": "{window_id}", "quote": "verbatim forward-looking statement or promise"}}\n'
            ' ]\n'
            "}\n\n"
            f'Text Window:\n"""{clean_text}"""'
        )
        estimated_tokens = int((len(system_prompt) + len(user_prompt)) / 3) + 1500
        empty_schema = json.dumps({
            "revenue_growth_guidance": "", "revenue_source_chunks": [],
            "ebitda_margin_trend": "", "ebitda_source_chunks": [],
            "capex_plans": "", "capex_source_chunks": [],
            "key_risks_mentioned": "", "risk_source_chunks": [],
            "management_sentiment": "", "raw_management_quotes": []
        })

        MAX_RETRIES = 5
        attempt = 0
        while attempt < MAX_RETRIES:
            attempt += 1
            try:
                await self.openrouter_limiter.consume(estimated_tokens)
                response = await self.openrouter_client.chat.completions.create(
                    model=self.MAP_MODEL,
                    messages=[
                        {"role": "system", "content": [
                            {"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}
                        ]},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.0,
                    extra_headers={
                        "HTTP-Referer": "https://github.com/alphaquant",
                        "X-Title": "AlphaQuant",
                    },
                )
                actual_tokens = (
                    response.usage.total_tokens if response.usage else estimated_tokens
                )
                await self.openrouter_limiter.correct_from_actual_usage(
                    estimated_tokens, actual_tokens
                )
                raw = response.choices[0].message.content.strip()
                return self._parse_json_response(raw)
            except openai.RateLimitError as e:
                logger.warning(
                    "[Map] OpenRouter rate limit for %s (attempt %d/%d).",
                    window_id, attempt, MAX_RETRIES
                )
                if attempt >= MAX_RETRIES:
                    logger.error(
                        "[Map Failure] %s exhausted %d retries on rate limiting. Storing empty schema.",
                        window_id, MAX_RETRIES
                    )
                    return empty_schema
                await self.openrouter_limiter.handle_openrouter_rate_limit(e)
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(
                    "[Map] JSON parse failure for %s: %s. Storing empty schema.",
                    window_id, e
                )
                return empty_schema
            except Exception as e:
                error_str = repr(e).encode('utf-8', errors='ignore').decode('utf-8')
                logger.warning(
                    "[Map Warning] Map call failed for %s: %s. Storing empty schema.",
                    window_id, error_str
                )
                return empty_schema

        # Should be unreachable, but guards against falling through the loop
        logger.error("[Map Failure] %s exhausted all retries unexpectedly. Storing empty schema.", window_id)
        return empty_schema

    # ----------------------------------------------------------------
    # Document download + dispatch
    # ----------------------------------------------------------------

    async def _download_and_map_document(
        self, client_http: httpx.AsyncClient, url: str, source_type: str, period: str):
        """Downloads one document, extracts windows, fires concurrent map calls."""
        async with get_db_connection() as db:
            async with db.execute(
                "SELECT 1 FROM raw_text_staging "
                "WHERE ticker = ? AND source = ? AND period = ? LIMIT 1",
                (self.ticker, source_type, period)
            ) as cursor:
                if await cursor.fetchone():
                    logger.info(
                        "⏭️ [%s] %s %s already staged. Skipping.",
                        self.ticker, source_type, period
                    )
                    return
        try:
            logger.info("[%s] Fetching %s (%s)...", self.ticker, source_type, period)
            response = await client_http.get(url, timeout=30.0)
            if response.status_code != 200:
                logger.warning("[%s] HTTP %s. Skipping.", self.ticker, response.status_code)
                return

            windows = await self._extract_windows_from_response(response, source_type)
            if not windows:
                logger.warning(
                    "[%s] No text extracted from %s (%s). Skipping.",
                    self.ticker, source_type, period
                )
                return

            logger.info(
                "[%s] %s (%s): %d windows. Firing map calls...",
                self.ticker, source_type, period, len(windows)
            )
            map_outputs = await asyncio.gather(*[
                self._map_window_guarded(
                    window_text,
                    f"{self.ticker}_{source_type}_{period}_{str(idx).zfill(3)}"
                )
                for idx, window_text in enumerate(windows)
            ])

            async with get_db_connection() as db:
                for idx, map_json in enumerate(map_outputs):
                    text_block_id = (
                        f"{self.ticker}_{source_type}_{period}_{str(idx).zfill(3)}"
                    )
                    await db.execute("""
                        INSERT OR IGNORE INTO raw_text_staging
                        (text_block_id, ticker, source, period, raw_content, is_parsed)
                        VALUES (?, ?, ?, ?, ?, 0)
                    """, (text_block_id, self.ticker, source_type, period, map_json))
                await db.commit()
            logger.info(
                "[%s] Staged %d map outputs for %s (%s).",
                self.ticker, len(map_outputs), source_type, period
            )
        except Exception as e:
            logger.error(
                "[%s] Error processing %s (%s): %s", self.ticker, source_type, period, e
            )

    async def build_download_tasks(
        self, client: httpx.AsyncClient) -> tuple[list, list]:
        """
        Scrapes the Screener profile page once and returns two separate lists
        of coroutines — AR tasks and concall tasks — without running them.
        Callers control when each list is awaited (main.py fires AR in the
        background via ensure_future so concall reasoning doesn't wait on it).
        """
        logger.info("[%s] Hitting Screener profile page...", self.ticker)
        res = await client.get(self.base_url)
        if res.status_code != 200:
            logger.error(
                "[%s] Invalid ticker or Screener block (Status %s)",
                self.ticker, res.status_code
            )
            return [], []

        soup = BeautifulSoup(res.text, 'html.parser')
        ar_tasks = []
        concall_tasks = []

        ar_section = soup.select_one('div.documents.annual-reports')
        if ar_section:
            ar_links = ar_section.find_all('a', href=True)
            if ar_links:
                latest_ar_link = ar_links[0]
                year_match = re.search(r'(20\d{2}|FY\d{2})', latest_ar_link.text)
                year_label = year_match.group(0) if year_match else "LATEST"
                ar_tasks.append(
                    self._download_and_map_document(
                        client, latest_ar_link['href'], "AR", year_label
                    )
                )

        concall_section = soup.select_one('div.documents.concalls')
        if concall_section:
            for li in concall_section.find_all('li')[:self.CONCALL_FETCH_LIMIT]:
                link = li.find(
                    'a', class_='concall-link',
                    string=re.compile(r'Transcript', re.IGNORECASE)
                )
                if link and link.has_attr('href'):
                    date_match = re.search(
                        r'([A-Za-z]{3})\s?(20\d{2})', li.get_text().strip()
                    )
                    period_label = (
                        f"{date_match.group(1).upper()}_{date_match.group(2)}"
                        if date_match else "TRANSCRIPT"
                    )
                    concall_tasks.append(
                        self._download_and_map_document(
                            client, link['href'], "CONCALL", period_label
                        )
                    )

        if not (concall_tasks or ar_tasks):
            logger.warning("[%s] No documents found on Screener profile.", self.ticker)

        return concall_tasks, ar_tasks

    async def run_pipeline(self):
        """
        Convenience wrapper for the cache-refresh path: scrapes and runs all
        tasks together. Use build_download_tasks() directly when you need the
        concall/AR split.
        """
        async with httpx.AsyncClient(headers=self.headers, follow_redirects=True) as client:
            concall_tasks, ar_tasks = await self.build_download_tasks(client)
            all_tasks = concall_tasks + ar_tasks
            if not all_tasks:
                return
            logger.info(
                "[%s] Launching %d concurrent download+map workers...",
                self.ticker, len(all_tasks)
            )
            await asyncio.gather(*all_tasks)