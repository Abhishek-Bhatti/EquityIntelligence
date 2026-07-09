import logging
import httpx
from datetime import datetime
from bs4 import BeautifulSoup
from src.database.connection import DB_PATH, get_db_connection

logger = logging.getLogger(__name__)


class CacheManager:
    def __init__(self):
        """
        Pure Real-Time Web Mismatch Cache Manager.
        Invalidates local analytical pipelines only when a new transcript/report date
        shows up on the Screener webserver vs what is currently cached in raw_data.db.
        """
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

    async def get_latest_local_document_date(self, ticker: str) -> datetime | None:
        """Pulls all staged periods for this ticker and returns the most recent parsed date."""
        async with get_db_connection() as db:
            async with db.execute("""
                SELECT DISTINCT period
                FROM raw_text_staging
                WHERE UPPER(ticker) = ?
            """, (ticker.upper(),)) as cursor:
                rows = await cursor.fetchall()

        parsed_dates = []
        for (period,) in rows:
            parsed = self._parse_period_string(period)
            if parsed:
                parsed_dates.append(parsed)
        return max(parsed_dates) if parsed_dates else None

    def _parse_period_string(self, period: str) -> datetime | None:
        """Converts stored period labels like 'FEB_2026' or '2026' to datetime objects."""
        # Format: FEB_2026 (concalls)
        try:
            return datetime.strptime(period, "%b_%Y")
        except ValueError:
            pass

        # Format: 2026 or FY26 (annual reports)
        try:
            year = int(period.replace("FY", "20")[:4])
            return datetime(year, 1, 1)
        except (ValueError, IndexError):
            pass

        return None

    async def fetch_screener_latest_date(self, ticker: str) -> datetime | None:
        """
        Scrapes Screener.in to find the absolute latest date between the
        investor concalls block and the annual reports block.
        """
        url = f"https://www.screener.in/company/{ticker.upper()}/"
        found_dates = []

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url, headers=self.headers, timeout=12.0)
                if response.status_code != 200:
                    logger.warning("[Web Check] Unable to fetch profile for %s. HTTP Status: %s", ticker, response.status_code)
                    return None

                soup = BeautifulSoup(response.text, 'html.parser')

                target_selectors = ["div.documents.concalls", "div.documents.annual-reports"]

                for selector in target_selectors:
                    tag, _, class_name = selector.partition('.')
                    class_list = class_name.split('.')

                    container = soup.find(tag, class_=lambda x: x and all(c in x.split() for c in class_list))

                    if container:
                        first_link = container.find('a', href=True)
                        if first_link:
                            date_span = first_link.find_next('span', class_='date')
                            if not date_span:
                                date_span = first_link.find_parent().find('span', class_='date')

                            if date_span:
                                try:
                                    parsed_date = self._clean_scraped_date(date_span.text.strip())
                                    found_dates.append(parsed_date)
                                except Exception as e:
                                    logger.warning("[Parser Warning] Failed processing date string inside selector %s: %s", selector, e)

            except Exception as e:
                logger.warning("[Web Check Warning] Connection or structural layout error on Screener scraping for %s: %s", ticker, e)
                return None

        if found_dates:
            return max(found_dates)

        logger.warning("[Web Check] No valid filing dates discovered inside target document elements for %s.", ticker)
        return None

    def _clean_scraped_date(self, date_str: str) -> datetime:
        """Normalizes irregular web strings like 'May 2026' or '14 Nov 2025' to datetime objects."""
        cleaned_str = date_str.replace("th", "").replace("st", "").replace("nd", "").replace("rd", "")

        for date_pattern in ("%d %b %Y", "%b %Y", "%d %B %Y", "%B %Y", "%Y-%m-%d"):
            try:
                return datetime.strptime(cleaned_str, date_pattern)
            except ValueError:
                continue

        raise ValueError(f"Failed to match scraped date pattern representation: '{date_str}'")

    async def evaluate_cache_staleness(self, ticker: str) -> bool:
        """
        Performs direct live verification.
        Returns True if a newer document exists on the web (invalidating cache),
        or False if local db is current.
        """
        ticker_upper = ticker.upper()
        local_date = await self.get_latest_local_document_date(ticker_upper)

        if not local_date:
            logger.info("[Cache Manager] No historical timeline context exists in DB for %s. Forcing fresh pipeline pass.", ticker_upper)
            return True

        if local_date.tzinfo is not None:
            local_date = local_date.replace(tzinfo=None)

        web_date = await self.fetch_screener_latest_date(ticker_upper)
        if not web_date:
            logger.warning("[Cache Manager] Web scraper couldn't resolve online date for %s. Defaulting to safe cache hold.", ticker_upper)
            return False

        if web_date.tzinfo is not None:
            web_date = web_date.replace(tzinfo=None)

        if web_date > local_date:
            logger.info("🔄 [Cache STALE] New filing found online for %s (%s > %s).", ticker_upper, web_date.date(), local_date.date())
            return True

        logger.info("✅ [Cache VALID] Local records match or lead the web context (%s). Skipping agent pipelines.", local_date.date())
        return False