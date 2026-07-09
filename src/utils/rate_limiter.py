import asyncio
import logging
import time
import re
import groq
import openai

logger = logging.getLogger(__name__)


class AsyncTokenBucketLimiter:
    def __init__(self, max_tokens: int, refill_rate_per_sec: float, max_requests: int = 30):
        self.capacity = max_tokens
        self.refill_rate = refill_rate_per_sec
        self.tokens = max_tokens

        self.max_requests = max_requests
        self.requests_capacity = float(max_requests)
        self.requests_tokens = float(max_requests)
        self.requests_refill_rate = max_requests / 60.0

        self.last_update = time.monotonic()
        self.lock = asyncio.Lock()

    async def correct_from_actual_usage(self, estimated: int, actual: int):
        delta = actual - estimated
        if delta > 0:
            async with self.lock:
                self.tokens = max(0, self.tokens - delta)

    async def _refill(self):
        now = time.monotonic()
        elapsed = now - self.last_update
        self.last_update = now
        self.tokens = min(self.capacity, self.tokens + (elapsed * self.refill_rate))
        self.requests_tokens = min(
            self.requests_capacity,
            self.requests_tokens + (elapsed * self.requests_refill_rate)
        )

    async def consume(self, token_count: int):
        while True:
            async with self.lock:
                await self._refill()
                if self.tokens >= token_count and self.requests_tokens >= 1.0:
                    self.tokens -= token_count
                    self.requests_tokens -= 1.0
                    return True

                wait_time_tokens = (
                    (token_count - self.tokens) / self.refill_rate
                    if self.tokens < token_count else 0.0
                )
                wait_time_requests = (
                    (1.0 - self.requests_tokens) / self.requests_refill_rate
                    if self.requests_tokens < 1.0 else 0.0
                )
                target_wait = max(wait_time_tokens, wait_time_requests, 0.1)
                reason = "tokens" if wait_time_tokens >= wait_time_requests else "requests/min"
                logger.debug(
                    "[RateLimiter] Bucket depleted (%s). Waiting %.2fs "
                    "(tokens: %.0f/%.0f, requests: %.2f/%.0f).",
                    reason, target_wait,
                    self.tokens, self.capacity,
                    self.requests_tokens, self.requests_capacity,
                )
            await asyncio.sleep(target_wait)

    async def handle_groq_daily_limit_backoff(self, error: groq.RateLimitError):
        """Parses Groq's TPD error message for the exact wait time and sleeps."""
        error_msg = error.message
        logger.error("🚨 [GROQ DAILY LIMIT] TPD limit hit during execution.")
        match = re.search(r'try again in (?:(\d+)h)?(?:(\d+)m)?([\d.]+)s', error_msg)
        if match:
            hours = int(match.group(1)) if match.group(1) else 0
            minutes = int(match.group(2)) if match.group(2) else 0
            seconds = float(match.group(3)) if match.group(3) else 0.0
            total_backoff_seconds = (hours * 3600) + (minutes * 60) + seconds + 5.0
            logger.warning(
                "💤 [Groq Backoff] Parsed cool-down: %dh %dm %.2fs. "
                "Freezing workers for %.2fs...",
                hours, minutes, seconds, total_backoff_seconds
            )
            await asyncio.sleep(total_backoff_seconds)
        else:
            fallback_minutes = 35
            logger.warning(
                "Could not parse Groq backoff window. "
                "Defaulting to %d minute freeze...", fallback_minutes
            )
            await asyncio.sleep(fallback_minutes * 60)
        logger.info("🚀 [Groq Backoff Concluded] Retrying...")

    async def handle_openrouter_rate_limit(self, error: openai.RateLimitError):
        """
        Handles OpenRouter 429 rate limit errors. Attempts to parse a retry-after
        duration from the error message; falls back to a 60-second window (one full
        RPM reset cycle) if the duration can't be parsed.
        """
        error_msg = str(error)
        logger.error("🚨 [OPENROUTER LIMIT] Rate limit hit during execution.")
        match = re.search(
            r'(?:retry after|try again in|wait)\s*(\d+(?:\.\d+)?)\s*s',
            error_msg, re.IGNORECASE
        )
        if match:
            wait_seconds = float(match.group(1)) + 2.0
            logger.warning("💤 [OpenRouter Backoff] Waiting %.2fs as indicated.", wait_seconds)
            await asyncio.sleep(wait_seconds)
        else:
            logger.warning(
                "💤 [OpenRouter Backoff] Could not parse wait time from error. "
                "Defaulting to 60s (one RPM window)."
            )
            await asyncio.sleep(60.0)
        logger.info("🚀 [OpenRouter Backoff Concluded] Retrying...")