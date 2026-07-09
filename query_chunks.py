"""
query_chunks.py — quick interactive lookup for staged/parsed chunks.

Run: python query_chunks.py
Type a ticker (e.g. KIRLOSENG) to list its chunks, or a full chunk_id
(e.g. KIRLOSENG_AR_2025_008) to see that chunk's raw content.
Type 'q' or 'quit' to exit.
"""
import asyncio
from src.database.connection import get_db_connection


async def show_ticker(ticker: str):
    async with get_db_connection() as db:
        async with db.execute(
            "SELECT text_block_id, source, period, is_parsed FROM raw_text_staging "
            "WHERE UPPER(ticker) = ? ORDER BY source, period, text_block_id",
            (ticker.upper(),)
        ) as cursor:
            rows = await cursor.fetchall()
    if not rows:
        print(f"No chunks found for {ticker.upper()}.")
        return
    for block_id, source, period, is_parsed in rows:
        status = "parsed" if is_parsed else "pending"
        print(f"  {block_id}  [{source} | {period}]  ({status})")


async def show_chunk(chunk_id: str):
    async with get_db_connection() as db:
        async with db.execute(
            "SELECT raw_content, is_parsed FROM raw_text_staging WHERE text_block_id = ?",
            (chunk_id,)
        ) as cursor:
            row = await cursor.fetchone()
    if not row:
        print(f"No chunk found with id '{chunk_id}'.")
        return
    raw_content, is_parsed = row
    print(f"--- {chunk_id} (parsed={bool(is_parsed)}) ---")
    print(raw_content)


async def main():
    print("Chunk query tool. Enter a ticker or full chunk_id. 'q' to quit.")
    while True:
        query = input("\n> ").strip()
        if query.lower() in ("q", "quit", "exit"):
            print("Bye.")
            break
        if not query:
            continue
        try:
            if query.count("_") >= 2:
                await show_chunk(query.upper())
            else:
                await show_ticker(query)
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())