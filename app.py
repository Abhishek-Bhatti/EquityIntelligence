import os
# Force proxy bypass for local addresses
os.environ["NO_PROXY"] = "localhost,127.0.0.1,0.0.0.0,::1"
os.environ["no_proxy"] = "localhost,127.0.0.1,0.0.0.0,::1"

import gradio as gr

# ... the rest of your code ...

import gradio as gr
import json
import asyncio
import os
import subprocess
import logging
from src.database.connection import get_db_connection, init_db
from src.utils.logging_config import setup_logging

setup_logging()
DEMO_PASSWORD = os.environ.get("DEMO_PASSWORD", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_REPO = os.environ.get("HF_REPO", "")  # e.g. "yourusername/alphaquant"
DB_PATH = os.path.join(os.path.dirname(__file__), "data", "raw_data.db")


# ----------------------------------------------------------------
# Git LFS push helper
# ----------------------------------------------------------------

def _clean_markdown_field(text: str) -> str:
    """Ensures markdown block elements (headers, bullets) get real paragraph
    breaks — a single \\n is sometimes swallowed as a soft break by renderers."""
    return text.replace("\n", "\n\n") if text else text

def build_report_markdown(report: dict) -> str:
    summary = _clean_markdown_field(report.get('summary_text', ''))
    financials = _clean_markdown_field(report.get('financials_and_notes_analysis', ''))
    moats = _clean_markdown_field(report.get('company_moats', ''))
    red_flags = _clean_markdown_field(report.get('red_flags', ''))
    promise = report.get('promise_evaluator_matrix', '')
    sentiment = _clean_markdown_field(report.get('management_sentiment_synthesis', ''))
    return (
        f"# {report['ticker']} — {report['score']}\n\n"
        f"*Generated: {report['generated_at']}*\n\n"
        f"---\n\n"
        f"## 📋 Summary\n\n{summary}\n\n"
        f"## 📝 Financials\n\n{financials}\n\n"
        f"## 🛡 Moats\n\n{moats}\n\n"
        f"## 🚨 Red Flags\n\n{red_flags}\n\n"
        f"## 🔮 Promise Matrix\n\n{promise}\n\n"
        f"## 🤝 Sentiment\n\n{sentiment}\n\n"
    )

def show_report(ticker_choice: str) -> str:
    reports = fetch_all_reports()
    for r in reports:
        if r["ticker"] == ticker_choice:
            return build_report_markdown(r)
    return "*Report not found.*"

def _configure_git_remote() -> bool:
    if not HF_TOKEN or not HF_REPO:
        return False
    authed_url = f"https://alphaquant-bot:{HF_TOKEN}@huggingface.co/spaces/{HF_REPO}"
    result = subprocess.run(
        ["git", "remote", "set-url", "hf", authed_url],
        capture_output=True, text=True,
        cwd=os.path.dirname(__file__)
    )
    return result.returncode == 0

def _push_db_to_repo(ticker: str) -> tuple[bool, str]:
    """
    Stages the SQLite DB, commits with a descriptive message, and pushes
    to the HF Space repo via Git LFS. Returns (success, message).
    Runs synchronously — called from a thread so it doesn't block the
    event loop.
    """
    cwd = os.path.dirname(os.path.abspath(__file__))

    if not _configure_git_remote():
        return False, "⚠️ HF_TOKEN or HF_REPO not set — skipping DB push."

    steps = [
        (["git", "lfs", "track", "data/raw_data.db"], "LFS track"),
        (["git", "add", ".gitattributes", DB_PATH], "git add"),
        (["git", "commit", "--allow-empty", "-m",
          f"chore: persist DB after processing {ticker}"], "git commit"),
        (["git", "push", "hf", "main"], "git push"),
    ]

    for cmd, label in steps:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
        if result.returncode != 0:
            # "nothing to commit" is not a real failure — skip gracefully
            if "nothing to commit" in result.stdout or "nothing to commit" in result.stderr:
                continue
            return False, (
                f"❌ Git step '{label}' failed:\n"
                f"stdout: {result.stdout}\nstderr: {result.stderr}"
            )

    return True, f"✅ DB pushed to repo after processing {ticker}."


# ----------------------------------------------------------------
# Report fetching + rendering (unchanged from before)
# ----------------------------------------------------------------

async def _fetch_all_reports() -> list[dict]:
    await init_db()
    async with get_db_connection() as db:
        async with db.execute("""
            SELECT ticker, investment_score, report_payload, generated_at
            FROM equity_research_reports
            ORDER BY generated_at DESC
        """) as cursor:
            rows = await cursor.fetchall()
    results = []
    for ticker, score, payload, generated_at in rows:
        report = json.loads(payload)
        results.append({
            "ticker": ticker,
            "score": score,
            "generated_at": generated_at,
            **report
        })
    return results


def fetch_all_reports():
    return asyncio.run(_fetch_all_reports())

def get_ticker_choices() -> list[str]:
    reports = fetch_all_reports()
    return [r["ticker"] for r in reports]


# ----------------------------------------------------------------
# New ticker processing — with DB push on completion
# ----------------------------------------------------------------

def process_new_ticker(ticker: str, password: str):
    if not password or password != DEMO_PASSWORD:
        yield "❌ Incorrect password."
        return
    if not ticker or not ticker.strip():
        yield "❌ Please enter a ticker symbol."
        return

    ticker = ticker.upper().strip()
    yield f"⚡ Starting pipeline for {ticker}...\n"

    from main import run_alpha_quant_pipeline

    log_lines = []

    class GradioLogHandler(logging.Handler):
        def emit(self, record):
            log_lines.append(self.format(record))

    handler = GradioLogHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    )
    logging.getLogger().addHandler(handler)

    pipeline_success = False
    try:
        asyncio.run(run_alpha_quant_pipeline(ticker))
        pipeline_success = True
        log_lines.append(f"✅ Pipeline complete for {ticker}.")
        yield "\n".join(log_lines)
    except Exception as e:
        log_lines.append(f"❌ Pipeline failed: {e}")
        yield "\n".join(log_lines)
        return
    finally:
        logging.getLogger().removeHandler(handler)

    if not pipeline_success:
        return

    # --- DB push phase ---
    yield "\n".join(log_lines) + "\n\n💾 Pushing updated DB to repo..."
    push_ok, push_msg = _push_db_to_repo(ticker)
    log_lines.append(push_msg)
    yield "\n".join(log_lines)

    if push_ok:
        yield (
            "\n".join(log_lines)
            + f"\n\n🎉 Done. Refresh the 'View Reports' tab to see {ticker}."
        )
    else:
        yield (
            "\n".join(log_lines)
            + "\n\n⚠️ Report is in the live DB but push failed — "
            "it will be lost on Space restart. Re-run the push manually."
        )


# ----------------------------------------------------------------
# Gradio UI
# ----------------------------------------------------------------

with gr.Blocks(title="AlphaQuant — Indian Equity Research") as demo:
    gr.Markdown(
        "# 📊 AlphaQuant\n"
        "### LLM-powered equity research for Indian public companies"
    )

    with gr.Tab("View Reports"):
        gr.Markdown("Select a processed ticker to view its full equity research report.")
        ticker_dropdown = gr.Dropdown(
            choices=get_ticker_choices(),
            label="Ticker",
            interactive=True
        )
        view_btn = gr.Button("Load Report")
        report_display = gr.Markdown()
        view_btn.click(
            fn=show_report,
            inputs=ticker_dropdown,
            outputs=report_display
        )

    with gr.Tab("Process New Ticker"):
        gr.Markdown(
            "⚠️ **Password required.** "
            "Pipeline takes 10–30 minutes. "
            "The database will be committed to the repo automatically on completion."
        )
        ticker_input = gr.Textbox(label="Ticker Symbol (e.g. KIRLOSENG)")
        password_input = gr.Textbox(label="Password", type="password")
        process_btn = gr.Button("Run Pipeline")
        log_output = gr.Textbox(
            label="Pipeline Log",
            lines=30,
            interactive=False
        )
        process_btn.click(
            fn=process_new_ticker,
            inputs=[ticker_input, password_input],
            outputs=log_output
        )

demo.launch(server_name="0.0.0.0", server_port=7860)