import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

REPORTS_DIR = os.path.join("data", "reports")


def write_report_to_txt(ticker: str, report) -> str:
    """
    Renders a FinalInvestmentReportSchema into a clean, human-readable .txt
    file and saves it to data/reports/{ticker}.txt. Overwrites any existing
    file for that ticker — the .txt always reflects the most recently
    generated or cache-served report, regardless of which pipeline path
    produced it (full run, incremental update, or straight cache hit).
    Returns the path written to.
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)
    out_path = os.path.join(REPORTS_DIR, f"{ticker.upper()}.txt")

    sections = [
        ("EQUITY RESEARCH DOSSIER", ticker.upper()),
        ("INVESTMENT SCORE", report.investment_score),
        ("FINANCIALS", report.financials_and_notes_analysis,
         report.financials_referenced_chunks),
        ("COMPANY MOATS", report.company_moats,
         report.moats_referenced_chunks),
        ("MANAGEMENT SENTIMENT", report.management_sentiment_synthesis,
         report.sentiment_referenced_chunks),
        ("PROMISE MATRIX", report.promise_evaluator_matrix,
         report.promise_matrix_referenced_chunks),
        ("RED FLAGS", report.red_flags,
         report.red_flags_referenced_chunks),
        ("SUMMARY", report.summary_text),
    ]

    lines = []
    lines.append("=" * 90)
    lines.append(f"EQUITY RESEARCH DOSSIER: {ticker.upper()}")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 90)
    lines.append("")
    lines.append(f"INVESTMENT SCORE: {report.investment_score}")
    lines.append("")

    body_sections = sections[2:]  # skip the two header entries already rendered above
    for title, content, *chunks in body_sections:
        lines.append("-" * 90)
        lines.append(title)
        lines.append("-" * 90)
        lines.append(content.strip())
        if chunks and chunks[0]:
            lines.append("")
            lines.append(f"Referenced chunks: {', '.join(chunks[0])}")
        lines.append("")

    lines.append("=" * 90)
    lines.append("SUMMARY")
    lines.append("=" * 90)
    lines.append(report.summary_text.strip())
    lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info("📄 [Report Writer] Saved clean report to %s", out_path)
    return out_path