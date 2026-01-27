"""
Reporting Package

Clean, modular presentation layer for HTML reports, Markdown summaries, and CLI output.
"""

from .report_builder import build_html_report
from .artifacts import generate_summary_md, generate_top5_csv, generate_run_card_json

__all__ = [
    "build_html_report",
    "generate_summary_md",
    "generate_top5_csv",
    "generate_run_card_json",
]

