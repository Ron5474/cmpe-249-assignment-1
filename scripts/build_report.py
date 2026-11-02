"""Render report/report.md to report/report.pdf.

  uv run --with markdown --with weasyprint scripts/build_report.py
"""
from pathlib import Path

import markdown
from weasyprint import HTML

REPORT = Path(__file__).resolve().parents[1] / "report"

CSS = """
@page { size: Letter; margin: 0.6in 0.65in; @bottom-center { content: counter(page); font-size: 8.5pt; color: #6b6a64; } }
body { font-family: "DejaVu Serif", Georgia, serif; font-size: 9.5pt; line-height: 1.33; color: #1f1f1e; }
h1 { font-size: 17pt; margin: 0 0 4pt; }
h2 { font-size: 13pt; margin: 16pt 0 5pt; border-bottom: 1px solid #d8d7d2; padding-bottom: 2pt; }
h3 { font-size: 11pt; margin: 11pt 0 4pt; }
p, li { text-align: left; margin: 4pt 0; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 8pt; font-family: "DejaVu Sans", sans-serif; font-size: 8pt; page-break-inside: avoid; }
th, td { border-bottom: 1px solid #e4e3de; padding: 2.5pt 4pt; text-align: left; vertical-align: top; }
th { border-bottom: 1.5px solid #6b6a64; }
code { font-family: "DejaVu Sans Mono", monospace; font-size: 8pt; background: #f3f2ee; padding: 0 2pt; }
pre { background: #f3f2ee; padding: 6pt; font-size: 7.5pt; line-height: 1.3; white-space: pre-wrap; page-break-inside: avoid; }
pre code { background: none; padding: 0; }
img { max-width: 88%; display: block; margin: 6pt auto; page-break-inside: avoid; }
a { color: #1c5cab; text-decoration: none; }
img[src*="failures"] { max-width: 100%; }
"""


def main():
    html = markdown.markdown((REPORT / "report.md").read_text(), extensions=["tables", "fenced_code"])
    doc = f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{html}</body></html>"
    HTML(string=doc, base_url=str(REPORT)).write_pdf(REPORT / "report.pdf")
    print(REPORT / "report.pdf")


if __name__ == "__main__":
    main()
