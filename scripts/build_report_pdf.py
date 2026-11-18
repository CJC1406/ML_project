"""Render docs/report/TECHNICAL_REPORT.md to PDF (Markdown -> HTML -> headless Chrome).

Usage (repo root): python scripts/build_report_pdf.py      Output: docs/report/TECHNICAL_REPORT.pdf
"""
import subprocess
from pathlib import Path

import markdown

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "docs" / "report" / "TECHNICAL_REPORT.md"
HTML = REPO / "docs" / "report" / "TECHNICAL_REPORT.html"
PDF = REPO / "docs" / "report" / "TECHNICAL_REPORT.pdf"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
CSS = """
@page { size: A4; margin: 18mm 16mm; }
body { font: 10.5pt/1.45 -apple-system, "Helvetica Neue", Arial, sans-serif; color: #111; }
h1 { font-size: 20pt; margin: 0 0 4pt; } h2 { font-size: 14pt; margin: 18pt 0 6pt; border-bottom: 1px solid #ccc; padding-bottom: 2pt; }
h3 { font-size: 11.5pt; margin: 12pt 0 4pt; }
table { border-collapse: collapse; margin: 6pt 0 10pt; font-size: 9pt; page-break-inside: avoid; }
th, td { border: 1px solid #d0d0d0; padding: 3pt 6pt; } th { background: #f2f2f2; text-align: left; }
td { font-variant-numeric: tabular-nums; }
img { max-width: 100%; page-break-inside: avoid; display: block; margin: 6pt auto; }
code { font-size: 9pt; background: #f4f4f4; padding: 0 2pt; } pre { background: #f4f4f4; padding: 6pt; font-size: 8.5pt; white-space: pre-wrap; }
blockquote { color: #444; border-left: 3px solid #ccc; margin: 0; padding-left: 8pt; }
"""


def main():
    body = markdown.markdown(SRC.read_text(), extensions=["tables", "fenced_code", "toc"])
    HTML.write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>PathSnap technical report</title>"
                    f"<style>{CSS}</style></head><body>{body}</body></html>")
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    "--virtual-time-budget=5000", f"--print-to-pdf={PDF}", HTML.as_uri()],
                   check=True, capture_output=True, timeout=180)
    print(PDF)


if __name__ == "__main__":
    main()
