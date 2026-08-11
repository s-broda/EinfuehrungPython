"""OCR scanned invoice PDFs and extract company + amount into a spreadsheet."""

from __future__ import annotations

import io
import os
import re
import shutil
import sys
from pathlib import Path

import pandas as pd


def _configure_tesseract() -> None:
    """Point pytesseract at the conda Tesseract install (Windows-friendly)."""
    import pytesseract

    for tess in (
        Path(sys.prefix) / "Library" / "bin" / "tesseract.exe",
        Path(sys.prefix) / "Library" / "bin" / "tesseract",
        Path(shutil.which("tesseract") or ""),
    ):
        if tess and tess.is_file():
            pytesseract.pytesseract.tesseract_cmd = str(tess)
            break

    for td in (
        Path(sys.prefix) / "share" / "tessdata",
        Path(sys.prefix) / "Library" / "share" / "tessdata",
    ):
        if (td / "eng.traineddata").is_file():
            os.environ["TESSDATA_PREFIX"] = str(td)
            break

    libbin = Path(sys.prefix) / "Library" / "bin"
    if libbin.is_dir():
        os.environ["PATH"] = str(libbin) + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(libbin))


def ocr_pdf(path: Path, zoom: float = 2.5) -> str:
    """Render each page to an image and run Tesseract OCR."""
    import fitz  # PyMuPDF
    import pytesseract
    from PIL import Image

    doc = fitz.open(path)
    parts = []
    mat = fitz.Matrix(zoom, zoom)
    for page in doc:
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        parts.append(pytesseract.image_to_string(img, lang="eng"))
    doc.close()
    return "\n".join(parts)


# Labels vary across invoices (EN/DE/FR-ish) — keep rules simple and transparent
COMPANY_PATTERNS = [
    re.compile(r"(?im)^(?:From|Vendor(?: name)?|Rechnungssteller|Issued by)\s*:\s*(.+)$"),
]
LEGAL_FORM_RE = re.compile(r"(?i)\b(GmbH|AG|SA|Ltd\.?|LLC|Inc\.?)\b")
AMOUNT_PATTERNS = [
    re.compile(
        r"(?im)(?:TOTAL DUE|Grand Total(?:\s*\([^)]*\))?|Gesamtbetrag|Montant total TTC|Amount Due)\s*[:\-]?\s*(.+)$"
    ),
]


def parse_amount(raw: str) -> float | None:
    """Normalize US (1,234.56) and EU (1.234,56 / 1 234,56) money strings."""
    s = raw.strip()
    for token in ("EUR", "USD", "CHF", "€", "$"):
        s = s.replace(token, "")
    s = re.sub(r"[^\d,.\s]", "", s).strip().replace(" ", "")
    if not s:
        return None
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")  # 1.234,56
        else:
            s = s.replace(",", "")  # 1,234.56
    elif "," in s:
        if len(s.split(",")[-1]) == 2:
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def extract_company(text: str) -> str | None:
    for pat in COMPANY_PATTERNS:
        m = pat.search(text)
        if m:
            return m.group(1).strip()
    for line in text.splitlines():
        line = line.strip()
        if LEGAL_FORM_RE.search(line) and len(line) < 80:
            return line
    return None


def extract_amount(text: str) -> float | None:
    for pat in AMOUNT_PATTERNS:
        m = pat.search(text)
        if m:
            val = parse_amount(m.group(1))
            if val is not None:
                return val
    for line in text.splitlines():
        if re.search(r"(?i)total|betrag|amount|due|ttc|gesamt", line):
            nums = re.findall(r"[\d][\d\s.,]{2,}", line)
            if nums:
                val = parse_amount(nums[-1])
                if val is not None:
                    return val
    return None


def ocr_invoices_to_spreadsheet(
    invoice_dir: str | Path = "invoices",
    pattern: str = "inv_*.pdf",
    output_name: str = "invoice_summary.xlsx",
    zoom: float = 2.5,
) -> pd.DataFrame:
    """OCR all invoice PDFs in a folder and write company + amount to Excel.

    The PDFs are treated as scanned images (render → Tesseract → regex).
    No LLM is used.

    Returns the resulting DataFrame (also written to ``invoice_dir/output_name``).
    """
    _configure_tesseract()

    invoice_dir = Path(invoice_dir)
    if not invoice_dir.is_dir():
        raise FileNotFoundError(f"Expected invoice PDFs in {invoice_dir.resolve()}")

    rows = []
    for pdf_path in sorted(invoice_dir.glob(pattern)):
        text = ocr_pdf(pdf_path, zoom=zoom)
        company = extract_company(text)
        amount = extract_amount(text)
        rows.append({"file": pdf_path.name, "company": company, "amount": amount})
        print(f"{pdf_path.name:20}  company={company!r:30}  amount={amount}")

    df = pd.DataFrame(rows)
    xlsx = invoice_dir / output_name
    df.to_excel(xlsx, index=False)
    print()
    print(df.to_string(index=False))
    print(f"\nClean spreadsheet written to: {xlsx.resolve()}")
    return df


if __name__ == "__main__":
    ocr_invoices_to_spreadsheet()
