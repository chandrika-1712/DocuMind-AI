
"""
Short Description:
Extract and analyze text from PDF and TXT documents.

Approach:
1. Validate the uploaded file type.
2. Extract text using the appropriate reader.
3. Clean unnecessary whitespace.
4. Calculate basic document statistics.
"""

import re
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader


def extract_text(file_bytes: bytes, filename: str) -> str:
    """Extract text from a PDF or TXT file."""

    extension = Path(filename).suffix.lower()

    if extension not in {".pdf", ".txt"}:
        raise ValueError("Only PDF and TXT files are supported.")

    if not file_bytes:
        raise ValueError("The uploaded file is empty.")

    if extension == ".txt":
        text = file_bytes.decode("utf-8-sig")

    else:
        reader = PdfReader(BytesIO(file_bytes))

        if reader.is_encrypted:
            raise ValueError(
                "Password-protected PDFs are not supported."
            )

        pages = []

        for page in reader.pages:
            pages.append(page.extract_text() or "")

        text = "\n".join(pages)

    # Normalize whitespace without removing paragraph breaks.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    if not text:
        raise ValueError(
            "No readable text found. The PDF may be scanned "
            "or contain only images."
        )

    return text


def get_document_stats(text: str) -> dict:
    """Calculate basic statistics for extracted text."""

    words = re.findall(r"\b[\w'-]+\b", text)

    return {
        "characters": len(text),
        "words": len(words),
        "lines": len(text.splitlines()),
        "paragraphs": len(
            [p for p in re.split(r"\n\s*\n", text) if p.strip()]
        ),
    }
