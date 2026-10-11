"""
cv/trocr/ — TrOCR book-title OCR reader.

Phase 5 implementation.

Exports
-------
BookTextReader   : Main TrOCR reader with catalog matching.
BookReadResult   : Result dataclass for a single OCR pass.
BOOK_CATALOG     : Built-in local catalog (5 titles).
UNCERTAIN_LABEL  : Fallback display string for low-confidence reads.
"""

from cv.trocr.book_text_reader import (
    BookTextReader,
    BookReadResult,
    BOOK_CATALOG,
    UNCERTAIN_LABEL,
)

__all__ = [
    "BookTextReader",
    "BookReadResult",
    "BOOK_CATALOG",
    "UNCERTAIN_LABEL",
]
