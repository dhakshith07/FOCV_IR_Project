"""
cv/trocr/book_text_reader.py
=============================
Phase 5 — Microsoft TrOCR integration for book title recognition.

Pipeline
--------
  BookTrack / Detection (bbox)
      ↓
  BookTextReader.read_from_frame(frame, bbox)
      ↓  crop + preprocess
  TrOCR (microsoft/trocr-base-printed)
      ↓  raw text + sequence score
  text cleaning
      ↓
  catalog fuzzy-match
      ↓
  BookReadResult  { book_id, title, raw_text, confidence, uncertain }

Local catalog (built-in, always available offline)
---------------------------------------------------
  1  Machine Learning
  2  Mathematics
  3  Database Management Systems
  4  Computer Networks
  5  Operating Systems

Low-confidence fallback
-----------------------
  If confidence < threshold → uncertain = True
  Display text: "Book identification uncertain"

Usage
-----
    reader = BookTextReader()
    reader.load()                          # downloads model once, then cached

    result = reader.read_from_frame(frame, bbox=(x1, y1, x2, y2))
    print(result.display_title)            # "Machine Learning" or
                                           # "Book identification uncertain"
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher, get_close_matches
from typing import Dict, List, Optional, Tuple

import numpy as np

from utils.logger import get_logger

log = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  Local book catalog
# ═══════════════════════════════════════════════════════════════════════════════

#: The canonical local catalog: id → title
BOOK_CATALOG: Dict[int, str] = {
    1: "Machine Learning",
    2: "Mathematics",
    3: "Database Management Systems",
    4: "Computer Networks",
    5: "Operating Systems",
}

#: Fallback display string when confidence is below threshold
UNCERTAIN_LABEL = "Book identification uncertain"

#: Default model
DEFAULT_MODEL = "microsoft/trocr-base-printed"

#: Preprocessing target size fed to TrOCR processor (width, height)
TROCR_INPUT_SIZE = (384, 384)

#: Minimum OCR confidence to trust the matched title
DEFAULT_CONFIDENCE_THRESHOLD = 0.40

#: Minimum fuzzy-match ratio to accept a catalog hit
CATALOG_MATCH_CUTOFF = 0.40


# ═══════════════════════════════════════════════════════════════════════════════
#  Result dataclass
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BookReadResult:
    """
    Result of one TrOCR + catalog-match pass on a book crop.

    Attributes
    ----------
    track_id    : ByteTrack ID (0 if called outside tracking context).
    raw_text    : Raw string produced by TrOCR before cleaning.
    clean_text  : Cleaned / normalised OCR text.
    book_id     : Matched catalog ID (None if no match or uncertain).
    title       : Matched catalog title (or UNCERTAIN_LABEL).
    confidence  : Estimated OCR confidence [0, 1].
    match_score : Fuzzy-match ratio against the best catalog entry [0, 1].
    uncertain   : True when confidence < threshold or no catalog match.
    elapsed_ms  : Wall-clock time for the TrOCR inference.
    """

    track_id:    int
    raw_text:    str
    clean_text:  str
    book_id:     Optional[int]
    title:       str
    confidence:  float
    match_score: float
    uncertain:   bool
    elapsed_ms:  float = 0.0

    @property
    def display_title(self) -> str:
        """User-facing title string — always safe to display."""
        return UNCERTAIN_LABEL if self.uncertain else self.title

    def to_dict(self) -> dict:
        return {
            "track_id":    self.track_id,
            "raw_text":    self.raw_text,
            "clean_text":  self.clean_text,
            "book_id":     self.book_id,
            "title":       self.title,
            "display_title": self.display_title,
            "confidence":  round(self.confidence, 4),
            "match_score": round(self.match_score, 4),
            "uncertain":   self.uncertain,
            "elapsed_ms":  round(self.elapsed_ms, 1),
        }

    def __repr__(self) -> str:
        status = "uncertain" if self.uncertain else f"id={self.book_id}"
        return (
            f"BookReadResult(track={self.track_id}, "
            f"{status}, title={self.title!r}, "
            f"conf={self.confidence:.2f}, match={self.match_score:.2f})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  BookTextReader
# ═══════════════════════════════════════════════════════════════════════════════

class BookTextReader:
    """
    TrOCR-powered book title reader with local catalog matching.

    Parameters
    ----------
    model_name           : HuggingFace model ID (default: microsoft/trocr-base-printed).
    device               : 'cpu' or 'cuda'.
    confidence_threshold : Minimum sequence-score to trust OCR output [0, 1].
    catalog              : Dict[int, str] of book_id → title. Defaults to BOOK_CATALOG.
    max_new_tokens       : Maximum generated token count for TrOCR.

    Example
    -------
        reader = BookTextReader()
        reader.load()

        frame = cv2.imread("desk.jpg")
        result = reader.read_from_frame(frame, bbox=(100, 80, 350, 280))
        print(result.display_title)
    """

    def __init__(
        self,
        model_name:           str   = DEFAULT_MODEL,
        device:               str   = "cpu",
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        catalog:              Optional[Dict[int, str]] = None,
        max_new_tokens:       int   = 32,
    ) -> None:
        self.model_name           = model_name
        self.device               = device
        self.confidence_threshold = confidence_threshold
        self.catalog              = catalog if catalog is not None else dict(BOOK_CATALOG)
        self.max_new_tokens       = max_new_tokens

        self._processor = None   # TrOCRProcessor
        self._model     = None   # VisionEncoderDecoderModel
        self._loaded    = False
        self._load_time_s = 0.0
        self._total_reads = 0

    # ── Model lifecycle ───────────────────────────────────────────────────
    def load(self) -> bool:
        """
        Download (once) and load the TrOCR model from HuggingFace Hub.

        Returns True on success, False on failure (no crash).
        Subsequent calls are no-ops if the model is already loaded.
        """
        if self._loaded:
            return True

        try:
            # Prevent faulty Anaconda sentencepiece C-extension from causing native Windows SEH access violation (0xC0000005)
            import sys
            if "sentencepiece" not in sys.modules:
                sys.modules["sentencepiece"] = None
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        except ImportError:
            log.error(
                "transformers is not installed. "
                "Run: pip install transformers>=4.40.0"
            )
            return False

        t0 = time.time()
        try:
            # Try loading from local HuggingFace cache first for fast 1s offline startup
            try:
                self._processor = TrOCRProcessor.from_pretrained(self.model_name, local_files_only=True)
                self._model     = VisionEncoderDecoderModel.from_pretrained(self.model_name, local_files_only=True)
            except Exception:
                log.info(f"Downloading TrOCR model: {self.model_name} from HuggingFace Hub (first-time setup)...")
                self._processor = TrOCRProcessor.from_pretrained(self.model_name, local_files_only=False)
                self._model     = VisionEncoderDecoderModel.from_pretrained(self.model_name, local_files_only=False)

            self._model.eval()

            # Move to device (graceful CPU fallback)
            try:
                import torch
                target = self.device
                if target == "cuda" and not torch.cuda.is_available():
                    log.warning("CUDA not available — falling back to CPU.")
                    target = "cpu"
                self._model = self._model.to(target)
                self.device = target
            except Exception as exc:
                log.warning(f"Device placement failed ({exc}) — staying on CPU.")

            self._loaded      = True
            self._load_time_s = time.time() - t0
            log.info(
                f"TrOCR loaded in {self._load_time_s:.1f}s  "
                f"| device={self.device}  "
                f"| catalog={len(self.catalog)} titles"
            )
            return True

        except Exception as exc:
            log.error(f"TrOCR load failed: {exc}")
            return False

    # ── Main public API ───────────────────────────────────────────────────
    def read_from_frame(
        self,
        frame: np.ndarray,
        bbox:  Tuple[float, float, float, float],
        track_id: int = 0,
    ) -> BookReadResult:
        """
        Crop *bbox* from *frame*, run TrOCR, and match against the catalog.

        Parameters
        ----------
        frame    : BGR numpy array (H × W × 3).
        bbox     : (x1, y1, x2, y2) bounding box in pixel coordinates.
        track_id : ByteTrack ID of the book (0 if unknown).

        Returns
        -------
        BookReadResult — never raises; uncertain=True on any failure.
        """
        try:
            crop = self._crop_frame(frame, bbox)
            return self.read_from_crop(crop, track_id=track_id)
        except Exception as exc:
            log.warning(f"read_from_frame failed (track={track_id}): {exc}")
            return self._uncertain_result(track_id, raw_text="", error=str(exc))

    def read_from_crop(
        self,
        crop: np.ndarray,
        track_id: int = 0,
    ) -> BookReadResult:
        """
        Run TrOCR directly on an already-cropped BGR image.

        Steps
        -----
        1. Preprocess crop (contrast, resize, RGB convert).
        2. TrOCR inference → raw_text + confidence.
        3. Clean text.
        4. Fuzzy-match against catalog.
        5. Return BookReadResult.
        """
        t0 = time.time()

        if not self._loaded:
            log.debug("TrOCR not loaded — returning uncertain result.")
            return self._uncertain_result(track_id, raw_text="[model not loaded]")

        try:
            pil_img = self._preprocess_crop(crop)
        except Exception as exc:
            log.warning(f"Preprocessing failed: {exc}")
            return self._uncertain_result(track_id, raw_text="[preprocess error]")

        try:
            raw_text, ocr_conf = self._run_trocr(pil_img)
        except Exception as exc:
            log.warning(f"TrOCR inference failed: {exc}")
            return self._uncertain_result(track_id, raw_text="[inference error]")

        clean  = self._clean_text(raw_text)
        result = self._match_catalog(
            track_id=track_id,
            raw_text=raw_text,
            clean_text=clean,
            ocr_conf=ocr_conf,
            elapsed_ms=(time.time() - t0) * 1000,
        )

        self._total_reads += 1
        log.debug(
            f"TrOCR track={track_id}: raw={raw_text!r} → "
            f"clean={clean!r} conf={ocr_conf:.2f} → {result.display_title!r}"
        )
        return result

    def read_from_image_path(
        self,
        image_path: str,
        track_id: int = 0,
    ) -> BookReadResult:
        """
        Read a book title from an image file on disk.
        Useful for testing or batch processing.
        """
        try:
            from PIL import Image
            import numpy as np
            pil = Image.open(image_path).convert("RGB")
            frame = np.array(pil)[:, :, ::-1]   # RGB→BGR
            h, w = frame.shape[:2]
            return self.read_from_frame(frame, bbox=(0, 0, w, h), track_id=track_id)
        except Exception as exc:
            log.warning(f"read_from_image_path failed: {exc}")
            return self._uncertain_result(track_id, raw_text="[file error]")

    # ── Preprocessing ─────────────────────────────────────────────────────
    @staticmethod
    def _crop_frame(
        frame: np.ndarray,
        bbox: Tuple[float, float, float, float],
    ) -> np.ndarray:
        """
        Safely crop *bbox* from *frame*, clamping to frame boundaries.

        Parameters
        ----------
        frame : BGR numpy array.
        bbox  : (x1, y1, x2, y2).

        Returns
        -------
        Cropped BGR numpy array (minimum 1×1 pixel).
        """
        h, w = frame.shape[:2]
        x1 = max(0, int(bbox[0]))
        y1 = max(0, int(bbox[1]))
        x2 = min(w, int(bbox[2]))
        y2 = min(h, int(bbox[3]))
        if x2 <= x1 or y2 <= y1:
            # degenerate bbox — return 1-pixel blank
            return np.zeros((4, 4, 3), dtype=np.uint8)
        return frame[y1:y2, x1:x2].copy()

    @staticmethod
    def _preprocess_crop(crop: np.ndarray):
        """
        Prepare a BGR crop for TrOCR:

        1. BGR → RGB.
        2. Adaptive contrast enhancement (CLAHE-style via PIL).
        3. Resize to TROCR_INPUT_SIZE (keeps aspect ratio with letterboxing).
        4. Return PIL.Image (RGB).
        """
        from PIL import Image, ImageOps, ImageEnhance, ImageFilter

        # BGR → PIL RGB
        rgb = crop[:, :, ::-1]
        img = Image.fromarray(rgb.astype("uint8"), mode="RGB")

        # Convert to greyscale for enhancement, then back to RGB
        grey = img.convert("L")
        # Auto-level (normalise histogram)
        grey = ImageOps.autocontrast(grey, cutoff=2)
        # Mild sharpening
        grey = grey.filter(ImageFilter.SHARPEN)
        # Back to RGB
        enhanced = grey.convert("RGB")
        # Contrast boost
        enhanced = ImageEnhance.Contrast(enhanced).enhance(1.5)

        # Letterbox-resize to TROCR_INPUT_SIZE
        enhanced.thumbnail(TROCR_INPUT_SIZE, Image.LANCZOS)
        canvas = Image.new("RGB", TROCR_INPUT_SIZE, (255, 255, 255))
        offset = (
            (TROCR_INPUT_SIZE[0] - enhanced.width)  // 2,
            (TROCR_INPUT_SIZE[1] - enhanced.height) // 2,
        )
        canvas.paste(enhanced, offset)
        return canvas

    def _run_trocr(self, pil_img) -> Tuple[str, float]:
        """
        Run TrOCR on a preprocessed PIL image.

        Returns
        -------
        (raw_text, confidence)  where confidence ∈ [0, 1].
        """
        import torch

        pixel_values = self._processor(
            images=pil_img,
            return_tensors="pt",
        ).pixel_values.to(self.device)

        with torch.no_grad():
            outputs = self._model.generate(
                pixel_values,
                max_new_tokens=self.max_new_tokens,
                output_scores=True,
                return_dict_in_generate=True,
            )

        # Decode generated tokens → raw text
        sequences = outputs.sequences
        generated_ids = sequences[:, 1:]   # strip BOS token
        raw_text = self._processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()

        # Estimate confidence from per-token log-probabilities
        confidence = self._estimate_confidence(outputs)

        return raw_text, confidence

    @staticmethod
    def _estimate_confidence(outputs) -> float:
        """
        Convert generation scores to a single confidence value.

        Uses the geometric mean of the per-token softmax-max probabilities
        across all generated tokens.  Falls back to 0.0 on any error.
        """
        try:
            import torch
            scores = outputs.scores          # tuple of (vocab) tensors
            if not scores:
                return 0.0
            probs = []
            for step_scores in scores:
                # step_scores: (batch=1, vocab)
                step_probs = torch.softmax(step_scores[0], dim=-1)
                max_prob   = step_probs.max().item()
                if max_prob > 0:
                    probs.append(max_prob)
            if not probs:
                return 0.0
            # Geometric mean
            import math
            geo_mean = math.exp(sum(math.log(p) for p in probs) / len(probs))
            return float(min(max(geo_mean, 0.0), 1.0))
        except Exception:
            return 0.0

    # ── Text cleaning ─────────────────────────────────────────────────────
    @staticmethod
    def _clean_text(raw: str) -> str:
        """
        Normalise OCR output for catalog matching:

        - Strip leading/trailing whitespace.
        - Collapse multiple spaces.
        - Remove non-printable characters.
        - Remove common OCR artefacts (pure punctuation runs, lone digits).
        - Title-case.
        """
        if not raw:
            return ""
        # Remove non-printable / non-ASCII junk
        cleaned = re.sub(r"[^\x20-\x7E]", " ", raw)
        # Collapse whitespace
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        # Drop tokens that are only punctuation or digits ≤ 1 char
        tokens = [t for t in cleaned.split() if re.search(r"[A-Za-z]", t)]
        cleaned = " ".join(tokens)
        return cleaned.strip()

    # ── Catalog matching ──────────────────────────────────────────────────
    def _match_catalog(
        self,
        track_id:  int,
        raw_text:  str,
        clean_text: str,
        ocr_conf:  float,
        elapsed_ms: float,
    ) -> BookReadResult:
        """
        Fuzzy-match *clean_text* against ``self.catalog``.

        Strategy:
        1. If OCR confidence < threshold → uncertain immediately.
        2. Compute SequenceMatcher ratio against each catalog title.
        3. Best match above CATALOG_MATCH_CUTOFF wins.
        4. If no match → uncertain.
        """
        # Low confidence shortcut
        if ocr_conf < self.confidence_threshold or not clean_text:
            return BookReadResult(
                track_id=track_id,
                raw_text=raw_text,
                clean_text=clean_text,
                book_id=None,
                title=UNCERTAIN_LABEL,
                confidence=ocr_conf,
                match_score=0.0,
                uncertain=True,
                elapsed_ms=elapsed_ms,
            )

        # Fuzzy match against all catalog titles
        best_id:    Optional[int]   = None
        best_title: str             = UNCERTAIN_LABEL
        best_score: float           = 0.0

        query_lower = clean_text.lower()

        for book_id, title in self.catalog.items():
            title_lower = title.lower()
            # SequenceMatcher ratio
            ratio = SequenceMatcher(
                None, query_lower, title_lower, autojunk=False
            ).ratio()
            # Bonus: if the query is a substring of the title (partial match)
            if query_lower in title_lower or title_lower in query_lower:
                ratio = max(ratio, 0.65)
            if ratio > best_score:
                best_score = ratio
                best_id    = book_id
                best_title = title

        uncertain = best_score < CATALOG_MATCH_CUTOFF

        return BookReadResult(
            track_id=track_id,
            raw_text=raw_text,
            clean_text=clean_text,
            book_id=best_id if not uncertain else None,
            title=best_title if not uncertain else UNCERTAIN_LABEL,
            confidence=ocr_conf,
            match_score=best_score,
            uncertain=uncertain,
            elapsed_ms=elapsed_ms,
        )

    # ── Catalog management ────────────────────────────────────────────────
    def add_to_catalog(self, book_id: int, title: str) -> None:
        """Add or overwrite a catalog entry at runtime."""
        self.catalog[book_id] = title
        log.info(f"Catalog updated: {book_id} → {title!r}  ({len(self.catalog)} total)")

    def get_catalog(self) -> Dict[int, str]:
        """Return a copy of the current catalog."""
        return dict(self.catalog)

    # ── Frame annotation ─────────────────────────────────────────────────
    def draw_result(
        self,
        frame: np.ndarray,
        bbox:  Tuple[float, float, float, float],
        result: BookReadResult,
    ) -> np.ndarray:
        """
        Overlay the BookReadResult title on *frame* above the bounding box.

        Returns an annotated copy (original unchanged).
        """
        import cv2
        output = frame.copy()

        x1, y1, x2, y2 = (int(v) for v in bbox)
        label = result.display_title
        conf_tag = f"OCR:{result.confidence:.2f}"

        font       = cv2.FONT_HERSHEY_SIMPLEX
        color      = (72, 187, 120) if not result.uncertain else (60, 60, 220)
        font_scale = 0.55
        thickness  = 1

        (tw,  th),  baseline = cv2.getTextSize(label, font, font_scale, thickness)
        (tw2, th2), _        = cv2.getTextSize(conf_tag, font, 0.38, 1)

        # Title bar
        lx = x1
        ly = max(y1 - th - th2 - 14, th + 6)
        cv2.rectangle(output, (lx, ly - th - 6), (lx + tw + 6, ly + baseline), color, -1)
        cv2.putText(output, label, (lx + 3, ly - baseline),
                    font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        # Confidence sub-tag
        cv2.putText(output, conf_tag, (lx + 3, ly + th2 + 2),
                    font, 0.38, (180, 180, 180), 1, cv2.LINE_AA)

        return output

    # ── Helpers ───────────────────────────────────────────────────────────
    def _uncertain_result(
        self, track_id: int, raw_text: str, error: str = ""
    ) -> BookReadResult:
        msg = f"[{error}]" if error else ""
        return BookReadResult(
            track_id=track_id,
            raw_text=raw_text + msg,
            clean_text="",
            book_id=None,
            title=UNCERTAIN_LABEL,
            confidence=0.0,
            match_score=0.0,
            uncertain=True,
        )

    # ── Properties ────────────────────────────────────────────────────────
    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def total_reads(self) -> int:
        return self._total_reads

    @property
    def load_time_seconds(self) -> float:
        return self._load_time_s

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "not loaded"
        return (
            f"BookTextReader("
            f"model={self.model_name!r}, "
            f"status={status}, "
            f"catalog={len(self.catalog)}, "
            f"reads={self._total_reads})"
        )
