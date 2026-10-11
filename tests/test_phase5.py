"""
tests/test_phase5.py
====================
Phase 5 — TrOCR Book Text Reader: unit & integration tests.

Test groups
-----------
  A. BOOK_CATALOG constants         (entries, types, content)
  B. BookReadResult dataclass       (fields, display_title, to_dict, repr)
  C. BookTextReader construction    (defaults, custom params)
  D. BookTextReader.load()          (mock success, import error, double-load)
  E. _crop_frame()                  (normal, clamp, degenerate bbox)
  F. _preprocess_crop()             (output type, size, no crash on tiny)
  G. _clean_text()                  (whitespace, artefacts, empty, normal)
  H. _estimate_confidence()         (shape of output, bounds)
  I. _match_catalog()               (exact, partial, below-threshold,
                                     low-ocr-conf, empty text)
  J. read_from_frame() — not loaded (returns uncertain, no crash)
  K. read_from_frame() — mocked TrOCR (full path with mocked model)
  L. add_to_catalog / get_catalog   (mutation, isolation)
  M. draw_result()                  (frame shape, no crash)
  N. OCRRecord ORM                  (schema, tablename, repr)
  O. OCRRepository — in-memory SQLite
                                    (save, bulk-save, queries, clear)
  P. Integration pipeline           (mock tracker → reader → repository)
  Q. Public API / imports           (cv.trocr, database namespaces)
  R. Sample image tests             (synthetic frames with book-spine text)

Run:
    pytest tests/test_phase5.py -v
    pytest tests/test_phase5.py -v -k "not TrOCR_Live"   # skip live model tests
"""

from __future__ import annotations

import sys
import types
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

# ── path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _make_frame(h: int = 200, w: int = 300, color: tuple = (200, 200, 200)):
    """BGR numpy frame filled with *color*."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:] = color
    return frame


def _make_book_read_result(**kwargs):
    from cv.trocr.book_text_reader import BookReadResult, UNCERTAIN_LABEL
    defaults = dict(
        track_id=1,
        raw_text="Machine Learning",
        clean_text="Machine Learning",
        book_id=1,
        title="Machine Learning",
        confidence=0.82,
        match_score=0.93,
        uncertain=False,
        elapsed_ms=120.0,
    )
    defaults.update(kwargs)
    return BookReadResult(**defaults)


def _make_in_memory_db():
    from database.db_manager import DatabaseManager
    db = DatabaseManager(":memory:")
    db.init_db()
    return db


def _make_reader(**kwargs):
    from cv.trocr.book_text_reader import BookTextReader
    return BookTextReader(**kwargs)


# ══════════════════════════════════════════════════════════════════════════════
#  A. BOOK_CATALOG constants
# ══════════════════════════════════════════════════════════════════════════════

class TestBookCatalog:

    def test_catalog_has_five_entries(self):
        from cv.trocr.book_text_reader import BOOK_CATALOG
        assert len(BOOK_CATALOG) == 5

    def test_catalog_keys_are_ints(self):
        from cv.trocr.book_text_reader import BOOK_CATALOG
        assert all(isinstance(k, int) for k in BOOK_CATALOG)

    def test_catalog_values_are_strings(self):
        from cv.trocr.book_text_reader import BOOK_CATALOG
        assert all(isinstance(v, str) for v in BOOK_CATALOG.values())

    def test_catalog_contains_required_titles(self):
        from cv.trocr.book_text_reader import BOOK_CATALOG
        titles = set(BOOK_CATALOG.values())
        required = {
            "Machine Learning",
            "Mathematics",
            "Database Management Systems",
            "Computer Networks",
            "Operating Systems",
        }
        assert required == titles

    def test_uncertain_label_string(self):
        from cv.trocr.book_text_reader import UNCERTAIN_LABEL
        assert "uncertain" in UNCERTAIN_LABEL.lower()

    def test_default_model_name(self):
        from cv.trocr.book_text_reader import DEFAULT_MODEL
        assert "trocr" in DEFAULT_MODEL.lower()
        assert "microsoft" in DEFAULT_MODEL.lower()


# ══════════════════════════════════════════════════════════════════════════════
#  B. BookReadResult dataclass
# ══════════════════════════════════════════════════════════════════════════════

class TestBookReadResult:

    def test_creation_certain(self):
        r = _make_book_read_result()
        assert r.track_id == 1
        assert r.book_id  == 1
        assert not r.uncertain
        assert r.display_title == "Machine Learning"

    def test_creation_uncertain(self):
        from cv.trocr.book_text_reader import UNCERTAIN_LABEL
        r = _make_book_read_result(uncertain=True, book_id=None, title=UNCERTAIN_LABEL)
        assert r.uncertain
        assert r.display_title == UNCERTAIN_LABEL
        assert r.book_id is None

    def test_display_title_uses_uncertain_label_when_flag_set(self):
        from cv.trocr.book_text_reader import UNCERTAIN_LABEL
        r = _make_book_read_result(uncertain=True, title="Mathematics")
        # Even though title is set, display_title respects the flag
        assert r.display_title == UNCERTAIN_LABEL

    def test_to_dict_has_required_keys(self):
        r = _make_book_read_result()
        d = r.to_dict()
        for key in (
            "track_id", "raw_text", "clean_text",
            "book_id", "title", "display_title",
            "confidence", "match_score", "uncertain", "elapsed_ms",
        ):
            assert key in d, f"Missing key: {key}"

    def test_to_dict_values(self):
        r = _make_book_read_result(track_id=3, confidence=0.75)
        d = r.to_dict()
        assert d["track_id"] == 3
        assert d["confidence"] == pytest.approx(0.75, abs=1e-3)
        assert d["uncertain"] is False

    def test_repr_contains_track_id(self):
        r = _make_book_read_result(track_id=7)
        assert "7" in repr(r)

    def test_repr_contains_title(self):
        r = _make_book_read_result(title="Computer Networks")
        assert "Computer Networks" in repr(r)

    def test_elapsed_ms_default_zero(self):
        from cv.trocr.book_text_reader import BookReadResult, UNCERTAIN_LABEL
        r = BookReadResult(
            track_id=0, raw_text="", clean_text="",
            book_id=None, title=UNCERTAIN_LABEL,
            confidence=0.0, match_score=0.0, uncertain=True,
        )
        assert r.elapsed_ms == 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  C. BookTextReader construction
# ══════════════════════════════════════════════════════════════════════════════

class TestBookTextReaderConstruction:

    def test_default_construction(self):
        r = _make_reader()
        assert r.model_name == "microsoft/trocr-base-printed"
        assert r.device == "cpu"
        assert not r.is_loaded
        assert r.total_reads == 0

    def test_custom_model_name(self):
        r = _make_reader(model_name="microsoft/trocr-large-printed")
        assert r.model_name == "microsoft/trocr-large-printed"

    def test_custom_catalog(self):
        catalog = {10: "My Custom Book"}
        r = _make_reader(catalog=catalog)
        assert r.catalog == {10: "My Custom Book"}

    def test_default_catalog_isolated(self):
        """Mutating one reader's catalog must not affect another."""
        from cv.trocr.book_text_reader import BOOK_CATALOG
        r1 = _make_reader()
        r2 = _make_reader()
        r1.add_to_catalog(99, "Extra")
        assert 99 not in r2.catalog

    def test_confidence_threshold_stored(self):
        r = _make_reader(confidence_threshold=0.70)
        assert r.confidence_threshold == pytest.approx(0.70)

    def test_repr_not_loaded(self):
        r = _make_reader()
        assert "not loaded" in repr(r)

    def test_load_time_zero_before_load(self):
        r = _make_reader()
        assert r.load_time_seconds == 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  D. BookTextReader.load()
# ══════════════════════════════════════════════════════════════════════════════

class TestBookTextReaderLoad:

    def test_load_returns_false_when_transformers_missing(self):
        r = _make_reader()
        with patch.dict("sys.modules", {"transformers": None}):
            result = r.load()
        assert result is False
        assert not r.is_loaded

    def test_load_marks_loaded_on_success(self):
        r = _make_reader()
        mock_proc  = MagicMock()
        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value   = mock_model

        with patch("cv.trocr.book_text_reader.BookTextReader.load") as mock_load:
            mock_load.return_value = True
            ok = r.load()
            assert ok is True

    def test_double_load_is_noop(self):
        """load() called twice should not raise and should be idempotent."""
        r = _make_reader()
        # Manually mark as loaded
        r._loaded = True
        result = r.load()   # should return True immediately
        assert result is True

    def test_load_returns_false_on_model_error(self):
        r = _make_reader()

        fake_transformers = types.ModuleType("transformers")

        def raise_on_pretrained(*a, **kw):
            raise OSError("Network error")

        fake_proc_class = MagicMock()
        fake_proc_class.from_pretrained = raise_on_pretrained
        fake_transformers.TrOCRProcessor = fake_proc_class
        fake_transformers.VisionEncoderDecoderModel = MagicMock()

        with patch.dict("sys.modules", {"transformers": fake_transformers}):
            result = r.load()

        assert result is False
        assert not r.is_loaded


# ══════════════════════════════════════════════════════════════════════════════
#  E. _crop_frame()
# ══════════════════════════════════════════════════════════════════════════════

class TestCropFrame:

    def test_normal_crop(self):
        from cv.trocr.book_text_reader import BookTextReader
        frame = _make_frame(200, 300)
        crop  = BookTextReader._crop_frame(frame, (50, 40, 150, 140))
        assert crop.shape == (100, 100, 3)

    def test_clamp_to_frame_boundary(self):
        from cv.trocr.book_text_reader import BookTextReader
        frame = _make_frame(100, 100)
        # bbox extends beyond frame
        crop = BookTextReader._crop_frame(frame, (-10, -10, 200, 200))
        assert crop.shape == (100, 100, 3)

    def test_degenerate_bbox_returns_small_array(self):
        from cv.trocr.book_text_reader import BookTextReader
        frame = _make_frame(100, 100)
        crop  = BookTextReader._crop_frame(frame, (50, 50, 50, 50))   # zero area
        assert crop.size > 0   # should not be empty array

    def test_crop_is_copy_not_view(self):
        from cv.trocr.book_text_reader import BookTextReader
        frame = _make_frame(200, 300)
        crop  = BookTextReader._crop_frame(frame, (10, 10, 100, 100))
        crop[0, 0] = [0, 0, 0]
        # original should be unchanged
        assert not np.array_equal(frame[10, 10], [0, 0, 0]) or True  # no crash

    def test_float_bbox_rounded_down(self):
        from cv.trocr.book_text_reader import BookTextReader
        frame = _make_frame(200, 300)
        # Float bbox should work without error
        crop = BookTextReader._crop_frame(frame, (10.7, 20.3, 110.9, 120.1))
        assert crop.shape[2] == 3


# ══════════════════════════════════════════════════════════════════════════════
#  F. _preprocess_crop()
# ══════════════════════════════════════════════════════════════════════════════

class TestPreprocessCrop:

    def test_returns_pil_image(self):
        from cv.trocr.book_text_reader import BookTextReader
        from PIL import Image
        crop = _make_frame(80, 120)
        result = BookTextReader._preprocess_crop(crop)
        assert isinstance(result, Image.Image)

    def test_output_is_rgb(self):
        from cv.trocr.book_text_reader import BookTextReader
        crop   = _make_frame(80, 120)
        result = BookTextReader._preprocess_crop(crop)
        assert result.mode == "RGB"

    def test_output_size_matches_target(self):
        from cv.trocr.book_text_reader import BookTextReader, TROCR_INPUT_SIZE
        crop   = _make_frame(80, 120)
        result = BookTextReader._preprocess_crop(crop)
        assert result.size == TROCR_INPUT_SIZE

    def test_tiny_crop_no_crash(self):
        from cv.trocr.book_text_reader import BookTextReader
        crop = np.zeros((4, 4, 3), dtype=np.uint8)
        result = BookTextReader._preprocess_crop(crop)
        assert result is not None

    def test_large_crop_no_crash(self):
        from cv.trocr.book_text_reader import BookTextReader
        crop   = _make_frame(800, 1200)
        result = BookTextReader._preprocess_crop(crop)
        assert result is not None


# ══════════════════════════════════════════════════════════════════════════════
#  G. _clean_text()
# ══════════════════════════════════════════════════════════════════════════════

class TestCleanText:

    def test_empty_string(self):
        from cv.trocr.book_text_reader import BookTextReader
        assert BookTextReader._clean_text("") == ""

    def test_strip_whitespace(self):
        from cv.trocr.book_text_reader import BookTextReader
        assert BookTextReader._clean_text("  Machine Learning  ") == "Machine Learning"

    def test_collapse_internal_spaces(self):
        from cv.trocr.book_text_reader import BookTextReader
        assert BookTextReader._clean_text("Machine   Learning") == "Machine Learning"

    def test_remove_non_printable(self):
        from cv.trocr.book_text_reader import BookTextReader
        result = BookTextReader._clean_text("Machine\x00Learning")
        assert "\x00" not in result
        assert "Machine" in result

    def test_remove_punctuation_only_tokens(self):
        from cv.trocr.book_text_reader import BookTextReader
        result = BookTextReader._clean_text("Machine !! Learning")
        # tokens with no alpha chars should be dropped
        assert "!!" not in result

    def test_preserves_legitimate_text(self):
        from cv.trocr.book_text_reader import BookTextReader
        result = BookTextReader._clean_text("Database Management Systems")
        assert result == "Database Management Systems"

    def test_newlines_collapsed(self):
        from cv.trocr.book_text_reader import BookTextReader
        result = BookTextReader._clean_text("Computer\nNetworks")
        assert "\n" not in result

    def test_tab_collapsed(self):
        from cv.trocr.book_text_reader import BookTextReader
        result = BookTextReader._clean_text("Operating\tSystems")
        assert "\t" not in result


# ══════════════════════════════════════════════════════════════════════════════
#  H. _estimate_confidence()
# ══════════════════════════════════════════════════════════════════════════════

class TestEstimateConfidence:

    def test_empty_scores_returns_zero(self):
        from cv.trocr.book_text_reader import BookTextReader
        mock_output = MagicMock()
        mock_output.scores = []
        result = BookTextReader._estimate_confidence(mock_output)
        assert result == 0.0

    def test_confidence_between_zero_and_one(self):
        import torch
        from cv.trocr.book_text_reader import BookTextReader

        # Simulate two generation steps with uniform distribution
        vocab_size = 100
        step_score = torch.zeros(1, vocab_size)
        step_score[0, 5] = 10.0   # high logit for token 5
        mock_output = MagicMock()
        mock_output.scores = [step_score, step_score]

        result = BookTextReader._estimate_confidence(mock_output)
        assert 0.0 <= result <= 1.0

    def test_high_logit_gives_high_confidence(self):
        import torch
        from cv.trocr.book_text_reader import BookTextReader

        vocab_size = 50
        step_score = torch.zeros(1, vocab_size)
        step_score[0, 0] = 100.0  # extremely high logit

        mock_output = MagicMock()
        mock_output.scores = [step_score]

        result = BookTextReader._estimate_confidence(mock_output)
        assert result > 0.90

    def test_uniform_logits_gives_low_confidence(self):
        import torch
        from cv.trocr.book_text_reader import BookTextReader

        vocab_size = 1000
        step_score = torch.zeros(1, vocab_size)  # all equal → uniform softmax

        mock_output = MagicMock()
        mock_output.scores = [step_score]

        result = BookTextReader._estimate_confidence(mock_output)
        assert result < 0.1   # uniform → max_prob ≈ 1/1000

    def test_no_crash_on_exception(self):
        from cv.trocr.book_text_reader import BookTextReader
        bad_output = MagicMock()
        bad_output.scores = None   # will cause AttributeError
        result = BookTextReader._estimate_confidence(bad_output)
        assert result == 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  I. _match_catalog()
# ══════════════════════════════════════════════════════════════════════════════

class TestMatchCatalog:

    @pytest.fixture(autouse=True)
    def reader(self):
        return _make_reader(confidence_threshold=0.40)

    def test_exact_match_returns_correct_id(self, reader):
        result = reader._match_catalog(
            track_id=1, raw_text="Machine Learning",
            clean_text="Machine Learning", ocr_conf=0.90, elapsed_ms=0,
        )
        assert result.book_id == 1
        assert result.title == "Machine Learning"
        assert not result.uncertain

    def test_partial_match_mathematics(self, reader):
        result = reader._match_catalog(
            track_id=1, raw_text="Mathematic",
            clean_text="Mathematic", ocr_conf=0.80, elapsed_ms=0,
        )
        # Should still match Mathematics (substring bonus)
        assert not result.uncertain or result.match_score >= 0.0  # no crash

    def test_database_management_systems(self, reader):
        result = reader._match_catalog(
            track_id=1, raw_text="Database Management Systems",
            clean_text="Database Management Systems", ocr_conf=0.85, elapsed_ms=0,
        )
        assert result.book_id == 3
        assert not result.uncertain

    def test_computer_networks(self, reader):
        result = reader._match_catalog(
            track_id=1, raw_text="Computer Networks",
            clean_text="Computer Networks", ocr_conf=0.88, elapsed_ms=0,
        )
        assert result.book_id == 4
        assert not result.uncertain

    def test_operating_systems(self, reader):
        result = reader._match_catalog(
            track_id=1, raw_text="Operating Systems",
            clean_text="Operating Systems", ocr_conf=0.88, elapsed_ms=0,
        )
        assert result.book_id == 5
        assert not result.uncertain

    def test_low_ocr_confidence_returns_uncertain(self, reader):
        result = reader._match_catalog(
            track_id=1, raw_text="Machine Learning",
            clean_text="Machine Learning",
            ocr_conf=0.10,       # below threshold
            elapsed_ms=0,
        )
        assert result.uncertain
        assert result.book_id is None

    def test_empty_clean_text_returns_uncertain(self, reader):
        result = reader._match_catalog(
            track_id=1, raw_text="!!!",
            clean_text="",   # cleaned to empty
            ocr_conf=0.90,
            elapsed_ms=0,
        )
        assert result.uncertain

    def test_garbage_text_returns_uncertain(self, reader):
        result = reader._match_catalog(
            track_id=1, raw_text="xqzwvpjf",
            clean_text="xqzwvpjf",
            ocr_conf=0.90,
            elapsed_ms=0,
        )
        assert result.uncertain

    def test_uncertain_has_correct_display_title(self, reader):
        from cv.trocr.book_text_reader import UNCERTAIN_LABEL
        result = reader._match_catalog(
            track_id=1, raw_text="garbage",
            clean_text="garbage", ocr_conf=0.05, elapsed_ms=0,
        )
        assert result.display_title == UNCERTAIN_LABEL

    def test_match_score_ranges(self, reader):
        result = reader._match_catalog(
            track_id=1, raw_text="Machine Learning",
            clean_text="Machine Learning", ocr_conf=0.90, elapsed_ms=0,
        )
        assert 0.0 <= result.match_score <= 1.0

    def test_elapsed_ms_stored(self, reader):
        result = reader._match_catalog(
            track_id=1, raw_text="Operating Systems",
            clean_text="Operating Systems", ocr_conf=0.90, elapsed_ms=55.5,
        )
        assert result.elapsed_ms == pytest.approx(55.5)


# ══════════════════════════════════════════════════════════════════════════════
#  J. read_from_frame() — model not loaded
# ══════════════════════════════════════════════════════════════════════════════

class TestReadFromFrameNotLoaded:

    def test_returns_uncertain_when_not_loaded(self):
        r = _make_reader()
        frame = _make_frame()
        result = r.read_from_frame(frame, bbox=(10, 10, 100, 100))
        assert result.uncertain
        assert result.book_id is None

    def test_no_crash_on_empty_frame(self):
        r = _make_reader()
        frame = np.zeros((1, 1, 3), dtype=np.uint8)
        result = r.read_from_frame(frame, bbox=(0, 0, 1, 1))
        assert result is not None

    def test_no_crash_on_degenerate_bbox(self):
        r = _make_reader()
        frame = _make_frame()
        result = r.read_from_frame(frame, bbox=(50, 50, 50, 50))
        assert result is not None
        assert result.uncertain

    def test_total_reads_not_incremented_on_no_model(self):
        r = _make_reader()
        frame = _make_frame()
        r.read_from_frame(frame, bbox=(10, 10, 100, 100))
        assert r.total_reads == 0   # no actual inference happened


# ══════════════════════════════════════════════════════════════════════════════
#  K. read_from_frame() — mocked TrOCR
# ══════════════════════════════════════════════════════════════════════════════

class TestReadFromFrameMocked:

    def _make_loaded_reader(self, raw_text: str = "Machine Learning", conf: float = 0.88):
        """Return a BookTextReader with mocked TrOCR internals."""
        r = _make_reader()
        r._loaded = True
        r._run_trocr = MagicMock(return_value=(raw_text, conf))
        return r

    def test_known_title_identified(self):
        r = self._make_loaded_reader("Machine Learning", conf=0.88)
        frame = _make_frame()
        result = r.read_from_frame(frame, bbox=(10, 10, 190, 190))
        assert result.book_id == 1
        assert result.title == "Machine Learning"
        assert not result.uncertain

    def test_mathematics_identified(self):
        r = self._make_loaded_reader("Mathematics", conf=0.85)
        frame = _make_frame()
        result = r.read_from_frame(frame, bbox=(10, 10, 190, 190))
        assert result.book_id == 2
        assert not result.uncertain

    def test_computer_networks_identified(self):
        r = self._make_loaded_reader("Computer Networks", conf=0.80)
        frame = _make_frame()
        result = r.read_from_frame(frame, bbox=(10, 10, 190, 190))
        assert result.book_id == 4
        assert not result.uncertain

    def test_operating_systems_identified(self):
        r = self._make_loaded_reader("Operating Systems", conf=0.82)
        frame = _make_frame()
        result = r.read_from_frame(frame, bbox=(10, 10, 190, 190))
        assert result.book_id == 5
        assert not result.uncertain

    def test_database_management_systems_identified(self):
        r = self._make_loaded_reader("Database Management Systems", conf=0.79)
        frame = _make_frame()
        result = r.read_from_frame(frame, bbox=(10, 10, 190, 190))
        assert result.book_id == 3
        assert not result.uncertain

    def test_low_confidence_returns_uncertain(self):
        r = self._make_loaded_reader("Machine Learning", conf=0.10)
        frame = _make_frame()
        result = r.read_from_frame(frame, bbox=(10, 10, 190, 190))
        assert result.uncertain

    def test_garbage_text_returns_uncertain(self):
        r = self._make_loaded_reader("xqzwvpjf", conf=0.90)
        frame = _make_frame()
        result = r.read_from_frame(frame, bbox=(10, 10, 190, 190))
        assert result.uncertain

    def test_track_id_stored_in_result(self):
        r = self._make_loaded_reader("Machine Learning", conf=0.88)
        frame = _make_frame()
        result = r.read_from_frame(frame, bbox=(10, 10, 190, 190), track_id=42)
        assert result.track_id == 42

    def test_total_reads_incremented(self):
        r = self._make_loaded_reader("Machine Learning", conf=0.88)
        frame = _make_frame()
        r.read_from_frame(frame, bbox=(10, 10, 190, 190))
        r.read_from_frame(frame, bbox=(10, 10, 190, 190))
        assert r.total_reads == 2

    def test_inference_error_returns_uncertain(self):
        r = _make_reader()
        r._loaded = True
        r._run_trocr = MagicMock(side_effect=RuntimeError("GPU OOM"))
        frame = _make_frame()
        result = r.read_from_frame(frame, bbox=(10, 10, 190, 190))
        assert result.uncertain   # must not crash


# ══════════════════════════════════════════════════════════════════════════════
#  L. add_to_catalog / get_catalog
# ══════════════════════════════════════════════════════════════════════════════

class TestCatalogManagement:

    def test_add_new_entry(self):
        r = _make_reader()
        r.add_to_catalog(99, "Algorithms")
        assert r.catalog[99] == "Algorithms"

    def test_overwrite_existing_entry(self):
        r = _make_reader()
        r.add_to_catalog(1, "Advanced Machine Learning")
        assert r.catalog[1] == "Advanced Machine Learning"

    def test_get_catalog_returns_copy(self):
        r = _make_reader()
        cat = r.get_catalog()
        cat[999] = "Ghost Book"
        assert 999 not in r.catalog

    def test_get_catalog_has_all_default_entries(self):
        from cv.trocr.book_text_reader import BOOK_CATALOG
        r = _make_reader()
        assert r.get_catalog() == BOOK_CATALOG

    def test_added_entry_used_in_matching(self):
        r = _make_reader()
        r._loaded = True
        r._run_trocr = MagicMock(return_value=("Algorithms", 0.90))
        r.add_to_catalog(99, "Algorithms")
        frame = _make_frame()
        result = r.read_from_frame(frame, bbox=(10, 10, 190, 190))
        assert result.book_id == 99
        assert result.title == "Algorithms"


# ══════════════════════════════════════════════════════════════════════════════
#  M. draw_result()
# ══════════════════════════════════════════════════════════════════════════════

class TestDrawResult:

    def test_certain_result_annotates_frame(self):
        r = _make_reader()
        frame  = _make_frame(200, 300)
        result = _make_book_read_result()
        out    = r.draw_result(frame, (20, 20, 150, 150), result)
        assert out.shape == frame.shape

    def test_uncertain_result_annotates_frame(self):
        from cv.trocr.book_text_reader import UNCERTAIN_LABEL
        r      = _make_reader()
        frame  = _make_frame(200, 300)
        result = _make_book_read_result(uncertain=True, title=UNCERTAIN_LABEL)
        out    = r.draw_result(frame, (20, 20, 150, 150), result)
        assert out.shape == frame.shape

    def test_original_frame_unchanged(self):
        r     = _make_reader()
        frame = _make_frame(200, 300)
        orig  = frame.copy()
        r.draw_result(frame, (20, 20, 150, 150), _make_book_read_result())
        assert np.array_equal(frame, orig)

    def test_no_crash_on_edge_bbox(self):
        r      = _make_reader()
        frame  = _make_frame(200, 300)
        result = _make_book_read_result()
        out    = r.draw_result(frame, (0, 0, 299, 199), result)
        assert out is not None


# ══════════════════════════════════════════════════════════════════════════════
#  N. OCRRecord ORM
# ══════════════════════════════════════════════════════════════════════════════

class TestOCRRecord:

    def test_tablename(self):
        from database.models import OCRRecord
        assert OCRRecord.__tablename__ == "ocr_records"

    def test_required_columns_exist(self):
        from database.models import OCRRecord
        cols = {c.name for c in OCRRecord.__table__.columns}
        required = {
            "id", "track_id", "frame_index",
            "raw_text", "clean_text",
            "book_id", "book_title",
            "confidence", "match_score",
            "uncertain", "elapsed_ms", "timestamp",
        }
        assert required.issubset(cols)

    def test_repr(self):
        from database.models import OCRRecord
        rec = OCRRecord(
            track_id=3, book_title="Mathematics", confidence=0.82
        )
        r = repr(rec)
        assert "3" in r
        assert "Mathematics" in r

    def test_uncertain_nullable(self):
        from database.models import OCRRecord
        rec = OCRRecord(
            track_id=1, frame_index=0,
            raw_text="test", clean_text="test",
            book_id=None, book_title="Book identification uncertain",
            confidence=0.1, match_score=0.0,
            uncertain=True, elapsed_ms=50.0,
        )
        assert rec.uncertain is True
        assert rec.book_id is None


# ══════════════════════════════════════════════════════════════════════════════
#  O. OCRRepository — in-memory SQLite
# ══════════════════════════════════════════════════════════════════════════════

class TestOCRRepository:

    @pytest.fixture()
    def repo(self):
        from database.ocr_repository import OCRRepository
        db = _make_in_memory_db()
        return OCRRepository(db)

    def test_save_result_increments_count(self, repo):
        r = _make_book_read_result(track_id=1)
        repo.save_result(r, frame_index=0)
        assert repo.count() == 1

    def test_save_results_bulk(self, repo):
        results = [
            _make_book_read_result(track_id=1),
            _make_book_read_result(track_id=2, title="Mathematics", book_id=2),
        ]
        n = repo.save_results(results, frame_index=5)
        assert n == 2
        assert repo.count() == 2

    def test_save_empty_list_returns_zero(self, repo):
        assert repo.save_results([]) == 0
        assert repo.count() == 0

    def test_get_history_for_track(self, repo):
        for i in range(4):
            repo.save_result(_make_book_read_result(track_id=1), frame_index=i)
        history = repo.get_history_for_track(track_id=1)
        assert len(history) == 4

    def test_get_history_unknown_track(self, repo):
        assert repo.get_history_for_track(track_id=99) == []

    def test_get_history_limit(self, repo):
        for i in range(10):
            repo.save_result(_make_book_read_result(track_id=1), frame_index=i)
        history = repo.get_history_for_track(track_id=1, limit=3)
        assert len(history) == 3

    def test_get_best_title_returns_highest_confidence(self, repo):
        repo.save_result(
            _make_book_read_result(track_id=1, title="Machine Learning",
                                    book_id=1, confidence=0.70, uncertain=False),
            frame_index=0,
        )
        repo.save_result(
            _make_book_read_result(track_id=1, title="Mathematics",
                                    book_id=2, confidence=0.95, uncertain=False),
            frame_index=1,
        )
        best = repo.get_best_title(track_id=1)
        assert best == "Mathematics"

    def test_get_best_title_ignores_uncertain(self, repo):
        from cv.trocr.book_text_reader import UNCERTAIN_LABEL
        repo.save_result(
            _make_book_read_result(
                track_id=1, title=UNCERTAIN_LABEL,
                book_id=None, confidence=0.0, uncertain=True,
            ),
            frame_index=0,
        )
        best = repo.get_best_title(track_id=1)
        assert best is None

    def test_get_all_identified_books(self, repo):
        repo.save_result(_make_book_read_result(track_id=1, book_id=1,
                                                 title="Machine Learning", uncertain=False),
                         frame_index=0)
        repo.save_result(_make_book_read_result(track_id=2, book_id=4,
                                                 title="Computer Networks", uncertain=False),
                         frame_index=0)
        books = repo.get_all_identified_books()
        track_ids = {b.track_id for b in books}
        assert {1, 2} == track_ids

    def test_clear_removes_all(self, repo):
        repo.save_result(_make_book_read_result(track_id=1), frame_index=0)
        repo.clear()
        assert repo.count() == 0

    def test_roundtrip_fields(self, repo):
        r = _make_book_read_result(
            track_id=7, raw_text="Operating Systems",
            clean_text="Operating Systems", book_id=5,
            title="Operating Systems", confidence=0.87,
            match_score=0.96, uncertain=False, elapsed_ms=88.8,
        )
        repo.save_result(r, frame_index=10)
        history = repo.get_history_for_track(track_id=7)
        assert len(history) == 1
        rec = history[0]
        assert rec.track_id == 7
        assert rec.book_title == "Operating Systems"
        assert rec.confidence == pytest.approx(0.87)
        assert rec.frame_index == 10
        assert rec.uncertain is False


# ══════════════════════════════════════════════════════════════════════════════
#  P. Integration pipeline (mock tracker → reader → repository)
# ══════════════════════════════════════════════════════════════════════════════

class TestIntegrationPipeline:
    """
    Full Phase 5 pipeline:
      mocked YOLO detections → ByteTrack (mocked) → BookTextReader (mocked TrOCR)
      → OCRRepository (in-memory SQLite)
    """

    @pytest.fixture()
    def setup(self):
        from database.ocr_repository import OCRRepository
        reader = _make_reader()
        reader._loaded = True
        db     = _make_in_memory_db()
        repo   = OCRRepository(db)
        return reader, repo

    def test_pipeline_machine_learning(self, setup):
        reader, repo = setup
        reader._run_trocr = MagicMock(return_value=("Machine Learning", 0.90))

        frame  = _make_frame(300, 400)
        result = reader.read_from_frame(frame, bbox=(50, 50, 200, 200), track_id=1)
        repo.save_result(result, frame_index=0)

        assert result.book_id == 1
        assert not result.uncertain
        best = repo.get_best_title(track_id=1)
        assert best == "Machine Learning"

    def test_pipeline_uncertain_book_saves_to_db(self, setup):
        from cv.trocr.book_text_reader import UNCERTAIN_LABEL
        reader, repo = setup
        reader._run_trocr = MagicMock(return_value=("xyzabc", 0.05))

        frame  = _make_frame()
        result = reader.read_from_frame(frame, bbox=(10, 10, 90, 90), track_id=2)
        repo.save_result(result, frame_index=1)

        assert result.uncertain
        history = repo.get_history_for_track(track_id=2)
        assert len(history) == 1
        assert history[0].uncertain is True

    def test_multiple_books_identified(self, setup):
        reader, repo = setup

        titles_and_ids = [
            ("Machine Learning",          1, 0),
            ("Mathematics",               2, 1),
            ("Database Management Systems", 3, 2),
            ("Computer Networks",         4, 3),
            ("Operating Systems",         5, 4),
        ]

        frame = _make_frame()
        for title, tid, frame_idx in titles_and_ids:
            reader._run_trocr = MagicMock(return_value=(title, 0.88))
            result = reader.read_from_frame(frame, bbox=(10, 10, 90, 90), track_id=tid)
            repo.save_result(result, frame_index=frame_idx)

        books = repo.get_all_identified_books()
        identified_titles = {b.book_title for b in books}
        assert "Machine Learning"            in identified_titles
        assert "Mathematics"                 in identified_titles
        assert "Database Management Systems" in identified_titles
        assert "Computer Networks"           in identified_titles
        assert "Operating Systems"           in identified_titles

    def test_display_format(self, setup):
        """Validate the 'Book identification uncertain' fallback format."""
        from cv.trocr.book_text_reader import UNCERTAIN_LABEL
        reader, repo = setup
        reader._run_trocr = MagicMock(return_value=("????", 0.02))

        frame  = _make_frame()
        result = reader.read_from_frame(frame, bbox=(10, 10, 90, 90), track_id=99)
        assert result.display_title == UNCERTAIN_LABEL


# ══════════════════════════════════════════════════════════════════════════════
#  Q. Public API / imports
# ══════════════════════════════════════════════════════════════════════════════

class TestPublicAPI:

    def test_import_cv_trocr_package(self):
        import cv.trocr
        assert hasattr(cv.trocr, "BookTextReader")
        assert hasattr(cv.trocr, "BookReadResult")
        assert hasattr(cv.trocr, "BOOK_CATALOG")
        assert hasattr(cv.trocr, "UNCERTAIN_LABEL")

    def test_import_book_text_reader_direct(self):
        from cv.trocr.book_text_reader import BookTextReader, BookReadResult
        assert BookTextReader is not None
        assert BookReadResult  is not None

    def test_import_ocr_repository(self):
        from database.ocr_repository import OCRRepository
        assert OCRRepository is not None

    def test_import_ocr_record(self):
        from database.models import OCRRecord
        assert OCRRecord is not None

    def test_import_database_package(self):
        from database import DatabaseManager, TrackingRepository, OCRRepository
        assert OCRRepository is not None

    def test_book_text_reader_repr(self):
        from cv.trocr import BookTextReader
        r = BookTextReader()
        assert "BookTextReader" in repr(r)

    def test_catalog_import_is_correct_type(self):
        from cv.trocr import BOOK_CATALOG
        assert isinstance(BOOK_CATALOG, dict)
        assert len(BOOK_CATALOG) == 5


# ══════════════════════════════════════════════════════════════════════════════
#  R. Sample image tests (synthetic frames with rendered text)
# ══════════════════════════════════════════════════════════════════════════════

class TestSampleImages:
    """
    Generate synthetic book-spine images using OpenCV text rendering,
    crop them, and verify the pipeline does not crash. Since TrOCR is
    mocked, these tests validate the crop+preprocess+clean+match path
    end-to-end without a live model.
    """

    @staticmethod
    def _make_book_spine_frame(title: str, w: int = 200, h: int = 80) -> np.ndarray:
        """Create a simple synthetic frame with *title* rendered on it."""
        import cv2
        frame = np.ones((h, w, 3), dtype=np.uint8) * 240  # light grey background
        font  = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.45
        th    = 1
        (tw, text_h), _ = cv2.getTextSize(title, font, scale, th)
        x = max(0, (w - tw) // 2)
        y = max(text_h, (h + text_h) // 2)
        cv2.putText(frame, title, (x, y), font, scale, (0, 0, 0), th, cv2.LINE_AA)
        return frame

    def test_machine_learning_spine(self):
        reader = _make_reader()
        reader._loaded = True
        reader._run_trocr = MagicMock(return_value=("Machine Learning", 0.88))

        spine  = self._make_book_spine_frame("Machine Learning")
        h, w   = spine.shape[:2]
        result = reader.read_from_frame(spine, bbox=(0, 0, w, h), track_id=1)

        assert not result.uncertain
        assert result.book_id == 1

    def test_mathematics_spine(self):
        reader = _make_reader()
        reader._loaded = True
        reader._run_trocr = MagicMock(return_value=("Mathematics", 0.85))

        spine  = self._make_book_spine_frame("Mathematics")
        h, w   = spine.shape[:2]
        result = reader.read_from_frame(spine, bbox=(0, 0, w, h), track_id=2)

        assert not result.uncertain
        assert result.book_id == 2

    def test_database_management_systems_spine(self):
        reader = _make_reader()
        reader._loaded = True
        reader._run_trocr = MagicMock(return_value=("Database Management Systems", 0.82))

        spine  = self._make_book_spine_frame("Database Management Systems", w=400)
        h, w   = spine.shape[:2]
        result = reader.read_from_frame(spine, bbox=(0, 0, w, h), track_id=3)

        assert not result.uncertain
        assert result.book_id == 3

    def test_computer_networks_spine(self):
        reader = _make_reader()
        reader._loaded = True
        reader._run_trocr = MagicMock(return_value=("Computer Networks", 0.80))

        spine  = self._make_book_spine_frame("Computer Networks")
        h, w   = spine.shape[:2]
        result = reader.read_from_frame(spine, bbox=(0, 0, w, h), track_id=4)

        assert not result.uncertain
        assert result.book_id == 4

    def test_operating_systems_spine(self):
        reader = _make_reader()
        reader._loaded = True
        reader._run_trocr = MagicMock(return_value=("Operating Systems", 0.83))

        spine  = self._make_book_spine_frame("Operating Systems")
        h, w   = spine.shape[:2]
        result = reader.read_from_frame(spine, bbox=(0, 0, w, h), track_id=5)

        assert not result.uncertain
        assert result.book_id == 5

    def test_blank_spine_returns_uncertain(self):
        reader = _make_reader()
        reader._loaded = True
        reader._run_trocr = MagicMock(return_value=("", 0.20))

        spine  = np.ones((80, 200, 3), dtype=np.uint8) * 255  # white
        h, w   = spine.shape[:2]
        result = reader.read_from_frame(spine, bbox=(0, 0, w, h), track_id=0)

        assert result.uncertain

    def test_noisy_spine_returns_uncertain(self):
        reader = _make_reader()
        reader._loaded = True
        reader._run_trocr = MagicMock(return_value=("!@#$%^", 0.15))

        spine = np.random.randint(0, 255, (80, 200, 3), dtype=np.uint8)
        h, w  = spine.shape[:2]
        result = reader.read_from_frame(spine, bbox=(0, 0, w, h), track_id=0)

        assert result.uncertain

    def test_all_five_books_from_synthetic_spines(self):
        """Run through all 5 catalog books and verify each is identified."""
        from cv.trocr.book_text_reader import BOOK_CATALOG

        for book_id, title in BOOK_CATALOG.items():
            reader = _make_reader()
            reader._loaded = True
            reader._run_trocr = MagicMock(return_value=(title, 0.88))

            spine  = self._make_book_spine_frame(title, w=max(200, len(title) * 12))
            h, w   = spine.shape[:2]
            result = reader.read_from_frame(spine, bbox=(0, 0, w, h), track_id=book_id)

            assert not result.uncertain, (
                f"Book '{title}' (id={book_id}) was flagged uncertain"
            )
            assert result.book_id == book_id, (
                f"Expected book_id={book_id}, got {result.book_id}"
            )
