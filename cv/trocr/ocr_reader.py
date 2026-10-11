"""
cv/trocr/ocr_reader.py
TrOCR book-title OCR reader wrapper.

Status: STUB — Implementation in Phase 3.
"""
from dataclasses import dataclass


@dataclass
class OCRResult:
    """Result of OCR on a single cropped image."""
    track_id: int
    text: str
    confidence: float


class TrOCRReader:
    """
    Wraps microsoft/trocr-base-printed for book title recognition.
    Phase 3 implementation will use transformers pipeline.
    """

    STATUS = "NOT_STARTED"

    def __init__(self, model_name: str = "microsoft/trocr-base-printed", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._processor = None
        self._model = None

    def load(self) -> None:
        """Download and load TrOCR model from HuggingFace cache."""
        # TODO Phase 3:
        #   from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        #   self._processor = TrOCRProcessor.from_pretrained(self.model_name)
        #   self._model = VisionEncoderDecoderModel.from_pretrained(self.model_name)
        raise NotImplementedError("TrOCRReader: Phase 3 not yet implemented.")

    def read(self, image_crop, track_id: int) -> OCRResult:
        """Run OCR on a cropped book image."""
        raise NotImplementedError("TrOCRReader: Phase 3 not yet implemented.")
