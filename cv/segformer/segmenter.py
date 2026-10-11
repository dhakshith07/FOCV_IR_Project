"""
cv/segformer/segmenter.py
SegFormer semantic desk zone segmenter wrapper.

Status: STUB — Implementation in Phase 3.
"""
from dataclasses import dataclass
from typing import Dict
import numpy as np


@dataclass
class SegmentationResult:
    """Output from SegFormer segmentation."""
    mask: np.ndarray          # (H, W) int array of class IDs
    class_map: Dict[int, str] # {class_id: class_name}
    zone_polygons: Dict[str, list]  # {zone_name: polygon_points}


class SegFormerSegmenter:
    """
    Wraps nvidia/segformer-b0-finetuned-ade-512-512 for desk zone segmentation.
    Phase 3 implementation will use transformers pipeline.
    """

    STATUS = "NOT_STARTED"

    def __init__(
        self,
        model_name: str = "nvidia/segformer-b0-finetuned-ade-512-512",
        device: str = "cpu"
    ):
        self.model_name = model_name
        self.device = device
        self._feature_extractor = None
        self._model = None

    def load(self) -> None:
        """Download and load SegFormer model from HuggingFace cache."""
        # TODO Phase 3:
        #   from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
        #   self._feature_extractor = SegformerImageProcessor.from_pretrained(self.model_name)
        #   self._model = SegformerForSemanticSegmentation.from_pretrained(self.model_name)
        raise NotImplementedError("SegFormerSegmenter: Phase 3 not yet implemented.")

    def segment(self, frame) -> SegmentationResult:
        """Segment a BGR frame into desk zones."""
        raise NotImplementedError("SegFormerSegmenter: Phase 3 not yet implemented.")
