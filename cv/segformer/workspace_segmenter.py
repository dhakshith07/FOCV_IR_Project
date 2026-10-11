"""
cv/segformer/workspace_segmenter.py
=====================================
Phase 6 - SegFormer semantic segmentation for workspace understanding.

Purpose
-------
Understand the workspace at pixel/region level. Produces four derived maps
from a single SegFormer inference pass:

  1. workspace_mask   - pixels that are valid, reachable table/desk surface
  2. obstacle_map     - pixels occupied by objects that block placement
  3. invalid_mask     - union of all non-placeable regions
  4. nav_map          - free-space overlay for robot path planning

Operating Modes
---------------
  PRETRAINED (default)
    Uses nvidia/segformer-b0-finetuned-ade-512-512 trained on ADE20K
    (150 indoor classes). This is a *generic* indoor segmenter. It was
    NOT trained on book-workspace data. Results are clearly labelled
    "[WARNING] PRETRAINED / DEMO MODE".

  CUSTOM (optional)
    Pass custom_checkpoint=<path or HF-repo> to load a fine-tuned
    checkpoint for your specific workspace. Interface is identical.

  SYNTHETIC (fallback)
    When no model is loadable, returns procedurally-generated masks that
    demonstrate the full system without any deep-learning dependency.
    Clearly labelled "[SYNTHETIC DEMO MODE]".

ADE20K class mapping
--------------------
  Workspace  : 3=floor, 15=table, 33=desk, 56=pool table, 64=coffee table
  Obstacle   : 10=cabinet, 12=person, 19=chair, 24=shelf, 30=armchair,
               36=lamp, 62=bookcase, 67=book, 74=computer, 75=swivel chair,
               85=chandelier, 130=screen, 141=crt screen, 143=monitor
  User/hand  : 12=person
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from utils.logger import get_logger

log = get_logger(__name__)


# ===================================================================
#  Constants - ADE20K class IDs
# ===================================================================

WORKSPACE_CLASS_IDS: List[int] = [3, 15, 33, 56, 64]

OBSTACLE_CLASS_IDS: List[int] = [
    10, 12, 19, 24, 30, 36, 62, 67, 74, 75, 85, 130, 141, 143
]

USER_CLASS_IDS: List[int] = [12]

_COLOR_WORKSPACE = (72,  187, 120)   # green  (BGR)
_COLOR_OBSTACLE  = (60,   60, 220)   # red-blue
_COLOR_USER      = (251, 191,  36)   # yellow
_COLOR_FREE      = (34,  211, 238)   # cyan
_COLOR_UNKNOWN   = (100, 100, 100)   # grey

DEFAULT_MODEL = "nvidia/segformer-b0-finetuned-ade-512-512"

ADE20K_LABELS: Dict[int, str] = {
    3: "floor", 10: "cabinet", 12: "person", 15: "table",
    19: "chair", 24: "shelf", 30: "armchair", 33: "desk",
    36: "lamp", 56: "pool table", 62: "bookcase", 64: "coffee table",
    67: "book", 74: "computer", 75: "swivel chair",
    85: "chandelier", 130: "screen", 141: "crt screen", 143: "monitor",
}


# ===================================================================
#  Enums
# ===================================================================

class SegmenterMode(Enum):
    NOT_LOADED = "not_loaded"
    PRETRAINED = "pretrained"
    CUSTOM     = "custom"
    SYNTHETIC  = "synthetic"


class WorkspaceZone(Enum):
    WORKSPACE = "workspace"
    OBSTACLE  = "obstacle"
    USER      = "user"
    FREE      = "free"
    UNKNOWN   = "unknown"


# ===================================================================
#  Result dataclasses
# ===================================================================

@dataclass
class WorkspaceRegion:
    """A contiguous workspace sub-region from contour analysis."""
    contour:  np.ndarray
    bbox:     Tuple[int, int, int, int]
    area_px:  int
    centroid: Tuple[int, int]
    is_valid: bool = True


@dataclass
class SegmentationResult:
    """
    All outputs from a single SegFormer inference pass.

    Attributes
    ----------
    raw_mask           : (H,W) int32 - ADE20K class ID per pixel
    label_mask         : (H,W) object array - WorkspaceZone per pixel
    workspace_mask     : (H,W) bool - True where placement is valid
    obstacle_map       : (H,W) bool - True where something blocks
    invalid_mask       : (H,W) bool - True where placement is invalid
    nav_map            : (H,W) bool - navigable free space for robot
    nav_distance_map   : (H,W) float32 - distance-to-obstacle (0-255)
    segmentation_bgr   : coloured full segmentation overlay (BGR)
    workspace_mask_bgr : green overlay - valid workspace
    obstacle_map_bgr   : red overlay - obstacles
    nav_map_bgr        : heat-map overlay - robot nav space
    workspace_regions  : list of WorkspaceRegion contours
    class_areas        : {class_id: pixel_count}
    workspace_pct      : fraction of frame that is valid workspace
    obstacle_pct       : fraction of frame that is obstacle
    mode               : "pretrained"|"custom"|"synthetic"|"not_loaded"
    mode_label         : human-readable mode string
    elapsed_ms         : inference wall-clock time
    frame_shape        : (H, W, C) of the input frame
    """
    raw_mask:           np.ndarray
    label_mask:         np.ndarray
    workspace_mask:     np.ndarray
    obstacle_map:       np.ndarray
    invalid_mask:       np.ndarray
    nav_map:            np.ndarray
    nav_distance_map:   np.ndarray
    segmentation_bgr:   np.ndarray
    workspace_mask_bgr: np.ndarray
    obstacle_map_bgr:   np.ndarray
    nav_map_bgr:        np.ndarray
    workspace_regions:  List[WorkspaceRegion]
    class_areas:        Dict[int, int]
    workspace_pct:      float
    obstacle_pct:       float
    mode:               str
    mode_label:         str
    elapsed_ms:         float
    frame_shape:        Tuple

    @property
    def is_workspace_detected(self) -> bool:
        return bool(self.workspace_mask.any())

    @property
    def workspace_area_px(self) -> int:
        return int(self.workspace_mask.sum())

    @property
    def obstacle_area_px(self) -> int:
        return int(self.obstacle_map.sum())

    @property
    def largest_workspace_region(self) -> Optional["WorkspaceRegion"]:
        if not self.workspace_regions:
            return None
        return max(self.workspace_regions, key=lambda r: r.area_px)

    def to_dict(self) -> dict:
        return {
            "mode":              self.mode,
            "mode_label":        self.mode_label,
            "elapsed_ms":        round(self.elapsed_ms, 1),
            "frame_shape":       list(self.frame_shape),
            "workspace_pct":     round(self.workspace_pct, 4),
            "obstacle_pct":      round(self.obstacle_pct, 4),
            "workspace_area_px": self.workspace_area_px,
            "obstacle_area_px":  self.obstacle_area_px,
            "workspace_regions": len(self.workspace_regions),
            "class_areas":       {str(k): v for k, v in self.class_areas.items()},
        }

    def __repr__(self) -> str:
        return (
            f"SegmentationResult("
            f"mode={self.mode!r}, "
            f"workspace={self.workspace_pct:.1%}, "
            f"obstacle={self.obstacle_pct:.1%}, "
            f"regions={len(self.workspace_regions)}, "
            f"elapsed={self.elapsed_ms:.0f}ms)"
        )


# ===================================================================
#  WorkspaceSegmenter
# ===================================================================

class WorkspaceSegmenter:
    """
    SegFormer-based semantic workspace segmenter.

    IMPORTANT - Model honesty
    -------------------------
    PRETRAINED mode uses nvidia/segformer-b0-finetuned-ade-512-512 which
    is a generic indoor ADE20K segmenter. It was NOT trained on
    book-workspace or robotics data. Results in pretrained mode are
    approximate and clearly labelled.

    For accurate workspace segmentation supply a fine-tuned checkpoint
    via custom_checkpoint or call set_custom_checkpoint() at runtime.

    Parameters
    ----------
    custom_checkpoint    : path or HF repo of a fine-tuned model
    pretrained_model     : fallback model (ADE20K pretrained)
    device               : "cpu" or "cuda"
    workspace_class_ids  : override ADE20K IDs treated as workspace
    obstacle_class_ids   : override ADE20K IDs treated as obstacles
    min_region_px        : minimum contiguous region to keep (noise filter)
    """

    def __init__(
        self,
        custom_checkpoint:   Optional[str]       = None,
        pretrained_model:    str                 = DEFAULT_MODEL,
        device:              str                 = "cpu",
        workspace_class_ids: Optional[List[int]] = None,
        obstacle_class_ids:  Optional[List[int]] = None,
        min_region_px:       int                 = 500,
    ) -> None:
        self.custom_checkpoint   = custom_checkpoint
        self.pretrained_model    = pretrained_model
        self.device              = device
        self.workspace_class_ids = workspace_class_ids or list(WORKSPACE_CLASS_IDS)
        self.obstacle_class_ids  = obstacle_class_ids  or list(OBSTACLE_CLASS_IDS)
        self.min_region_px       = min_region_px

        self._processor    = None
        self._model        = None
        self._id2label:    Dict[int, str] = {}
        self.mode          = SegmenterMode.NOT_LOADED
        self._load_time_s  = 0.0
        self._total_frames = 0

    # -------------------------------------------------------------------
    # Loading
    # -------------------------------------------------------------------

    def load(self) -> bool:
        """
        Load SegFormer. Priority order:
          1. Custom fine-tuned checkpoint (if provided and found)
          2. Pretrained ADE20K model (nvidia/segformer-b0-finetuned-ade-512-512)
          3. Synthetic demo mode (no model, procedural masks)

        Always returns True - synthetic mode is always usable.
        """
        if self.mode not in (SegmenterMode.NOT_LOADED, SegmenterMode.SYNTHETIC):
            return True  # already loaded

        try:
            from transformers import (
                SegformerImageProcessor,
                SegformerForSemanticSegmentation,
            )
        except ImportError:
            log.error("transformers not installed. pip install transformers>=4.40.0")
            log.warning("Falling back to SYNTHETIC demo mode.")
            self.mode = SegmenterMode.SYNTHETIC
            return True

        t0 = time.time()

        # 1. Try custom checkpoint
        if self.custom_checkpoint:
            cp      = self.custom_checkpoint
            cp_path = Path(cp) if not cp.startswith(("http", "hf://")) else None
            exists  = cp_path.exists() if cp_path else True

            if exists:
                try:
                    log.info(f"Loading CUSTOM SegFormer: {cp}")
                    self._processor = SegformerImageProcessor.from_pretrained(cp)
                    self._model     = SegformerForSemanticSegmentation.from_pretrained(cp)
                    self._model.eval()
                    self._id2label  = dict(self._model.config.id2label)
                    self.mode       = SegmenterMode.CUSTOM
                    self._load_time_s = time.time() - t0
                    log.info(
                        f"CUSTOM SegFormer loaded in {self._load_time_s:.1f}s "
                        f"({len(self._id2label)} classes)"
                    )
                    return True
                except Exception as exc:
                    log.warning(f"Custom checkpoint failed: {exc}. Falling back to pretrained.")
            else:
                log.info(
                    f"Custom checkpoint not found at: {cp}\n"
                    "   Provide a valid path or HF repo via set_custom_checkpoint().\n"
                    "   Falling back to pretrained ADE20K model."
                )

        # 2. Pretrained ADE20K
        try:
            log.info(
                f"Loading PRETRAINED SegFormer: {self.pretrained_model}\n"
                "   [WARNING] Generic ADE20K model - NOT a workspace-specific segmenter.\n"
                "   Results are approximate. Fine-tune on workspace data for production."
            )
            self._processor = SegformerImageProcessor.from_pretrained(self.pretrained_model)
            self._model     = SegformerForSemanticSegmentation.from_pretrained(self.pretrained_model)
            self._model.eval()
            self._id2label  = dict(self._model.config.id2label)
            self.mode       = SegmenterMode.PRETRAINED
            self._load_time_s = time.time() - t0
            log.info(
                f"PRETRAINED SegFormer loaded in {self._load_time_s:.1f}s "
                f"| {len(self._id2label)} classes"
            )
            return True
        except Exception as exc:
            log.error(f"SegFormer load failed: {exc}. Falling back to SYNTHETIC.")
            self.mode = SegmenterMode.SYNTHETIC
            return True

    # -------------------------------------------------------------------
    # Main inference
    # -------------------------------------------------------------------

    def segment(self, frame: np.ndarray) -> SegmentationResult:
        """
        Run SegFormer on frame and return a full SegmentationResult.

        Parameters
        ----------
        frame : BGR numpy array (H x W x 3)

        Returns
        -------
        SegmentationResult - always returns, never raises.
        """
        if self.mode in (SegmenterMode.NOT_LOADED, SegmenterMode.SYNTHETIC):
            if self.mode == SegmenterMode.NOT_LOADED:
                log.warning("WorkspaceSegmenter not loaded - call load() first. Returning synthetic.")
            return self._synthetic_result(frame)

        t0 = time.time()
        try:
            raw_mask = self._run_inference(frame)
        except Exception as exc:
            log.error(f"SegFormer inference failed: {exc}. Returning synthetic fallback.")
            return self._synthetic_result(frame)

        result = self._build_result(frame, raw_mask, time.time() - t0)
        self._total_frames += 1
        log.debug(
            f"Frame {self._total_frames}: workspace={result.workspace_pct:.1%} "
            f"obstacle={result.obstacle_pct:.1%} elapsed={result.elapsed_ms:.0f}ms"
        )
        return result

    # -------------------------------------------------------------------
    # Inference internals
    # -------------------------------------------------------------------

    def _run_inference(self, frame: np.ndarray) -> np.ndarray:
        """Run HuggingFace SegFormer and return (H,W) int32 upsampled mask."""
        import torch
        from PIL import Image

        H, W = frame.shape[:2]
        rgb  = frame[:, :, ::-1]
        pil  = Image.fromarray(rgb.astype("uint8"))

        inputs = self._processor(images=pil, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        logits    = outputs.logits                       # (1, C, H/4, W/4)
        upsampled = torch.nn.functional.interpolate(
            logits, size=(H, W), mode="bilinear", align_corners=False
        )                                                # (1, C, H, W)
        pred = upsampled.argmax(dim=1).squeeze(0)        # (H, W)
        return pred.cpu().numpy().astype(np.int32)

    # -------------------------------------------------------------------
    # Result assembly
    # -------------------------------------------------------------------

    def _build_result(
        self,
        frame:    np.ndarray,
        raw_mask: np.ndarray,
        elapsed:  float,
    ) -> SegmentationResult:
        """Build all derived maps from a raw class-ID mask."""
        H, W  = frame.shape[:2]
        total = H * W

        workspace_mask = self._build_mask(raw_mask, self.workspace_class_ids)
        obstacle_map   = self._build_mask(raw_mask, self.obstacle_class_ids)
        user_mask      = self._build_mask(raw_mask, USER_CLASS_IDS)

        workspace_mask = self._morph_clean(workspace_mask)
        obstacle_map   = self._morph_clean(obstacle_map)

        invalid_mask = obstacle_map | user_mask | ~workspace_mask
        nav_map      = workspace_mask & ~obstacle_map & ~user_mask
        nav_dist     = self._distance_map(nav_map, obstacle_map)

        workspace_regions = self._extract_regions(workspace_mask)

        unique, counts = np.unique(raw_mask, return_counts=True)
        class_areas    = {int(u): int(c) for u, c in zip(unique, counts)}

        workspace_pct = float(workspace_mask.sum()) / total
        obstacle_pct  = float(obstacle_map.sum())   / total

        label_mask    = self._build_label_mask(raw_mask, workspace_mask, obstacle_map, user_mask)
        seg_bgr       = self._colorise_segmentation(raw_mask, H, W)
        workspace_bgr = self._overlay(frame, workspace_mask, _COLOR_WORKSPACE)
        obstacle_bgr  = self._overlay(frame, obstacle_map,   _COLOR_OBSTACLE)
        nav_bgr       = self._draw_nav_overlay(frame, nav_map, nav_dist)

        mode_str   = self.mode.value
        mode_label = self._build_mode_label(mode_str)

        return SegmentationResult(
            raw_mask           = raw_mask,
            label_mask         = label_mask,
            workspace_mask     = workspace_mask,
            obstacle_map       = obstacle_map,
            invalid_mask       = invalid_mask,
            nav_map            = nav_map,
            nav_distance_map   = nav_dist,
            segmentation_bgr   = seg_bgr,
            workspace_mask_bgr = workspace_bgr,
            obstacle_map_bgr   = obstacle_bgr,
            nav_map_bgr        = nav_bgr,
            workspace_regions  = workspace_regions,
            class_areas        = class_areas,
            workspace_pct      = workspace_pct,
            obstacle_pct       = obstacle_pct,
            mode               = mode_str,
            mode_label         = mode_label,
            elapsed_ms         = elapsed * 1000,
            frame_shape        = frame.shape,
        )

    # -------------------------------------------------------------------
    # Mask / map helpers
    # -------------------------------------------------------------------

    @staticmethod
    def _build_mask(raw_mask: np.ndarray, class_ids: List[int]) -> np.ndarray:
        result = np.zeros(raw_mask.shape, dtype=bool)
        for cid in class_ids:
            result |= (raw_mask == cid)
        return result

    @staticmethod
    def _build_label_mask(
        raw_mask: np.ndarray,
        workspace_mask: np.ndarray,
        obstacle_map: np.ndarray,
        user_mask: np.ndarray,
    ) -> np.ndarray:
        label = np.full(raw_mask.shape, WorkspaceZone.UNKNOWN, dtype=object)
        label[workspace_mask]                              = WorkspaceZone.WORKSPACE
        label[obstacle_map]                                = WorkspaceZone.OBSTACLE
        label[user_mask]                                   = WorkspaceZone.USER
        label[workspace_mask & ~obstacle_map & ~user_mask] = WorkspaceZone.FREE
        return label

    def _morph_clean(self, mask: np.ndarray) -> np.ndarray:
        """Remove noise pixels with morphological open/close."""
        try:
            import cv2
            m = mask.astype(np.uint8) * 255
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  kernel, iterations=2)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=2)
            return m > 0
        except Exception:
            return mask

    def _extract_regions(self, workspace_mask: np.ndarray) -> List[WorkspaceRegion]:
        """Extract contiguous workspace sub-regions via contour analysis."""
        try:
            import cv2
            m = workspace_mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            regions = []
            for cnt in contours:
                area = int(cv2.contourArea(cnt))
                if area < self.min_region_px:
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                M  = cv2.moments(cnt)
                cx = int(M["m10"] / M["m00"]) if M["m00"] > 0 else x + w // 2
                cy = int(M["m01"] / M["m00"]) if M["m00"] > 0 else y + h // 2
                regions.append(WorkspaceRegion(
                    contour  = cnt,
                    bbox     = (x, y, w, h),
                    area_px  = area,
                    centroid = (cx, cy),
                    is_valid = True,
                ))
            return sorted(regions, key=lambda r: r.area_px, reverse=True)
        except Exception:
            return []

    @staticmethod
    def _distance_map(nav_map: np.ndarray, obstacle_map: np.ndarray) -> np.ndarray:
        """Normalised [0-255] distance-to-nearest-obstacle map."""
        try:
            import cv2
            free = (~obstacle_map).astype(np.uint8) * 255
            dist = cv2.distanceTransform(free, cv2.DIST_L2, 5)
            dist[~nav_map] = 0.0
            max_val = dist.max()
            if max_val > 0:
                dist = (dist / max_val * 255).astype(np.float32)
            return dist
        except Exception:
            return np.zeros(nav_map.shape, dtype=np.float32)

    def _colorise_segmentation(self, raw_mask: np.ndarray, H: int, W: int) -> np.ndarray:
        bgr = np.full((H, W, 3), _COLOR_UNKNOWN, dtype=np.uint8)
        for cid in self.workspace_class_ids:
            bgr[raw_mask == cid] = _COLOR_WORKSPACE
        for cid in self.obstacle_class_ids:
            bgr[raw_mask == cid] = _COLOR_OBSTACLE
        for cid in USER_CLASS_IDS:
            bgr[raw_mask == cid] = _COLOR_USER
        return bgr

    @staticmethod
    def _overlay(
        frame: np.ndarray,
        mask:  np.ndarray,
        color: Tuple[int, int, int],
        alpha: float = 0.45,
    ) -> np.ndarray:
        """Alpha-blend color onto frame pixels where mask is True."""
        output  = frame.copy()
        colored = np.zeros_like(frame)
        colored[mask] = color
        blend = output.astype(float) * (1 - alpha) + colored.astype(float) * alpha
        output[mask] = blend[mask].astype(np.uint8)
        return output

    @staticmethod
    def _draw_nav_overlay(
        frame:    np.ndarray,
        nav_map:  np.ndarray,
        nav_dist: np.ndarray,
        alpha:    float = 0.50,
    ) -> np.ndarray:
        """
        Heat-map navigation overlay.
        Violet = near obstacle (danger zone).
        Cyan   = far from obstacle (safe passage).
        """
        output = frame.copy()
        if not nav_map.any():
            return output

        d  = nav_dist / 255.0          # (H,W) in [0,1]
        # Hue transitions 270 deg (violet) -> 180 deg (cyan) as d increases
        h  = 270.0 + (180.0 - 270.0) * d
        h_ = h / 60.0
        i  = h_.astype(int) % 6
        f  = h_ - i.astype(float)
        v  = np.ones_like(d)
        p  = np.zeros_like(d)
        q  = 1.0 - f
        t  = f

        r_ch = np.zeros_like(d)
        g_ch = np.zeros_like(d)
        b_ch = np.zeros_like(d)

        for idx, (rc, gc, bc) in enumerate([
            (v, t, p),   # i=0
            (q, v, p),   # i=1
            (p, v, t),   # i=2
            (p, q, v),   # i=3
            (t, p, v),   # i=4
            (v, p, q),   # i=5
        ]):
            m_i = (i == idx) & nav_map
            r_ch[m_i] = rc[m_i]
            g_ch[m_i] = gc[m_i]
            b_ch[m_i] = bc[m_i]

        nav_color = np.stack([
            (b_ch * 255).astype(np.uint8),
            (g_ch * 255).astype(np.uint8),
            (r_ch * 255).astype(np.uint8),
        ], axis=2)

        blend = output.astype(float) * (1 - alpha) + nav_color.astype(float) * alpha
        output[nav_map] = blend[nav_map].astype(np.uint8)
        return output

    # -------------------------------------------------------------------
    # Four-panel display
    # -------------------------------------------------------------------

    def draw_all_panels(
        self,
        frame:  np.ndarray,
        result: SegmentationResult,
    ) -> np.ndarray:
        """
        Compose a 2x2 panel image:

          +-------------------+-------------------+
          | Original Image    | Segmentation Mask |
          +-------------------+-------------------+
          | Workspace Mask    | Obstacle Map      |
          +-------------------+-------------------+

        Returns a single BGR numpy array.
        """
        import cv2

        H, W  = frame.shape[:2]
        font  = cv2.FONT_HERSHEY_SIMPLEX
        pad   = 6
        bar_h = 28

        def _labeled(img: np.ndarray, title: str, sub: str = "") -> np.ndarray:
            out = img.copy()
            cv2.rectangle(out, (0, 0), (W, bar_h), (15, 15, 22), -1)
            cv2.putText(out, title, (pad, bar_h - 8), font, 0.50,
                        (220, 220, 220), 1, cv2.LINE_AA)
            if sub:
                cv2.rectangle(out, (0, H - 20), (W, H), (15, 15, 22), -1)
                cv2.putText(out, sub, (pad, H - 6), font, 0.36,
                            (150, 150, 150), 1, cv2.LINE_AA)
            return out

        orig_ann = frame.copy()
        for reg in result.workspace_regions:
            cv2.drawContours(orig_ann, [reg.contour], -1, _COLOR_WORKSPACE, 2)
            cx, cy = reg.centroid
            cv2.circle(orig_ann, (cx, cy), 5, _COLOR_WORKSPACE, -1)

        p_orig = _labeled(
            orig_ann,
            "Original Image",
            f"{result.mode_label}  |  {result.elapsed_ms:.0f}ms"
        )
        p_seg  = _labeled(
            result.segmentation_bgr,
            "Segmentation Mask",
            f"workspace={result.workspace_pct:.1%}  obstacle={result.obstacle_pct:.1%}"
        )
        p_ws   = _labeled(
            result.workspace_mask_bgr,
            "Workspace Mask  (Valid Placement Region)",
            f"area={result.workspace_area_px:,}px  regions={len(result.workspace_regions)}"
        )
        p_obs  = _labeled(
            result.obstacle_map_bgr,
            "Obstacle Map  (Invalid Placement Region)",
            f"blocked={result.obstacle_area_px:,}px"
        )

        top    = np.concatenate([p_orig, p_seg],  axis=1)
        bottom = np.concatenate([p_ws,   p_obs],  axis=1)
        return np.concatenate([top, bottom], axis=0)

    def draw_nav_map_panel(
        self,
        frame:  np.ndarray,
        result: SegmentationResult,
    ) -> np.ndarray:
        """Return a labelled robot navigation map panel."""
        import cv2
        out = result.nav_map_bgr.copy()
        H   = out.shape[0]
        cv2.rectangle(out, (0, 0), (out.shape[1], 28), (15, 15, 22), -1)
        cv2.putText(out, "Robot Navigation Map", (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.rectangle(out, (0, H - 20), (out.shape[1], H), (15, 15, 22), -1)
        cv2.putText(out, "Violet=near obstacle  Cyan=safe passage",
                    (6, H - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                    (150, 150, 150), 1, cv2.LINE_AA)
        return out

    # -------------------------------------------------------------------
    # Synthetic demo (no model needed)
    # -------------------------------------------------------------------

    def _synthetic_result(self, frame: np.ndarray) -> SegmentationResult:
        """
        Generate a procedurally-correct synthetic segmentation for demo mode.

        Layout simulated on the frame:
          - Desk surface occupying lower 60% of frame
          - Three rectangular book obstacles on the desk
          - A laptop/monitor obstacle in the centre
          - User/hand region in lower-right corner
          - Free workspace zones between obstacles

        Clearly labelled [SYNTHETIC DEMO MODE] - no neural model used.
        """
        t0   = time.time()
        H, W = frame.shape[:2]

        raw_mask = np.zeros((H, W), dtype=np.int32)   # background = wall (0)

        # Desk surface
        dy1, dy2 = int(H * 0.35), H
        dx1, dx2 = int(W * 0.08), int(W * 0.92)
        raw_mask[dy1:dy2, dx1:dx2] = 33   # ADE20K: desk

        # Three book obstacles
        book_specs = [
            (0.12, 0.42, 0.14, 0.32),   # (x_frac, y_frac, w_frac, h_frac)
            (0.35, 0.38, 0.18, 0.36),
            (0.62, 0.45, 0.14, 0.30),
        ]
        for xf, yf, wf, hf in book_specs:
            by1, by2 = int(H * yf), int(H * (yf + hf))
            bx1, bx2 = int(W * xf), int(W * (xf + wf))
            raw_mask[by1:by2, bx1:bx2] = 67   # ADE20K: book

        # Laptop/monitor in centre
        raw_mask[int(H*0.36):int(H*0.54), int(W*0.40):int(W*0.60)] = 74  # computer

        # User/hand region bottom-right
        raw_mask[int(H*0.74):H, int(W*0.70):W] = 12   # person

        # Build semantic masks
        workspace_mask = self._build_mask(raw_mask, self.workspace_class_ids)
        obstacle_map   = self._build_mask(raw_mask, self.obstacle_class_ids)
        user_mask      = self._build_mask(raw_mask, USER_CLASS_IDS)

        workspace_mask &= ~obstacle_map

        workspace_mask = self._morph_clean(workspace_mask)
        obstacle_map   = self._morph_clean(obstacle_map)

        invalid_mask = obstacle_map | user_mask | ~workspace_mask
        nav_map      = workspace_mask & ~obstacle_map & ~user_mask
        nav_dist     = self._distance_map(nav_map, obstacle_map)

        workspace_regions = self._extract_regions(workspace_mask)

        total         = H * W
        workspace_pct = float(workspace_mask.sum()) / total
        obstacle_pct  = float(obstacle_map.sum())   / total

        unique, counts = np.unique(raw_mask, return_counts=True)
        class_areas    = {int(u): int(c) for u, c in zip(unique, counts)}

        label_mask    = self._build_label_mask(raw_mask, workspace_mask, obstacle_map, user_mask)
        seg_bgr       = self._colorise_segmentation(raw_mask, H, W)
        workspace_bgr = self._overlay(frame, workspace_mask, _COLOR_WORKSPACE)
        obstacle_bgr  = self._overlay(frame, obstacle_map,   _COLOR_OBSTACLE)
        nav_bgr       = self._draw_nav_overlay(frame, nav_map, nav_dist)

        elapsed = (time.time() - t0) * 1000

        return SegmentationResult(
            raw_mask           = raw_mask,
            label_mask         = label_mask,
            workspace_mask     = workspace_mask,
            obstacle_map       = obstacle_map,
            invalid_mask       = invalid_mask,
            nav_map            = nav_map,
            nav_distance_map   = nav_dist,
            segmentation_bgr   = seg_bgr,
            workspace_mask_bgr = workspace_bgr,
            obstacle_map_bgr   = obstacle_bgr,
            nav_map_bgr        = nav_bgr,
            workspace_regions  = workspace_regions,
            class_areas        = class_areas,
            workspace_pct      = workspace_pct,
            obstacle_pct       = obstacle_pct,
            mode               = "synthetic",
            mode_label         = "[SYNTHETIC DEMO MODE] No model loaded",
            elapsed_ms         = elapsed,
            frame_shape        = frame.shape,
        )

    # -------------------------------------------------------------------
    # Custom checkpoint interface (fine-tune upgrade path)
    # -------------------------------------------------------------------

    def set_custom_checkpoint(self, path_or_repo: str) -> None:
        """
        Switch to a fine-tuned custom checkpoint at runtime.

        Resets the currently loaded model. Call load() again to activate.

        Parameters
        ----------
        path_or_repo : local filesystem path OR HuggingFace repo ID.
                       Examples:
                         "models/my_workspace_segformer"
                         "myorg/workspace-segformer-v1"
        """
        self.custom_checkpoint = path_or_repo
        self.mode              = SegmenterMode.NOT_LOADED
        self._model            = None
        self._processor        = None
        log.info(f"Custom checkpoint set: {path_or_repo!r}  |  Call load() to activate.")

    def set_workspace_classes(self, class_ids: List[int]) -> None:
        """Override the ADE20K class IDs treated as workspace surface."""
        self.workspace_class_ids = list(class_ids)

    def set_obstacle_classes(self, class_ids: List[int]) -> None:
        """Override the ADE20K class IDs treated as obstacles."""
        self.obstacle_class_ids = list(class_ids)

    # -------------------------------------------------------------------
    # Label / catalog helpers
    # -------------------------------------------------------------------

    def get_label(self, class_id: int) -> str:
        """Return string label for an ADE20K class_id."""
        return self._id2label.get(
            class_id, ADE20K_LABELS.get(class_id, f"class_{class_id}")
        )

    def detected_workspace_classes(self, result: SegmentationResult) -> List[str]:
        return [
            self.get_label(cid) for cid in self.workspace_class_ids
            if result.class_areas.get(cid, 0) > 0
        ]

    def detected_obstacle_classes(self, result: SegmentationResult) -> List[str]:
        return [
            self.get_label(cid) for cid in self.obstacle_class_ids
            if result.class_areas.get(cid, 0) > 0
        ]

    # -------------------------------------------------------------------
    # Static helpers
    # -------------------------------------------------------------------

    @staticmethod
    def _build_mode_label(mode: str) -> str:
        return {
            "custom":     "[CUSTOM checkpoint]",
            "pretrained": "[WARNING] PRETRAINED / DEMO - generic ADE20K, not workspace-specific",
            "synthetic":  "[SYNTHETIC DEMO MODE] No model loaded",
            "not_loaded": "[NOT LOADED]",
        }.get(mode, mode)

    # -------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self.mode != SegmenterMode.NOT_LOADED

    @property
    def is_pretrained_mode(self) -> bool:
        return self.mode == SegmenterMode.PRETRAINED

    @property
    def is_custom_mode(self) -> bool:
        return self.mode == SegmenterMode.CUSTOM

    @property
    def is_synthetic_mode(self) -> bool:
        return self.mode == SegmenterMode.SYNTHETIC

    @property
    def total_frames(self) -> int:
        return self._total_frames

    @property
    def load_time_seconds(self) -> float:
        return self._load_time_s

    def __repr__(self) -> str:
        return (
            f"WorkspaceSegmenter("
            f"mode={self.mode.value!r}, "
            f"model={self.pretrained_model!r}, "
            f"frames={self._total_frames})"
        )
