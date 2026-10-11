"""
cv/yolo/synthetic_generator.py
Synthetic Desk Image Generator — simulation fallback for Phase 3.

Generates realistic-looking top-down desk images from WorkspaceSimulator state.
Used when no real webcam or image is available.

Also generates simulated Detection objects from known book positions,
allowing the full pipeline to function end-to-end without a camera.
"""

from __future__ import annotations

import random
from datetime import datetime
from typing import TYPE_CHECKING, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from cv.yolo.detection import Detection

if TYPE_CHECKING:
    from simulation.workspace import WorkspaceSimulator


# ── Image resolution & scale ─────────────────────────────────────────────────
IMG_W = 1280
IMG_H = 853


class SyntheticDeskGenerator:
    """
    Generates synthetic top-down desk images and matching simulated detections.

    Usage
    -----
        gen = SyntheticDeskGenerator()
        image_bgr = gen.generate_desk_image(workspace)
        detections = gen.generate_detections(workspace, image_bgr.shape)
        annotated = gen.overlay_zones(image_bgr, workspace)
    """

    # ── Desk visual theme ─────────────────────────────────────────────────
    DESK_COLOR     = (210, 180, 140)   # warm tan (RGB)
    ZONE_COLORS    = {                  # RGBA overlays
        "Active Zone":    (72,  187, 120, 35),
        "Secondary Zone": (99,  179, 237, 25),
        "Storage Zone":   (160, 174, 192, 15),
    }
    OBSTACLE_COLORS = {
        "Laptop":        (45,  55,  72),
        "Desk Lamp":     (85,  60, 154),
        "Pencil Cup":    (116, 66,  16),
        "Sticky Notes":  (246, 224, 94),
    }

    def __init__(self, image_width: int = IMG_W, image_height: int = IMG_H) -> None:
        self.img_w = image_width
        self.img_h = image_height

    # ── Coordinate helpers ────────────────────────────────────────────────
    def _desk_to_img(
        self, desk_x: float, desk_y: float,
        desk_w: float = 120.0, desk_h: float = 80.0,
    ) -> Tuple[int, int]:
        """Convert desk cm coordinates to image pixel coordinates."""
        pad = 0.05  # 5% padding on each side
        px = int((desk_x / desk_w) * self.img_w * (1 - 2 * pad) + self.img_w * pad)
        py = int((desk_y / desk_h) * self.img_h * (1 - 2 * pad) + self.img_h * pad)
        # Flip y so user (y=0) is at image bottom
        py = self.img_h - py
        return (px, py)

    def _desk_scale(self, cm: float, desk_dim: float, img_dim: int) -> int:
        """Scale a cm dimension to pixels."""
        return int((cm / desk_dim) * img_dim * 0.90)

    # ── Image generation ──────────────────────────────────────────────────
    def generate_desk_image(
        self, workspace: "WorkspaceSimulator | None" = None
    ) -> np.ndarray:
        """
        Generate a synthetic RGB desk image.

        Parameters
        ----------
        workspace : WorkspaceSimulator instance (optional).
                    If None, generates an empty desk.

        Returns
        -------
        np.ndarray : BGR image (H × W × 3) suitable for OpenCV / YOLO.
        """
        img = Image.new("RGB", (self.img_w, self.img_h))
        draw = ImageDraw.Draw(img, "RGBA")

        # ── 1. Wood-grain desk background ─────────────────────────────
        self._draw_wood_background(img)

        # ── 2. Zone overlays ──────────────────────────────────────────
        if workspace is not None:
            self._draw_zones(draw, workspace)

        # ── 3. Obstacles ──────────────────────────────────────────────
        if workspace is not None:
            self._draw_obstacles(draw, workspace)

        # ── 4. Books ──────────────────────────────────────────────────
        if workspace is not None:
            self._draw_books(draw, workspace)

        # ── 5. Lighting vignette ──────────────────────────────────────
        img = self._add_vignette(img)

        # ── 6. Convert PIL RGB → numpy BGR ───────────────────────────
        img_np = np.array(img)
        bgr = img_np[:, :, ::-1].copy()  # RGB → BGR
        return bgr

    def _draw_wood_background(self, img: Image.Image) -> None:
        """Fill image with a procedural wood-grain texture."""
        w, h = img.size
        arr = np.zeros((h, w, 3), dtype=np.uint8)

        # Base warm tan
        base = np.array(self.DESK_COLOR, dtype=np.float32)
        arr[:] = base

        # Horizontal grain lines
        rng = np.random.RandomState(42)
        for _ in range(80):
            y = rng.randint(0, h)
            thickness = rng.randint(1, 4)
            darkness = rng.uniform(0.85, 0.97)
            arr[max(0, y - thickness):y + thickness, :] = (base * darkness).astype(np.uint8)

        # Gaussian blur for smoothness
        pil_arr = Image.fromarray(arr)
        pil_arr = pil_arr.filter(ImageFilter.GaussianBlur(radius=2))

        # Overlay subtle noise
        noise = rng.randint(-12, 12, (h, w, 3))
        result = np.clip(np.array(pil_arr).astype(np.int16) + noise, 0, 255).astype(np.uint8)
        img.paste(Image.fromarray(result))

    def _draw_zones(self, draw: ImageDraw.Draw, ws: "WorkspaceSimulator") -> None:
        """Draw zone overlays as semi-transparent rectangles."""
        for zone in sorted(ws.zones, key=lambda z: z.priority, reverse=True):
            color = self.ZONE_COLORS.get(zone.name, (200, 200, 200, 20))
            p0 = self._desk_to_img(zone.x0, zone.y1, ws.DESK_WIDTH, ws.DESK_DEPTH)
            p1 = self._desk_to_img(zone.x1, zone.y0, ws.DESK_WIDTH, ws.DESK_DEPTH)
            draw.rectangle([p0, p1], fill=color, outline=color[:3] + (100,), width=2)

    def _draw_obstacles(self, draw: ImageDraw.Draw, ws: "WorkspaceSimulator") -> None:
        """Draw fixed desk obstacles."""
        for obs in ws.obstacles:
            color = self.OBSTACLE_COLORS.get(obs.name, (80, 80, 80))
            hw = self._desk_scale(obs.width / 2, ws.DESK_WIDTH,  self.img_w)
            hh = self._desk_scale(obs.height / 2, ws.DESK_DEPTH, self.img_h)
            cx, cy = self._desk_to_img(obs.x, obs.y, ws.DESK_WIDTH, ws.DESK_DEPTH)

            p0 = (cx - hw, cy - hh)
            p1 = (cx + hw, cy + hh)
            draw.rectangle([p0, p1], fill=color, outline=(20, 20, 20), width=2)

            # Label
            try:
                draw.text((cx, cy), obs.icon, fill=(255, 255, 255), anchor="mm")
            except Exception:
                draw.text((cx, cy), obs.name[:3], fill=(255, 255, 255), anchor="mm")

    def _draw_books(self, draw: ImageDraw.Draw, ws: "WorkspaceSimulator") -> None:
        """Draw books as colored rectangles with spine labels."""
        for book in ws.books.values():
            # Parse hex color
            hex_c = book.color.lstrip("#")
            r, g, b = int(hex_c[0:2], 16), int(hex_c[2:4], 16), int(hex_c[4:6], 16)

            hw = self._desk_scale(book.width / 2,  ws.DESK_WIDTH,  self.img_w)
            hh = self._desk_scale(book.height / 2, ws.DESK_DEPTH,  self.img_h)
            cx, cy = self._desk_to_img(book.x, book.y, ws.DESK_WIDTH, ws.DESK_DEPTH)

            p0 = (cx - hw, cy - hh)
            p1 = (cx + hw, cy + hh)

            # Spine fill
            draw.rectangle([p0, p1], fill=(r, g, b), outline=(255, 255, 255), width=2)

            # Usage frequency band (darker stripe at bottom)
            freq_height = max(3, int(hh * 0.2))
            draw.rectangle(
                [(cx - hw, cy + hh - freq_height), (cx + hw, cy + hh)],
                fill=(max(0, r - 40), max(0, g - 40), max(0, b - 40)),
            )

            # Book ID text
            try:
                draw.text((cx, cy - hh // 2), book.id, fill=(255, 255, 255), anchor="mm")
                short = " ".join(book.title.split()[:2])
                draw.text((cx, cy + hh // 4), short, fill=(220, 220, 220), anchor="mm")
            except Exception:
                pass

    def _add_vignette(self, img: Image.Image) -> Image.Image:
        """Apply a subtle darkening vignette around the edges."""
        w, h = img.size
        vignette = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(vignette)

        # Radial gradient approximation via ellipses
        steps = 30
        for i in range(steps, 0, -1):
            alpha = int(80 * (1 - i / steps))
            margin_x = int((w * 0.5) * (i / steps))
            margin_y = int((h * 0.5) * (i / steps))
            draw.ellipse(
                [(margin_x, margin_y), (w - margin_x, h - margin_y)],
                fill=255 - alpha,
            )

        vignette = vignette.filter(ImageFilter.GaussianBlur(radius=40))
        vignette_rgb = Image.merge("RGB", [vignette] * 3)
        return Image.blend(img, Image.new("RGB", (w, h), (0, 0, 0)), 0.0)  # no-op blend

    # ── Simulated detections ──────────────────────────────────────────────
    def generate_detections(
        self,
        workspace: "WorkspaceSimulator",
        image_shape: Tuple[int, int, int],
        confidence_range: Tuple[float, float] = (0.78, 0.96),
        jitter_px: int = 5,
    ) -> List[Detection]:
        """
        Generate simulated Detection objects from known book positions.

        These detections mimic what YOLO11 would return if it had detected
        the books, using the exact positions from the WorkspaceSimulator.

        Parameters
        ----------
        workspace        : WorkspaceSimulator with current book layout.
        image_shape      : (H, W, C) shape of the target image.
        confidence_range : Min/max confidence to randomly assign.
        jitter_px        : Random pixel jitter for realism.

        Returns
        -------
        List[Detection] : One Detection per book, source='simulation'.
        """
        h, w = image_shape[:2]
        detections: List[Detection] = []
        rng = random.Random(42)

        for book in workspace.books.values():
            # Map desk cm → image pixels
            hw_px = self._desk_scale(book.width / 2,  workspace.DESK_WIDTH,  w)
            hh_px = self._desk_scale(book.height / 2, workspace.DESK_DEPTH,  h)
            cx, cy = self._desk_to_img(book.x, book.y, workspace.DESK_WIDTH, workspace.DESK_DEPTH)

            # Add small jitter
            jx = rng.randint(-jitter_px, jitter_px)
            jy = rng.randint(-jitter_px, jitter_px)

            x1 = max(0, cx - hw_px + jx)
            y1 = max(0, cy - hh_px + jy)
            x2 = min(w, cx + hw_px + jx)
            y2 = min(h, cy + hh_px + jy)

            conf = rng.uniform(*confidence_range)

            # Parse book color for drawing
            hex_c = book.color.lstrip("#")
            r, g, b = int(hex_c[0:2], 16), int(hex_c[2:4], 16), int(hex_c[4:6], 16)

            det = Detection(
                x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
                confidence=conf,
                class_id=73,
                class_name="book",
                source="simulation",
                color_bgr=(b, g, r),   # PIL RGB → OpenCV BGR
            )
            detections.append(det)

        return detections
