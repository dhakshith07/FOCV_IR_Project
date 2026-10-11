"""
simulation/book.py
Book dataclass — the fundamental data unit of the workspace organizer.

Every physical book on the desk is represented as a Book instance.
Coordinates are in centimetres from the desk's bottom-left origin.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Tuple


# ── Book colour palette (one per slot, used for visualisation) ─────────────
BOOK_COLORS: list[str] = [
    "#e53e3e",   # red
    "#3182ce",   # blue
    "#38a169",   # green
    "#d69e2e",   # gold
    "#805ad5",   # purple
    "#dd6b20",   # orange
    "#00b5d8",   # cyan
    "#d53f8c",   # pink
]


@dataclass
class Book:
    """
    Represents a single book on the virtual desk.

    Attributes
    ----------
    id              : Unique identifier (e.g. 'B01')
    title           : Full book title
    x               : Centre X position on desk (cm, 0 = left edge)
    y               : Centre Y position on desk (cm, 0 = front/user edge)
    width           : Visual width on desk (cm)
    height          : Visual height on desk (cm)
    usage_frequency : How many times the book was used (session total)
    last_used       : Timestamp of most recent interaction
    current_context : Subject context this book belongs to
    color           : Hex colour for visualisation
    """

    id:               str
    title:            str
    x:                float
    y:                float
    width:            float             = 9.0
    height:           float             = 13.0
    usage_frequency:  int               = 0
    last_used:        Optional[datetime] = field(default=None)
    current_context:  str               = "general"
    color:            str               = "#4299e1"

    def __post_init__(self) -> None:
        if self.last_used is None:
            self.last_used = datetime.now()

    # ── Geometry ──────────────────────────────────────────────────────────
    @property
    def rect(self) -> Tuple[float, float, float, float]:
        """Axis-aligned bounding box (x0, y0, x1, y1)."""
        hw, hh = self.width / 2, self.height / 2
        return (self.x - hw, self.y - hh, self.x + hw, self.y + hh)

    @property
    def corners(self) -> Tuple[list, list]:
        """
        Closed polygon vertices (xs, ys) suitable for Plotly fill traces.
        Vertices: BL → BR → TR → TL → BL (closed).
        """
        x0, y0, x1, y1 = self.rect
        xs = [x0, x1, x1, x0, x0]
        ys = [y0, y0, y1, y1, y0]
        return xs, ys

    def contains_point(self, px: float, py: float) -> bool:
        x0, y0, x1, y1 = self.rect
        return x0 <= px <= x1 and y0 <= py <= y1

    def overlaps(self, other: "Book", margin: float = 1.5) -> bool:
        """
        Return True if this book's bounding box (plus `margin` on each side)
        intersects with `other`'s bounding box.
        """
        ax0, ay0, ax1, ay1 = self.rect
        bx0, by0, bx1, by1 = other.rect
        # Expand self by margin
        ax0 -= margin; ay0 -= margin
        ax1 += margin; ay1 += margin
        return not (ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0)

    def overlaps_rect(
        self, rx0: float, ry0: float, rx1: float, ry1: float, margin: float = 0.0
    ) -> bool:
        """Return True if this book overlaps an arbitrary rectangle."""
        ax0, ay0, ax1, ay1 = self.rect
        return not (
            ax1 + margin < rx0
            or rx1 < ax0 - margin
            or ay1 + margin < ry0
            or ry1 < ay0 - margin
        )

    # ── Serialisation ─────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        return {
            "id":               self.id,
            "title":            self.title,
            "x":                round(self.x, 2),
            "y":                round(self.y, 2),
            "width":            self.width,
            "height":           self.height,
            "usage_frequency":  self.usage_frequency,
            "last_used":        self.last_used.strftime("%Y-%m-%d %H:%M:%S") if self.last_used else None,
            "current_context":  self.current_context,
            "color":            self.color,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Book":
        data = dict(data)
        if data.get("last_used") and isinstance(data["last_used"], str):
            data["last_used"] = datetime.fromisoformat(data["last_used"])
        return cls(**data)

    def __repr__(self) -> str:
        return f"Book(id={self.id!r}, title={self.title!r}, x={self.x:.1f}, y={self.y:.1f})"
