"""
simulation/zones.py
Zone and Obstacle definitions for the virtual desk workspace.

Coordinate system (all values in centimetres):
  Origin : bottom-left corner of desk surface
  X-axis : left (0) → right (120)
  Y-axis : front/user (0) → back (80)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


# ═══════════════════════════════════════════════════════════════════════════════
#  Zone
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Zone:
    """
    A named rectangular region of the desk surface.

    Attributes
    ----------
    name            : Human-readable label (e.g. 'Active Zone')
    x0, y0         : Bottom-left corner (cm)
    x1, y1         : Top-right corner (cm)
    color           : Fill colour (CSS / hex)
    border_color    : Border colour
    priority        : 1 = highest priority (closest to user)
    label_position  : (x, y) for the zone label annotation
    """

    name:           str
    x0:             float
    y0:             float
    x1:             float
    y1:             float
    color:          str
    border_color:   str
    priority:       int   = 3
    description:    str   = ""

    # ── Geometry ──────────────────────────────────────────────────────────
    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2)

    @property
    def label_pos(self) -> Tuple[float, float]:
        """Top-left corner for the zone label annotation."""
        return (self.x0 + 1.5, self.y1 - 3.5)

    def contains(self, x: float, y: float) -> bool:
        """Return True if (x, y) falls inside this zone."""
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1

    def contains_rect(self, rx0: float, ry0: float, rx1: float, ry1: float) -> bool:
        """Return True if an entire rectangle is contained within this zone."""
        return rx0 >= self.x0 and ry0 >= self.y0 and rx1 <= self.x1 and ry1 <= self.y1

    def clamp_center(
        self, cx: float, cy: float, half_w: float, half_h: float
    ) -> Tuple[float, float]:
        """
        Clamp object centre (cx, cy) so the object stays fully inside this zone.
        half_w / half_h are the object's half-dimensions.
        """
        cx = max(self.x0 + half_w, min(self.x1 - half_w, cx))
        cy = max(self.y0 + half_h, min(self.y1 - half_h, cy))
        return cx, cy

    def __repr__(self) -> str:
        return f"Zone(name={self.name!r}, x=[{self.x0},{self.x1}], y=[{self.y0},{self.y1}])"


# ═══════════════════════════════════════════════════════════════════════════════
#  Obstacle
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Obstacle:
    """
    A fixed, immovable object on the desk (laptop, lamp, cup, etc.).
    Books may not overlap with obstacles.

    Attributes
    ----------
    name            : Label (e.g. 'Laptop')
    x               : Centre X (cm)
    y               : Centre Y (cm)
    width           : cm
    height          : cm
    color           : Fill colour for visualisation
    icon            : Emoji icon for visualisation
    """

    name:   str
    x:      float
    y:      float
    width:  float
    height: float
    color:  str     = "#4a5568"
    icon:   str     = "📦"

    @property
    def x0(self) -> float:
        return self.x - self.width / 2

    @property
    def y0(self) -> float:
        return self.y - self.height / 2

    @property
    def x1(self) -> float:
        return self.x + self.width / 2

    @property
    def y1(self) -> float:
        return self.y + self.height / 2

    @property
    def rect(self) -> Tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)

    def blocks(self, cx: float, cy: float, hw: float, hh: float, margin: float = 1.5) -> bool:
        """
        Return True if a rectangle centred at (cx, cy) with half-dims (hw, hh)
        would overlap this obstacle.
        """
        return not (
            cx + hw + margin < self.x0
            or self.x1 < cx - hw - margin
            or cy + hh + margin < self.y0
            or self.y1 < cy - hh - margin
        )

    def __repr__(self) -> str:
        return f"Obstacle(name={self.name!r}, x={self.x}, y={self.y})"


# ═══════════════════════════════════════════════════════════════════════════════
#  Factory helpers
# ═══════════════════════════════════════════════════════════════════════════════

def create_default_zones() -> list[Zone]:
    """
    Build the three canonical desk zones for a 120 × 80 cm workspace.

      Active Zone    — front-centre; highest priority; daily-use items
      Secondary Zone — mid-range; moderate priority; session items
      Storage Zone   — full desk; lowest priority; background items
    """
    return [
        Zone(
            name="Storage Zone",
            x0=2.0, y0=2.0, x1=118.0, y1=78.0,
            color="rgba(160,174,192,0.07)",
            border_color="rgba(160,174,192,0.25)",
            priority=3,
            description="Rarely used objects — any clear area on the desk",
        ),
        Zone(
            name="Secondary Zone",
            x0=8.0, y0=8.0, x1=112.0, y1=58.0,
            color="rgba(99,179,237,0.10)",
            border_color="rgba(99,179,237,0.35)",
            priority=2,
            description="Occasionally used objects — within comfortable reach",
        ),
        Zone(
            name="Active Zone",
            x0=18.0, y0=6.0, x1=102.0, y1=36.0,
            color="rgba(72,187,120,0.14)",
            border_color="rgba(72,187,120,0.45)",
            priority=1,
            description="Frequently used objects — directly in front of user",
        ),
    ]


def create_default_obstacles() -> list[Obstacle]:
    """
    Build the fixed obstacles that represent objects always present on the desk
    (laptop, desk lamp, pencil cup, and sticky-note pad).
    """
    return [
        Obstacle(
            name="Laptop",
            x=72.0, y=58.0, width=30.0, height=20.0,
            color="#2d3748", icon="💻",
        ),
        Obstacle(
            name="Desk Lamp",
            x=8.0,  y=72.0, width=10.0, height=10.0,
            color="#553c9a", icon="💡",
        ),
        Obstacle(
            name="Pencil Cup",
            x=108.0, y=68.0, width=8.0, height=8.0,
            color="#744210", icon="✏️",
        ),
        Obstacle(
            name="Sticky Notes",
            x=25.0, y=56.0, width=16.0, height=12.0,
            color="#f6e05e", icon="📝",
        ),
    ]
