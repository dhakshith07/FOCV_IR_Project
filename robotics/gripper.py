"""
robotics/gripper.py
===================
Phase 12 -- Simulated Parallel-Jaw Gripper.

The gripper has two fingers that open/close symmetrically around the
tool-flange centre.  It tracks the grasp state and the book currently
held (if any).

Finger geometry (default):
    max_open   : 0.12 m  (finger tips 12 cm apart)
    min_closed : 0.01 m  (finger tips 1 cm apart)
    finger_length : 0.06 m

State machine:
    OPEN   -- fingers fully open, ready to approach a book
    CLOSED -- fingers gripping a book
    MOVING -- transitioning between states (animated)

The gripper does NOT use PyBullet constraints; it tracks which BookDef
is held and updates that book's position every step.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

import numpy as np


# ── Gripper state ─────────────────────────────────────────────────────────────

class GripperState(Enum):
    OPEN   = "open"
    CLOSED = "closed"
    MOVING = "moving"


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class GripperConfig:
    """Gripper geometry and motion parameters."""
    max_open:      float = 0.12    # metres (finger-tip gap when fully open)
    min_closed:    float = 0.010   # metres (finger-tip gap when gripping)
    finger_length: float = 0.060   # metres (finger rod length)
    finger_width:  float = 0.012   # metres (for rendering only)
    move_speed:    float = 0.05    # metres per second (open/close speed)
    grasp_dist:    float = 0.030   # metres: max EE-to-book distance to grasp
    color_open:    tuple = (0.7, 0.7, 0.8, 1.0)
    color_closed:  tuple = (0.3, 0.9, 0.4, 1.0)


# ── Gripper ───────────────────────────────────────────────────────────────────

class Gripper:
    """
    Simulated parallel-jaw gripper attached to the robot wrist.

    Public API
    ----------
    open()                           -- command gripper to open
    close()                          -- command gripper to close
    step(dt)                         -- advance animation by dt seconds
    try_grasp(book, ee_pos) -> bool  -- attempt to pick up a book
    release()                        -- drop held book
    update_held_book(ee_pos)         -- move held book to follow EE each step
    finger_positions(ee_transform)   -- get 3-D tip positions of both fingers
    info() -> dict                   -- status summary for UI
    """

    STATUS = "PHASE_12"

    def __init__(self, config: Optional[GripperConfig] = None) -> None:
        self.cfg = config or GripperConfig()

        # Kinematic state
        self._gap:        float        = self.cfg.max_open
        self._state:      GripperState = GripperState.OPEN
        self._target_gap: float        = self.cfg.max_open

        # Grasped book reference (set externally via try_grasp)
        self.held_book = None   # BookDef instance or None

        # Metrics
        self.total_grasps:   int = 0
        self.total_releases: int = 0

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def state(self) -> GripperState:
        return self._state

    @property
    def gap(self) -> float:
        """Current finger-tip separation [m]."""
        return self._gap

    @property
    def is_open(self) -> bool:
        return self._state == GripperState.OPEN

    @property
    def is_closed(self) -> bool:
        return self._state == GripperState.CLOSED

    @property
    def is_holding(self) -> bool:
        return self.held_book is not None

    @property
    def finger_color(self) -> tuple:
        return self.cfg.color_closed if self.is_closed else self.cfg.color_open

    @property
    def open_fraction(self) -> float:
        """0.0 = fully closed, 1.0 = fully open."""
        denom = self.cfg.max_open - self.cfg.min_closed
        return (self._gap - self.cfg.min_closed) / denom if denom > 0 else 0.0

    # ── Commands ─────────────────────────────────────────────────────────────

    def open(self) -> None:
        """Command gripper to open (begins animation)."""
        self._target_gap = self.cfg.max_open
        self._state = GripperState.MOVING

    def close(self) -> None:
        """Command gripper to close (begins animation)."""
        self._target_gap = self.cfg.min_closed
        self._state = GripperState.MOVING

    def step(self, dt: float = 0.05) -> None:
        """
        Advance the gripper animation by dt seconds.

        Moves finger gap towards target_gap at move_speed.
        Settles state to OPEN or CLOSED when target reached.
        """
        if self._state != GripperState.MOVING:
            return

        delta = self.cfg.move_speed * dt
        diff  = self._target_gap - self._gap

        if abs(diff) <= delta:
            self._gap = self._target_gap
            if abs(self._gap - self.cfg.max_open) < 1e-4:
                self._state = GripperState.OPEN
            else:
                self._state = GripperState.CLOSED
        else:
            self._gap += delta * (1 if diff > 0 else -1)

    # ── Grasp / release ───────────────────────────────────────────────────────

    def try_grasp(self, book, ee_pos: np.ndarray) -> bool:
        """
        Attempt to grasp a book.

        Parameters
        ----------
        book   : BookDef instance to attempt grasping.
        ee_pos : Current end-effector position [m].

        Returns
        -------
        True if the grasp succeeded (EE close enough to book top).
        """
        if self.held_book is not None:
            return False  # already holding something

        book_top = np.array(book.position, dtype=np.float64)
        book_top[2] += book.size[2] / 2

        dist = float(np.linalg.norm(ee_pos - book_top))
        if dist <= self.cfg.grasp_dist:
            self.held_book  = book
            book.is_held    = True
            self._state     = GripperState.CLOSED
            self._gap       = self.cfg.min_closed
            self.total_grasps += 1
            return True
        return False

    def release(self) -> None:
        """Release the currently held book (if any)."""
        if self.held_book is not None:
            self.held_book.is_held = False
            self.held_book = None
            self.total_releases += 1
        self.open()

    def force_grasp(self, book) -> None:
        """
        Force-attach a book without distance checking.

        Used during task execution when IK moves the arm to the book position
        and we want to guarantee the grasp succeeds.
        """
        if self.held_book is not None:
            self.release()
        self.held_book  = book
        book.is_held    = True
        self._state     = GripperState.CLOSED
        self._gap       = self.cfg.min_closed
        self.total_grasps += 1

    def update_held_book(self, ee_pos: np.ndarray) -> None:
        """
        Move the held book to follow the end-effector.

        Called every simulation step after FK is recomputed.
        The book centre is placed half-height below the EE.
        """
        if self.held_book is None:
            return
        new_pos = ee_pos.copy()
        new_pos[2] -= self.held_book.size[2] / 2 + 0.03
        self.held_book.position = new_pos.tolist()

    # ── Geometry for rendering ────────────────────────────────────────────────

    def finger_positions(
        self, ee_transform: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the 3-D base and tip positions of both fingers.

        The fingers are offset +/- gap/2 along the EE x-axis and
        extend forward along the EE z-axis.

        Parameters
        ----------
        ee_transform : (4, 4) end-effector homogeneous transform.

        Returns
        -------
        left_base, left_tip, right_base, right_tip : np.ndarray (3,) each.
        """
        R   = ee_transform[:3, :3]
        pos = ee_transform[:3,  3]

        x_axis = R[:, 0]   # gripper "width" axis
        z_axis = R[:, 2]   # gripper "approach" axis

        half_gap = self._gap / 2

        left_base  = pos + x_axis * half_gap
        right_base = pos - x_axis * half_gap
        left_tip   = left_base  + z_axis * self.cfg.finger_length
        right_tip  = right_base + z_axis * self.cfg.finger_length

        return left_base, left_tip, right_base, right_tip

    # ── Info ─────────────────────────────────────────────────────────────────

    def info(self) -> dict:
        return {
            "state":          self._state.value,
            "gap_m":          round(self._gap, 4),
            "open_fraction":  round(self.open_fraction, 3),
            "holding":        self.held_book.book_id if self.held_book else None,
            "total_grasps":   self.total_grasps,
            "total_releases": self.total_releases,
        }

    def reset(self) -> None:
        """Reset gripper to initial open state."""
        if self.held_book is not None:
            self.held_book.is_held = False
            self.held_book = None
        self._gap        = self.cfg.max_open
        self._target_gap = self.cfg.max_open
        self._state      = GripperState.OPEN

    def __repr__(self) -> str:
        h = self.held_book.book_id if self.held_book else "none"
        return f"Gripper(state={self._state.value}, gap={self._gap:.3f}m, holding={h})"
