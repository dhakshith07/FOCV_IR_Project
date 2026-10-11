"""
robotics/robot_model.py
=======================
Phase 12 — Software-based Multi-Joint Robot Arm Model.

Defines the full kinematic structure of a 6-DOF articulated desk-robot arm
used by the simulation.  All geometry is expressed in metres.

ARM GEOMETRY  (DH convention)
-----------------------------------------------
  Joint | Role           |  a (m)  |  alpha (rad) |  d (m)
  ------|----------------|---------|--------------|-------
    1   | Base-Rotation  |  0.000  |  +pi/2       |  0.150
    2   | Shoulder-Pitch |  0.250  |   0          |  0.000
    3   | Elbow-Pitch    |  0.200  |   0          |  0.000
    4   | Forearm-Yaw    |  0.000  |  +pi/2       |  0.100
    5   | Wrist-Pitch    |  0.000  |  -pi/2       |  0.000
    6   | Wrist-Roll     |  0.000  |   0          |  0.060

Joint limits (radians):
    J1: +-pi    J2: +-pi/2    J3: +-5pi/6
    J4: +-pi    J5: +-pi/2    J6: +-pi
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class JointDef:
    """Single revolute joint definition (DH parameters + motion limits)."""
    name:         str
    a:            float = 0.0
    alpha:        float = 0.0
    d:            float = 0.0
    theta0:       float = 0.0
    limit_lower:  float = -math.pi
    limit_upper:  float =  math.pi
    max_velocity: float =  1.0

    def clamp(self, angle: float) -> float:
        return max(self.limit_lower, min(self.limit_upper, float(angle)))

    def is_valid(self, angle: float) -> bool:
        return self.limit_lower <= angle <= self.limit_upper

    @property
    def range_deg(self) -> float:
        return math.degrees(self.limit_upper - self.limit_lower)


class RobotArm:
    """
    6-DOF revolute arm with DH forward kinematics.

    Provides:
      - Forward kinematics via DH matrices
      - End-effector position and full 4x4 transform
      - Per-joint frame chain for visualisation
      - Joint angle get/set with clamping
      - Workspace sampling
    """

    STATUS = "PHASE_12"

    def __init__(
        self,
        joints: Optional[List[JointDef]] = None,
        base_position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        self.joints: List[JointDef] = joints or self._default_joints()
        self.base_position = np.array(base_position, dtype=np.float64)
        self.dof: int = len(self.joints)
        self._angles: np.ndarray = np.zeros(self.dof, dtype=np.float64)

    @staticmethod
    def _default_joints() -> List[JointDef]:
        return [
            JointDef("Base-Rotation",  a=0.000, alpha=math.pi/2,   d=0.150,
                     limit_lower=-math.pi,       limit_upper=math.pi,       max_velocity=1.0),
            JointDef("Shoulder-Pitch", a=0.250, alpha=0.0,          d=0.000,
                     limit_lower=-math.pi/2,     limit_upper=math.pi/2,     max_velocity=1.0),
            JointDef("Elbow-Pitch",    a=0.200, alpha=0.0,          d=0.000,
                     limit_lower=-5*math.pi/6,   limit_upper=5*math.pi/6,   max_velocity=1.2),
            JointDef("Forearm-Yaw",    a=0.000, alpha=math.pi/2,   d=0.100,
                     limit_lower=-math.pi,        limit_upper=math.pi,       max_velocity=1.5),
            JointDef("Wrist-Pitch",    a=0.000, alpha=-math.pi/2,  d=0.000,
                     limit_lower=-math.pi/2,      limit_upper=math.pi/2,     max_velocity=1.5),
            JointDef("Wrist-Roll",     a=0.000, alpha=0.0,          d=0.060,
                     limit_lower=-math.pi,         limit_upper=math.pi,       max_velocity=2.0),
        ]

    @property
    def angles(self) -> np.ndarray:
        return self._angles.copy()

    @angles.setter
    def angles(self, values: np.ndarray) -> None:
        arr = np.asarray(values, dtype=np.float64).flatten()
        for i, jd in enumerate(self.joints):
            arr[i] = jd.clamp(arr[i])
        self._angles = arr

    @property
    def angles_deg(self) -> np.ndarray:
        return np.degrees(self._angles)

    def set_angles(self, angles: np.ndarray) -> None:
        self.angles = angles

    def set_home(self) -> None:
        self._angles = np.zeros(self.dof, dtype=np.float64)

    @staticmethod
    def _dh_matrix(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
        ct, st = math.cos(theta), math.sin(theta)
        ca, sa = math.cos(alpha), math.sin(alpha)
        return np.array([
            [ct,  -st*ca,  st*sa,  a*ct],
            [st,   ct*ca, -ct*sa,  a*st],
            [0.0,  sa,     ca,     d   ],
            [0.0,  0.0,    0.0,    1.0 ],
        ], dtype=np.float64)

    def forward_kinematics(
        self, angles: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        if angles is None:
            angles = self._angles
        T_base = np.eye(4, dtype=np.float64)
        T_base[:3, 3] = self.base_position
        T_curr = T_base.copy()
        frames = [T_curr.copy()]
        for i, jd in enumerate(self.joints):
            theta = float(angles[i]) + jd.theta0
            T_i   = self._dh_matrix(jd.a, jd.alpha, jd.d, theta)
            T_curr = T_curr @ T_i
            frames.append(T_curr.copy())
        return T_curr, frames

    def end_effector_position(self, angles: Optional[np.ndarray] = None) -> np.ndarray:
        T_ee, _ = self.forward_kinematics(angles)
        return T_ee[:3, 3].copy()

    def end_effector_transform(self, angles: Optional[np.ndarray] = None) -> np.ndarray:
        T_ee, _ = self.forward_kinematics(angles)
        return T_ee

    def joint_positions(self, angles: Optional[np.ndarray] = None) -> List[np.ndarray]:
        _, frames = self.forward_kinematics(angles)
        return [f[:3, 3].copy() for f in frames]

    @property
    def max_reach(self) -> float:
        return sum(abs(jd.a) + abs(jd.d) for jd in self.joints)

    def workspace_samples(self, n: int = 500, seed: Optional[int] = 42) -> np.ndarray:
        rng = np.random.default_rng(seed)
        pts = np.zeros((n, 3), dtype=np.float64)
        for k in range(n):
            a = np.array([rng.uniform(jd.limit_lower, jd.limit_upper) for jd in self.joints])
            pts[k] = self.end_effector_position(a)
        return pts

    def joint_info(self) -> List[dict]:
        result = []
        for i, jd in enumerate(self.joints):
            result.append({
                "index": i + 1,
                "name": jd.name,
                "angle_deg": round(math.degrees(self._angles[i]), 2),
                "limit_lower_deg": round(math.degrees(jd.limit_lower), 1),
                "limit_upper_deg": round(math.degrees(jd.limit_upper), 1),
                "max_vel_rad_s": jd.max_velocity,
            })
        return result

    def __repr__(self) -> str:
        pos = self.end_effector_position()
        return f"RobotArm(dof={self.dof}, ee=({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f})m)"


@dataclass
class TableDef:
    """Rectangular table surface."""
    width:      float = 1.20
    depth:      float = 0.80
    height:     float = 0.02
    position:   Tuple[float, float, float] = (0.0, 0.35, -0.01)
    color_rgba: Tuple[float, float, float, float] = (0.55, 0.40, 0.25, 1.0)

    @property
    def half_extents(self) -> Tuple[float, float, float]:
        return (self.width/2, self.depth/2, self.height/2)

    @property
    def surface_z(self) -> float:
        return self.position[2] + self.height / 2


@dataclass
class BookDef:
    """A book on the table."""
    book_id:    str
    title:      str
    position:   List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    size:       Tuple[float, float, float] = (0.04, 0.15, 0.22)
    color_rgba: Tuple[float, float, float, float] = (0.2, 0.5, 0.9, 1.0)
    is_held:    bool = False
    target_pos: Optional[List[float]] = None

    @property
    def half_extents(self) -> Tuple[float, float, float]:
        return (self.size[0]/2, self.size[1]/2, self.size[2]/2)

    def grasp_position(self) -> np.ndarray:
        """Ideal EE position for grasping: 3 cm above the book top face."""
        p = np.array(self.position, dtype=np.float64)
        p[2] += self.size[2] / 2 + 0.03
        return p

    def place_position(self) -> Optional[np.ndarray]:
        if self.target_pos is None:
            return None
        return np.array(self.target_pos, dtype=np.float64)

    @property
    def top_z(self) -> float:
        return self.position[2] + self.size[2] / 2


@dataclass
class ObstacleDef:
    """Cylindrical obstacle on the table."""
    obs_id:     str
    position:   List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    radius:     float = 0.03
    height:     float = 0.12
    color_rgba: Tuple[float, float, float, float] = (0.8, 0.2, 0.2, 0.8)


@dataclass
class SceneConfig:
    """Complete scene: arm + table + books + obstacles."""
    arm:       RobotArm
    table:     TableDef
    books:     List[BookDef]
    obstacles: List[ObstacleDef]

    @classmethod
    def default_scene(cls) -> "SceneConfig":
        arm   = RobotArm(base_position=(0.0, 0.0, 0.0))
        table = TableDef()
        books = [
            BookDef("B1", "Machine Learning",
                    position=[-0.25, 0.30, 0.11],
                    color_rgba=(0.2, 0.5, 0.9, 1.0),
                    target_pos=[0.25, 0.50, 0.11]),
            BookDef("B2", "Computer Networks",
                    position=[0.00, 0.30, 0.11],
                    color_rgba=(0.2, 0.8, 0.4, 1.0),
                    target_pos=[0.00, 0.55, 0.11]),
            BookDef("B3", "Operating Systems",
                    position=[0.25, 0.30, 0.11],
                    color_rgba=(0.9, 0.5, 0.2, 1.0),
                    target_pos=[-0.25, 0.50, 0.11]),
        ]
        obstacles = [
            ObstacleDef("O1", position=[0.15, 0.45, 0.06],  radius=0.025, height=0.12,
                        color_rgba=(0.85, 0.25, 0.25, 0.85)),
            ObstacleDef("O2", position=[-0.15, 0.45, 0.06], radius=0.025, height=0.12,
                        color_rgba=(0.85, 0.25, 0.25, 0.85)),
        ]
        return cls(arm=arm, table=table, books=books, obstacles=obstacles)

    def get_book(self, book_id: str) -> Optional[BookDef]:
        for b in self.books:
            if b.book_id == book_id:
                return b
        return None
