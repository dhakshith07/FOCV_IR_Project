"""
robotics/ik/jacobian_ik.py
===========================
Phase 11 — Jacobian-based Inverse Kinematics for a simulated 6-DOF robot arm.

PURPOSE
-------
Convert a desired end-effector position (and optionally orientation) into a
set of joint angles θ = [θ₁, θ₂, θ₃, θ₄, θ₅, θ₆] using an iterative
Jacobian-based solver.

EVERY step of the algorithm is visible and documented — no black-box IK
library is used as the primary implementation.

ARM MODEL — Denavit-Hartenberg (DH) Convention
----------------------------------------------
The simulated arm uses the standard DH convention.  Each joint i is
described by four DH parameters:

    aᵢ   — link length      (x-axis offset)
    αᵢ   — link twist       (x-axis rotation, radians)
    dᵢ   — link offset      (z-axis offset)
    θᵢ   — joint angle      (z-axis rotation, variable, radians)

The homogeneous transformation from frame i-1 to frame i is:

    Tᵢ(θᵢ) = Rz(θᵢ) · Tz(dᵢ) · Tx(aᵢ) · Rx(αᵢ)

         ⎡ cos θ   -sin θ·cos α   sin θ·sin α   a·cos θ ⎤
    Tᵢ = ⎢ sin θ    cos θ·cos α  -cos θ·sin α   a·sin θ ⎥
         ⎢  0         sin α          cos α          d    ⎥
         ⎣  0          0              0             1    ⎦

FORWARD KINEMATICS
------------------
    T₀₆ = T₁ · T₂ · T₃ · T₄ · T₅ · T₆

The end-effector position p = T₀₆[:3, 3].
The end-effector z-axis (approach direction) = T₀₆[:3, 2].

JACOBIAN (Geometric)
--------------------
For a revolute joint i the column of the position Jacobian is:

    Jᵥᵢ = zᵢ₋₁ × (p_ee − pᵢ₋₁)

where:
    zᵢ₋₁   = rotation axis of joint i (z-column of Tᵢ₋₁)
    p_ee    = current end-effector position
    pᵢ₋₁   = origin of frame i-1

The orientation Jacobian (angular velocity) column is:

    Jωᵢ = zᵢ₋₁

The full 6×n Jacobian stacks Jᵥ (3×n) above Jω (3×n):

    J = [Jᵥ]    ← shape (6, n)
        [Jω]

ITERATIVE SOLVER (Damped Least Squares)
---------------------------------------
At each iteration k:

    error_pos   = p_target − p_current          (3,)
    error_ori   = α_target − α_current          (3,)  (optional)
    error       = [error_pos; error_ori]         (6,)  or (3,) if pos-only

    J†  = Jᵀ (J Jᵀ + λ² I)⁻¹               (damped pseudo-inverse)

    Δθ  = J† · error                             (n,)

    θ   ← θ + α · Δθ                            (step with learning rate)

    Clamp θ to joint limits.

Convergence when |error_pos| ≤ tolerance.

CLASSES
-------
DHParam         — One DH row (a, α, d, θ_offset).
JointLimit      — (lower, upper) bounds in radians.
ArmConfig       — Full arm definition: DH table + joint limits.
IKConfig        — Solver hyper-parameters.
IKResult        — Rich result: joint angles, pose, error, iterations, status.
ForwardKinematics  — Pure FK engine (visible DH matrix math).
JacobianCalculator — Geometric Jacobian builder (visible cross-product code).
JacobianIK         — Iterative IK solver (Damped Least Squares, visible).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from utils.logger import get_logger

log = get_logger(__name__)


# ============================================================================
# DH Parameters and Joint Limits
# ============================================================================

@dataclass
class DHParam:
    """
    One row of the Denavit-Hartenberg parameter table.

    Attributes
    ----------
    a       : Link length (metres) — offset along x_i.
    alpha   : Link twist  (radians) — rotation about x_i.
    d       : Link offset (metres) — offset along z_{i-1}.
    theta0  : Constant offset added to the variable joint angle (radians).
              Allows the zero-configuration to be set ergonomically.
    """
    a:      float = 0.0
    alpha:  float = 0.0
    d:      float = 0.0
    theta0: float = 0.0


@dataclass
class JointLimit:
    """
    Joint angle bounds.

    Attributes
    ----------
    lower : Minimum angle (radians).
    upper : Maximum angle (radians).
    """
    lower: float
    upper: float

    def clamp(self, angle: float) -> float:
        """Return angle clamped to [lower, upper]."""
        return max(self.lower, min(self.upper, angle))

    def is_valid(self, angle: float) -> bool:
        return self.lower <= angle <= self.upper


# ============================================================================
# ArmConfig  — simulated 6-DOF desk robot
# ============================================================================

@dataclass
class ArmConfig:
    """
    Complete description of the simulated multi-joint arm.

    Default: 6-DOF articulated arm sized for a desk workspace (~70 cm reach).

    DH table (6 revolute joints):

      Joint │  a (m)  │  α (rad)      │  d (m)  │  θ_offset
      ──────┼─────────┼───────────────┼─────────┼──────────
        1   │  0.000  │  +π/2         │  0.150  │  0
        2   │  0.250  │   0           │  0.000  │  0
        3   │  0.200  │   0           │  0.000  │  0
        4   │  0.000  │  +π/2         │  0.100  │  0
        5   │  0.000  │  -π/2         │  0.000  │  0
        6   │  0.000  │   0           │  0.060  │  0

    Joint limits (radians):
        ±π for joints 1,4,6 (continuous-like)
        ±π/2 for joints 2,5
        ±5π/6 for joint 3

    Attributes
    ----------
    dh_params    : List of DHParam (one per joint).
    joint_limits : List of JointLimit (one per joint).
    dof          : Degrees of freedom (len of dh_params).
    """
    dh_params:    List[DHParam]
    joint_limits: List[JointLimit]

    @property
    def dof(self) -> int:
        return len(self.dh_params)

    @classmethod
    def desk_arm_6dof(cls) -> "ArmConfig":
        """
        Return the default 6-DOF simulated desk arm.

        The arm is sized roughly like a Kinova Gen3 Lite:
            shoulder  → elbow → wrist
        with 15 cm base height, 25 cm upper arm, 20 cm forearm,
        10 cm wrist-offset, 6 cm tool flange.
        """
        dh = [
            DHParam(a=0.000, alpha= math.pi/2, d=0.150, theta0=0.0),
            DHParam(a=0.250, alpha= 0.0,       d=0.000, theta0=0.0),
            DHParam(a=0.200, alpha= 0.0,       d=0.000, theta0=0.0),
            DHParam(a=0.000, alpha= math.pi/2, d=0.100, theta0=0.0),
            DHParam(a=0.000, alpha=-math.pi/2, d=0.000, theta0=0.0),
            DHParam(a=0.000, alpha= 0.0,       d=0.060, theta0=0.0),
        ]
        lim = [
            JointLimit(-math.pi,        math.pi),       # joint 1
            JointLimit(-math.pi/2,      math.pi/2),     # joint 2
            JointLimit(-5*math.pi/6,    5*math.pi/6),   # joint 3
            JointLimit(-math.pi,        math.pi),       # joint 4
            JointLimit(-math.pi/2,      math.pi/2),     # joint 5
            JointLimit(-math.pi,        math.pi),       # joint 6
        ]
        return cls(dh_params=dh, joint_limits=lim)

    @classmethod
    def from_config_dict(cls, d: dict) -> "ArmConfig":
        """
        Build ArmConfig from the project config.yaml 'jacobian_ik' section.

        Expected shape::

            jacobian_ik:
              dof: 6
              joint_limits:     # degrees
                - [-180, 180]
                ...
        """
        dof = int(d.get("dof", 6))
        raw_limits = d.get("joint_limits", [[-180, 180]] * dof)

        limits = [
            JointLimit(
                lower = math.radians(pair[0]),
                upper = math.radians(pair[1]),
            )
            for pair in raw_limits
        ]
        # Always use the default DH geometry; limits come from config
        default = cls.desk_arm_6dof()
        return cls(dh_params=default.dh_params[:dof], joint_limits=limits[:dof])


# ============================================================================
# IKConfig  — solver hyper-parameters
# ============================================================================

@dataclass
class IKConfig:
    """
    Hyper-parameters for the iterative Jacobian IK solver.

    Attributes
    ----------
    max_iterations  : Maximum number of Newton iterations.
    tolerance_m     : Convergence threshold on end-effector position error (metres).
    damping         : Levenberg-Marquardt damping factor λ (prevents singularity).
    step_size       : Learning rate α applied to each Δθ update.
    solve_orientation: If True the full 6-DOF task (pos + ori) is solved;
                      if False only the 3-DOF position task is solved.
    """
    max_iterations:     int   = 200
    tolerance_m:        float = 0.001         # 1 mm
    damping:            float = 0.01
    step_size:          float = 1.0
    solve_orientation:  bool  = False

    @classmethod
    def from_dict(cls, d: dict) -> "IKConfig":
        fields = {f for f in cls.__dataclass_fields__}
        kwargs = {}
        # Handle tolerance_mm → tolerance_m conversion
        if "tolerance_mm" in d:
            kwargs["tolerance_m"] = float(d["tolerance_mm"]) / 1000.0
        if "damping_lambda" in d:
            kwargs["damping"] = float(d["damping_lambda"])
        for k, v in d.items():
            if k in fields:
                kwargs[k] = v
        return cls(**kwargs)


# ============================================================================
# IKResult  — rich result container
# ============================================================================

@dataclass
class IKResult:
    """
    Output of one IK solve call.

    Attributes
    ----------
    converged           : True if position error < tolerance.
    joint_angles_rad    : Final joint angles (radians), shape (n,).
    joint_angles_deg    : Final joint angles (degrees), shape (n,).
    target_position     : Desired end-effector position (x, y, z) [metres].
    final_position      : Achieved end-effector position after solve [metres].
    position_error_m    : |target − final| (metres).
    iterations          : Number of iterations consumed.
    solve_time_s        : Wall-clock time for the solve.
    per_iter_error      : List of position errors at each iteration.
    within_joint_limits : True if all final angles are within bounds.
    """
    converged:           bool
    joint_angles_rad:    np.ndarray
    joint_angles_deg:    np.ndarray
    target_position:     np.ndarray
    final_position:      np.ndarray
    position_error_m:    float
    iterations:          int
    solve_time_s:        float
    per_iter_error:      List[float]
    within_joint_limits: bool

    # ------------------------------------------------------------------
    def display(self) -> str:
        """
        Formatted console-friendly display of all requested outputs.

        Shows:
          Target pose
          Current end-effector pose
          Joint angles (°)
          Position error (mm)
          Iterations
          Convergence status
        """
        lines = [
            "",
            "═" * 56,
            "  JACOBIAN IK  —  SOLVE RESULT",
            "═" * 56,
            f"  Convergence      : {'✓ YES' if self.converged else '✗ NO'}",
            f"  Iterations       : {self.iterations}",
            f"  Solve Time       : {self.solve_time_s * 1000:.2f} ms",
            "─" * 56,
            "  TARGET POSE",
            f"    x = {self.target_position[0]:+.4f} m",
            f"    y = {self.target_position[1]:+.4f} m",
            f"    z = {self.target_position[2]:+.4f} m",
            "─" * 56,
            "  ACHIEVED END-EFFECTOR POSE",
            f"    x = {self.final_position[0]:+.4f} m",
            f"    y = {self.final_position[1]:+.4f} m",
            f"    z = {self.final_position[2]:+.4f} m",
            "─" * 56,
            f"  POSITION ERROR   : {self.position_error_m * 1000:.3f} mm",
            "─" * 56,
            "  JOINT ANGLES",
        ]
        for i, (r, d) in enumerate(zip(self.joint_angles_rad, self.joint_angles_deg)):
            lines.append(f"    θ{i+1} = {d:+8.3f}°  ({r:+.5f} rad)")
        lines += [
            "─" * 56,
            f"  Joint limits OK  : {'YES' if self.within_joint_limits else 'NO'}",
            "═" * 56,
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "converged":            self.converged,
            "iterations":           self.iterations,
            "position_error_mm":    round(self.position_error_m * 1000, 4),
            "solve_time_ms":        round(self.solve_time_s * 1000, 3),
            "joint_angles_deg":     [round(d, 4) for d in self.joint_angles_deg.tolist()],
            "joint_angles_rad":     [round(r, 6) for r in self.joint_angles_rad.tolist()],
            "target_position":      self.target_position.tolist(),
            "final_position":       self.final_position.tolist(),
            "within_joint_limits":  self.within_joint_limits,
        }


# ============================================================================
# ForwardKinematics  — explicit DH matrix math
# ============================================================================

class ForwardKinematics:
    """
    Computes the forward kinematics of a DH-parameterised arm.

    Each joint transformation is computed from the DH matrix formula
    written out explicitly in the code below — no external FK library.

    Parameters
    ----------
    arm : ArmConfig describing the robot geometry.
    """

    def __init__(self, arm: ArmConfig) -> None:
        self.arm = arm

    # ------------------------------------------------------------------
    # DH matrix for one joint
    # ------------------------------------------------------------------

    @staticmethod
    def dh_matrix(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
        """
        Compute the 4×4 homogeneous DH transformation matrix.

        T = Rz(θ) · Tz(d) · Tx(a) · Rx(α)

        Expanded form:

            ⎡ cos θ   -sin θ·cos α    sin θ·sin α   a·cos θ ⎤
            ⎢ sin θ    cos θ·cos α   -cos θ·sin α   a·sin θ ⎥
            ⎢  0         sin α           cos α          d    ⎥
            ⎣  0          0               0             1    ⎦

        Parameters
        ----------
        a, alpha, d : DH parameters (metres / radians).
        theta       : Variable joint angle (radians).

        Returns
        -------
        np.ndarray of shape (4, 4).
        """
        ct = math.cos(theta)   # cos θ
        st = math.sin(theta)   # sin θ
        ca = math.cos(alpha)   # cos α
        sa = math.sin(alpha)   # sin α

        return np.array([
            [ct,  -st * ca,   st * sa,   a * ct],
            [st,   ct * ca,  -ct * sa,   a * st],
            [0.0,  sa,         ca,        d     ],
            [0.0,  0.0,        0.0,       1.0   ],
        ], dtype=np.float64)

    # ------------------------------------------------------------------
    # Full FK: T₀ₙ and per-joint frames
    # ------------------------------------------------------------------

    def compute(
        self,
        joint_angles: np.ndarray,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Compute the complete forward kinematics.

        Starting from the identity (world frame), multiply the DH matrices
        for each joint left-to-right:

            T₀₁ = T₁(θ₁)
            T₀₂ = T₀₁ · T₂(θ₂)
            ...
            T₀ₙ = T₀(ₙ₋₁) · Tₙ(θₙ)

        Parameters
        ----------
        joint_angles : np.ndarray of shape (n,), angles in radians.

        Returns
        -------
        T_ee      : (4, 4) homogeneous transform from base to end-effector.
        frames    : List of (4, 4) transforms T₀ᵢ for i = 0 … n
                    (index 0 = world frame = identity, index n = T_ee).
        """
        n      = self.arm.dof
        T_curr = np.eye(4, dtype=np.float64)
        frames = [T_curr.copy()]           # T₀₀ = I  (world frame)

        for i in range(n):
            dh    = self.arm.dh_params[i]
            theta = joint_angles[i] + dh.theta0   # apply constant offset

            T_i   = self.dh_matrix(dh.a, dh.alpha, dh.d, theta)
            T_curr = T_curr @ T_i          # accumulate left product
            frames.append(T_curr.copy())   # T₀ᵢ

        return T_curr, frames   # T_curr == T₀ₙ == T_ee

    def end_effector_position(self, joint_angles: np.ndarray) -> np.ndarray:
        """Return only the (x, y, z) end-effector position."""
        T_ee, _ = self.compute(joint_angles)
        return T_ee[:3, 3].copy()

    def end_effector_transform(self, joint_angles: np.ndarray) -> np.ndarray:
        """Return the full 4×4 end-effector transform."""
        T_ee, _ = self.compute(joint_angles)
        return T_ee


# ============================================================================
# JacobianCalculator  — geometric Jacobian (explicit cross-product code)
# ============================================================================

class JacobianCalculator:
    """
    Builds the geometric Jacobian of a DH-parameterised revolute arm.

    The Jacobian J is a (6 × n) matrix:

        J = [Jᵥ]    Jᵥ is the (3 × n) linear-velocity Jacobian
            [Jω]    Jω is the (3 × n) angular-velocity Jacobian

    For revolute joint i (1-indexed):

        Jᵥᵢ = z_{i-1} × (p_ee − p_{i-1})    ← cross product
        Jωᵢ = z_{i-1}                         ← rotation axis

    where:
        z_{i-1} = T_{0,i-1}[:3, 2]    (z-column of the i-1 frame)
        p_{i-1} = T_{0,i-1}[:3, 3]    (origin of frame i-1)
        p_ee    = T_{0,n}[:3, 3]      (end-effector origin)

    All maths is explicit — no robot-toolbox or sympy dependency.

    Parameters
    ----------
    arm : ArmConfig.
    fk  : ForwardKinematics engine.
    """

    def __init__(self, arm: ArmConfig, fk: ForwardKinematics) -> None:
        self.arm = arm
        self.fk  = fk

    def compute(
        self,
        joint_angles: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the (6 × n) geometric Jacobian.

        Steps
        -----
        1. Run full FK to get all intermediate frames T₀₀ … T₀ₙ.
        2. Extract end-effector origin p_ee = T₀ₙ[:3, 3].
        3. For each joint i (0-indexed):
             z_{i-1} = frames[i][:3, 2]    (z-axis of frame i-1)
             p_{i-1} = frames[i][:3, 3]    (origin of frame i-1)
             Jᵥᵢ = z_{i-1} × (p_ee − p_{i-1})   (cross product, shape (3,))
             Jωᵢ = z_{i-1}                         (shape (3,))
        4. Stack [Jᵥᵢ; Jωᵢ] as the i-th column.

        Parameters
        ----------
        joint_angles : np.ndarray of shape (n,).

        Returns
        -------
        J : np.ndarray of shape (6, n).
        """
        n         = self.arm.dof
        T_ee, frames = self.fk.compute(joint_angles)
        p_ee      = T_ee[:3, 3]           # end-effector position

        J = np.zeros((6, n), dtype=np.float64)

        for i in range(n):
            # Frame i-1 is at index i in `frames` (frames[0] = world = I)
            T_prev = frames[i]
            z_prev = T_prev[:3, 2]         # z-axis of frame i-1
            p_prev = T_prev[:3, 3]         # origin of frame i-1

            # Linear velocity component: Jᵥᵢ = z_{i-1} × (p_ee − p_{i-1})
            dp         = p_ee - p_prev
            J_v_i      = np.cross(z_prev, dp)

            # Angular velocity component: Jωᵢ = z_{i-1}
            J_omega_i  = z_prev

            J[:3, i] = J_v_i
            J[3:, i] = J_omega_i

        return J

    def position_jacobian(self, joint_angles: np.ndarray) -> np.ndarray:
        """Return only the (3 × n) linear-velocity part of J."""
        return self.compute(joint_angles)[:3, :]


# ============================================================================
# JacobianIK  — iterative Damped Least Squares IK solver
# ============================================================================

class JacobianIK:
    """
    Iterative Jacobian Inverse Kinematics solver.

    Algorithm: Damped Least Squares (Levenberg-Marquardt style).

    At each iteration:

        1. FK: compute current end-effector pose from current θ.
        2. Error:
               error_pos = p_target − p_current          (3,)
               error_ori = n_target − n_current           (3,) if ori enabled
               error     = concat(error_pos, [error_ori]) (3 or 6,)
        3. Jacobian:
               J = JacobianCalculator.compute(θ)          (6, n) or (3, n)
        4. Damped pseudo-inverse:
               J† = Jᵀ (J Jᵀ + λ² I)⁻¹
        5. Joint update:
               Δθ = J† · error
               θ  ← θ + α · Δθ
        6. Clamp θ to joint limits.
        7. Check convergence: |error_pos| ≤ tolerance.

    No external IK library is used.  The algorithm above is the complete
    implementation; every line is readable and explainable.

    Parameters
    ----------
    arm    : ArmConfig (DH table + joint limits).
    config : IKConfig (solver hyper-parameters).
    """

    STATUS = "PHASE_11"

    def __init__(
        self,
        arm:    Optional[ArmConfig] = None,
        config: Optional[IKConfig]  = None,
    ) -> None:
        self.arm = arm or ArmConfig.desk_arm_6dof()
        self.cfg = config or IKConfig()

        # Build sub-components (all visible, no black box)
        self.fk  = ForwardKinematics(self.arm)
        self.jac = JacobianCalculator(self.arm, self.fk)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(
        self,
        target_position:    np.ndarray,
        target_orientation: Optional[np.ndarray] = None,
        initial_angles:     Optional[np.ndarray] = None,
    ) -> IKResult:
        """
        Run the iterative IK solver.

        Parameters
        ----------
        target_position    : Desired end-effector (x, y, z) in metres. Shape (3,).
        target_orientation : Desired end-effector z-axis direction. Shape (3,).
                             Only used if cfg.solve_orientation=True.
        initial_angles     : Starting joint angles (radians). Shape (n,).
                             Defaults to zeros.

        Returns
        -------
        IKResult — see class docstring for fields.
        """
        t_start = time.perf_counter()

        target_pos = np.array(target_position, dtype=np.float64).flatten()
        if target_pos.shape != (3,):
            raise ValueError(f"target_position must have shape (3,), got {target_pos.shape}")

        n = self.arm.dof

        # Initialise joint angles
        if initial_angles is not None:
            theta = np.array(initial_angles, dtype=np.float64).flatten()
            if theta.shape != (n,):
                raise ValueError(f"initial_angles must have shape ({n},)")
        else:
            theta = np.zeros(n, dtype=np.float64)

        # Optional orientation target
        solve_ori    = self.cfg.solve_orientation and (target_orientation is not None)
        target_ori   = None
        if solve_ori:
            target_ori = np.array(target_orientation, dtype=np.float64).flatten()
            target_ori = target_ori / (np.linalg.norm(target_ori) + 1e-12)

        # ── Main iterative loop ──────────────────────────────────────
        per_iter_error: List[float] = []
        converged       = False
        iteration       = 0
        task_dim        = 6 if solve_ori else 3

        for iteration in range(1, self.cfg.max_iterations + 1):

            # ── Step 1: Forward kinematics ───────────────────────────
            T_ee, frames = self.fk.compute(theta)
            p_current    = T_ee[:3, 3]       # current end-effector position

            # ── Step 2: Error vector ─────────────────────────────────
            error_pos = target_pos - p_current          # (3,)
            pos_error_norm = float(np.linalg.norm(error_pos))
            per_iter_error.append(pos_error_norm)

            if solve_ori:
                n_current = T_ee[:3, 2]              # z-axis of end-effector
                error_ori = target_ori - n_current   # (3,)
                error     = np.concatenate([error_pos, error_ori])  # (6,)
            else:
                error = error_pos                    # (3,)

            # ── Step 3: Jacobian ─────────────────────────────────────
            J_full = self.jac.compute(theta)         # (6, n)
            if solve_ori:
                J = J_full                           # use all 6 rows
            else:
                J = J_full[:3, :]                    # position-only (3, n)

            # ── Step 4: Damped pseudo-inverse ────────────────────────
            #
            #   J† = Jᵀ (J Jᵀ + λ² I)⁻¹
            #
            lam_sq  = self.cfg.damping ** 2
            JJT     = J @ J.T                              # (task_dim, task_dim)
            damped  = JJT + lam_sq * np.eye(task_dim)      # add damping
            J_pinv  = J.T @ np.linalg.inv(damped)          # (n, task_dim)

            # ── Step 5: Joint update ─────────────────────────────────
            #
            #   Δθ = J† · error
            #   θ  ← θ + α · Δθ
            #
            delta_theta = J_pinv @ error                   # (n,)
            theta       = theta + self.cfg.step_size * delta_theta

            # ── Step 6: Clamp to joint limits ────────────────────────
            for j in range(n):
                theta[j] = self.arm.joint_limits[j].clamp(theta[j])

            # ── Step 7: Convergence check ────────────────────────────
            if pos_error_norm <= self.cfg.tolerance_m:
                converged = True
                break

        # ── Final FK for the achieved pose ───────────────────────────
        T_final, _  = self.fk.compute(theta)
        p_final     = T_final[:3, 3]
        final_error = float(np.linalg.norm(target_pos - p_final))

        # Check joint limits
        within_limits = all(
            self.arm.joint_limits[j].is_valid(theta[j])
            for j in range(n)
        )

        solve_time = time.perf_counter() - t_start

        if converged:
            log.info(
                f"IK converged in {iteration} iter | "
                f"error={final_error*1000:.3f} mm | time={solve_time*1000:.2f} ms"
            )
        else:
            log.warning(
                f"IK did not converge after {iteration} iter | "
                f"error={final_error*1000:.3f} mm"
            )

        return IKResult(
            converged           = converged,
            joint_angles_rad    = theta.copy(),
            joint_angles_deg    = np.degrees(theta),
            target_position     = target_pos.copy(),
            final_position      = p_final.copy(),
            position_error_m    = final_error,
            iterations          = iteration,
            solve_time_s        = solve_time,
            per_iter_error      = per_iter_error,
            within_joint_limits = within_limits,
        )

    # ------------------------------------------------------------------
    # Convenience wrappers
    # ------------------------------------------------------------------

    def reachable(self, target_position: np.ndarray) -> bool:
        """Return True if a target is reachable (IK converges)."""
        result = self.solve(target_position)
        return result.converged

    def workspace_sample(
        self,
        n_samples: int = 1000,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Sample n_samples reachable end-effector positions via random FK.

        Draws uniformly random joint angles within limits and runs FK.
        Returns an (n_samples, 3) array of reachable positions.
        """
        rng = np.random.default_rng(seed)
        positions = np.zeros((n_samples, 3), dtype=np.float64)
        for k in range(n_samples):
            angles = np.array([
                rng.uniform(lim.lower, lim.upper)
                for lim in self.arm.joint_limits
            ])
            T, _ = self.fk.compute(angles)
            positions[k] = T[:3, 3]
        return positions

    # ------------------------------------------------------------------
    # Class-level factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config_dict(cls, d: dict) -> "JacobianIK":
        """Build from the 'jacobian_ik' section of config.yaml."""
        arm = ArmConfig.from_config_dict(d)
        cfg = IKConfig.from_dict(d)
        return cls(arm=arm, config=cfg)
