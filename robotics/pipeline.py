"""
robotics/pipeline.py
====================
Phase 13 -- Integrated Planning Pipeline.

Connects three subsystems in strict order:

    Target Cartesian Position
        ↓
    Jacobian IK  (Phase 11)
        ↓
    Target Joint Configuration
        ↓
    Joint-Space RRT  (Phase 10, extended to N-D)
        ↓
    Collision-Free Joint-Space Path
        ↓
    Virtual Robot execution  (Phase 12)

ARCHITECTURE
------------
The RRT from Phase 10 works in a 2-D (x, y) Cartesian workspace.
Phase 13 wraps a generalised *joint-space* RRT that operates in the
6-D space θ ∈ ℝ⁶.  Each "configuration" is a 6-element joint-angle
vector (radians).  Collision checking is done by:

  1. FK: project the joint configuration to Cartesian (x, y) using
     the end-effector position.
  2. Check that Cartesian (x, y) point is collision-free in the
     Phase 10 ConfigSpace.
  3. Also enforce joint limits.

This means we reuse the existing Phase 10 CollisionChecker for the
Cartesian occupancy map while extending the RRT to handle N-D
joint space — no bypassing either subsystem.

CLASSES
-------
  PipelineStep          — Result of one pipeline stage
  PipelineResult        — Complete result of one target motion
  JointSpaceRRT         — N-D joint-space RRT using Phase-10 collision logic
  RobotPipeline         — Orchestrates IK → RRT → execution for each waypoint
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Phase 10 imports ──────────────────────────────────────────────────────────
from robotics.rrt.configuration_space import (
    Bounds2D, ConfigSpace, Obstacle, Config as Config2D,
)
from robotics.rrt.collision_checker import CollisionChecker, CollisionReport

# ── Phase 11 imports ──────────────────────────────────────────────────────────
from robotics.ik.jacobian_ik import (
    JacobianIK, ArmConfig, IKConfig, IKResult, ForwardKinematics,
)

# ── Phase 12 imports ──────────────────────────────────────────────────────────
from robotics.robot_model import RobotArm, SceneConfig, BookDef

from utils.logger import get_logger
log = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  Data containers
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PipelineStep:
    """Result of one stage in the pipeline."""
    stage:       str
    success:     bool
    duration_ms: float
    details:     dict = field(default_factory=dict)
    error:       str  = ""


@dataclass
class PipelineResult:
    """
    Complete result of one target-motion planning+execution pass.

    Attributes
    ----------
    target_pos       : Cartesian target (x, y, z) [m].
    ik_result        : Full IKResult from Phase 11.
    rrt_path_joints  : List of joint-angle vectors along the path (N x 6).
    rrt_path_cartesian: (x, y) projection of each RRT node [for display].
    rrt_n_nodes      : Total nodes in the RRT tree.
    rrt_n_waypoints  : Waypoints in the final path.
    rrt_length       : Total joint-space path length (radians).
    rrt_planning_ms  : RRT planning wall time.
    collision_free   : True if all path segments are collision-free.
    collision_status : Human-readable collision report.
    executed_steps   : Number of robot steps actually executed.
    final_ee_pos     : EE position after execution [m].
    final_error_m    : Distance from target [m].
    stages           : List of PipelineStep records.
    success          : True if IK + RRT + execution all succeeded.
    """
    target_pos:         np.ndarray
    ik_result:          Optional[IKResult]       = None
    rrt_path_joints:    List[np.ndarray]         = field(default_factory=list)
    rrt_path_cartesian: List[Tuple[float, float]] = field(default_factory=list)
    rrt_n_nodes:        int   = 0
    rrt_n_waypoints:    int   = 0
    rrt_length:         float = 0.0
    rrt_planning_ms:    float = 0.0
    collision_free:     bool  = True
    collision_status:   str   = "clear"
    executed_steps:     int   = 0
    final_ee_pos:       Optional[np.ndarray] = None
    final_error_m:      float = 0.0
    stages:             List[PipelineStep] = field(default_factory=list)
    success:            bool  = False

    def summary_lines(self) -> List[str]:
        lines = [
            f"Target : ({self.target_pos[0]:+.3f}, {self.target_pos[1]:+.3f}, {self.target_pos[2]:+.3f}) m",
        ]
        if self.ik_result:
            ik = self.ik_result
            lines += [
                f"IK     : {'✓ Converged' if ik.converged else '✗ Failed'} "
                f"in {ik.iterations} iters  error={ik.position_error_m*1000:.2f} mm",
                f"Joints : {', '.join(f'{d:.1f}°' for d in ik.joint_angles_deg.tolist())}",
            ]
        lines += [
            f"RRT    : {self.rrt_n_waypoints} waypoints  "
            f"{self.rrt_n_nodes} nodes  {self.rrt_length:.3f} rad  {self.rrt_planning_ms:.1f} ms",
            f"Collision: {self.collision_status}",
            f"Exec   : {self.executed_steps} steps  "
            f"final_err={self.final_error_m*1000:.2f} mm",
            f"Overall: {'SUCCESS' if self.success else 'FAILED'}",
        ]
        return lines


# ═══════════════════════════════════════════════════════════════════════════════
#  Joint-Space RRT  (N-dimensional, Phase-10 collision logic)
# ═══════════════════════════════════════════════════════════════════════════════

class JointSpaceRRT:
    """
    N-dimensional RRT operating in joint-angle space (radians).

    For each candidate configuration θ, collision is tested by:
      1. FK → end-effector (x, y) in Cartesian space.
      2. Phase-10 CollisionChecker.is_point_free(x, y).
      3. Joint limits check.

    Parameters
    ----------
    arm_config   : ArmConfig with DH params and joint limits.
    cspace_2d    : Phase-10 ConfigSpace (2-D Cartesian, metres).
    max_iter     : Maximum RRT iterations.
    step_size    : Max angular step per expansion (radians).
    goal_bias    : Probability of sampling the goal directly.
    goal_thr     : Goal-reached threshold in joint space (radians).
    seed         : Random seed.
    """

    def __init__(
        self,
        arm_config:  ArmConfig,
        cspace_2d:   ConfigSpace,
        max_iter:    int   = 3000,
        step_size:   float = 0.15,    # ~8.6° per step
        goal_bias:   float = 0.12,
        goal_thr:    float = 0.10,    # ~5.7°
        seed:        Optional[int] = 42,
    ) -> None:
        self.arm_config  = arm_config
        self.cspace_2d   = cspace_2d
        self.checker_2d  = CollisionChecker(cspace_2d, n_checks=15)
        self.fk          = ForwardKinematics(arm_config)
        self.max_iter    = max_iter
        self.step_size   = step_size
        self.goal_bias   = goal_bias
        self.goal_thr    = goal_thr
        self._rng        = np.random.default_rng(seed)

        self.dof = arm_config.dof
        self._lo = np.array([lim.lower for lim in arm_config.joint_limits])
        self._hi = np.array([lim.upper for lim in arm_config.joint_limits])

    # ── Collision in joint space (via FK → 2-D Cartesian) ────────────────────

    def _ee_xy(self, theta: np.ndarray) -> Config2D:
        """Project joint angles to (x, y) end-effector position."""
        T, _ = self.fk.compute(theta)
        return (float(T[0, 3]), float(T[1, 3]))

    def _is_config_free(self, theta: np.ndarray) -> bool:
        """
        True if joint configuration θ is collision-free.

        Checks:
          (a) Joint limits
          (b) Cartesian EE inside workspace bounds
          (c) EE not inside any obstacle
        """
        # (a) Joint limits
        if np.any(theta < self._lo) or np.any(theta > self._hi):
            return False
        # (b) + (c) Cartesian check
        xy = self._ee_xy(theta)
        return self.checker_2d.is_point_free(xy)

    def _is_segment_free(
        self, q_from: np.ndarray, q_to: np.ndarray, n_check: int = 12
    ) -> bool:
        """
        Interpolate `n_check` configs between q_from and q_to and
        verify each is collision-free.
        """
        for k in range(n_check + 1):
            t = k / n_check
            q = q_from + t * (q_to - q_from)
            if not self._is_config_free(q):
                return False
        return True

    # ── Sampling ─────────────────────────────────────────────────────────────

    def _sample(self, goal: np.ndarray) -> np.ndarray:
        """Sample a random joint config (with goal bias)."""
        if self._rng.random() < self.goal_bias:
            return goal.copy()
        return self._rng.uniform(self._lo, self._hi)

    # ── Nearest node ─────────────────────────────────────────────────────────

    @staticmethod
    def _nearest(nodes: List[np.ndarray], q: np.ndarray) -> int:
        """Return index of nearest node to q in joint space (L2)."""
        dists = [float(np.linalg.norm(n - q)) for n in nodes]
        return int(np.argmin(dists))

    # ── Steer ────────────────────────────────────────────────────────────────

    def _steer(self, q_near: np.ndarray, q_rand: np.ndarray) -> np.ndarray:
        """Move from q_near toward q_rand by at most step_size."""
        diff = q_rand - q_near
        dist = float(np.linalg.norm(diff))
        if dist <= self.step_size:
            return q_rand.copy()
        return q_near + (diff / dist) * self.step_size

    # ── Main plan call ────────────────────────────────────────────────────────

    def plan(
        self,
        q_start: np.ndarray,
        q_goal:  np.ndarray,
    ) -> Tuple[bool, List[np.ndarray], int, int, float, str]:
        """
        Plan a collision-free joint-space path from q_start to q_goal.

        Returns
        -------
        success       : bool
        path          : List of joint-angle arrays (start → goal)
        n_nodes       : Nodes in tree
        n_iter        : Iterations consumed
        path_length   : Sum of L2 norms along path (radians)
        collision_status : "clear" or failure description
        """
        t0 = time.perf_counter()

        # Validate endpoints
        if not self._is_config_free(q_start):
            return False, [], 0, 0, 0.0, "Start configuration in collision"
        if not self._is_config_free(q_goal):
            # Relax: if goal is just at joint limit boundary, snap it
            q_goal = np.clip(q_goal, self._lo + 1e-4, self._hi - 1e-4)
            if not self._is_config_free(q_goal):
                return False, [], 0, 0, 0.0, "Goal configuration in collision"

        # If already at goal
        if float(np.linalg.norm(q_goal - q_start)) <= self.goal_thr:
            return True, [q_start.copy(), q_goal.copy()], 2, 1, 0.0, "clear"

        nodes   = [q_start.copy()]
        parents = [-1]           # parent index
        found   = -1

        for iteration in range(1, self.max_iter + 1):
            q_rand  = self._sample(q_goal)
            i_near  = self._nearest(nodes, q_rand)
            q_new   = self._steer(nodes[i_near], q_rand)

            if not self._is_segment_free(nodes[i_near], q_new):
                continue

            nodes.append(q_new)
            parents.append(i_near)

            # Goal check
            if float(np.linalg.norm(q_new - q_goal)) <= self.goal_thr:
                # Try direct connection to exact goal
                if self._is_segment_free(q_new, q_goal):
                    nodes.append(q_goal.copy())
                    parents.append(len(nodes) - 2)
                found = len(nodes) - 1
                log.info(f"Joint-RRT: goal reached after {iteration} iters, "
                         f"{len(nodes)} nodes, {(time.perf_counter()-t0)*1000:.1f} ms")
                break

        if found < 0:
            log.warning(f"Joint-RRT: failed after {self.max_iter} iters")
            return False, [], len(nodes), self.max_iter, 0.0, \
                   f"Path not found within {self.max_iter} iterations"

        # Reconstruct path
        path = []
        idx  = found
        while idx >= 0:
            path.append(nodes[idx])
            idx = parents[idx]
        path.reverse()

        # Greedy shortcutting
        path = self._smooth(path)

        # Path length
        length = sum(
            float(np.linalg.norm(path[k+1] - path[k]))
            for k in range(len(path) - 1)
        )

        # Verify collision status of final path
        col_status = "clear"
        for k in range(len(path) - 1):
            if not self._is_segment_free(path[k], path[k+1], n_check=8):
                col_status = f"Segment {k} in collision after smoothing"
                break

        return True, path, len(nodes), iteration, length, col_status

    def _smooth(self, path: List[np.ndarray], iters: int = 40) -> List[np.ndarray]:
        """Greedy shortcut smoothing in joint space."""
        p = list(path)
        for _ in range(iters):
            if len(p) <= 2:
                break
            i = int(self._rng.integers(0, len(p) - 2))
            j = int(self._rng.integers(i + 1, len(p)))
            if j == i + 1:
                continue
            if self._is_segment_free(p[i], p[j]):
                p = p[:i+1] + p[j:]
        return p


# ═══════════════════════════════════════════════════════════════════════════════
#  RobotPipeline  — IK → RRT → execution
# ═══════════════════════════════════════════════════════════════════════════════

class RobotPipeline:
    """
    Orchestrates the full Phase 13 pipeline for each robot motion:

        Target Cartesian Position
            ↓  Jacobian IK (Phase 11)
        Target Joint Configuration
            ↓  Joint-Space RRT (Phase 10 collision logic)
        Collision-Free Joint-Space Path
            ↓  Execute on virtual robot (Phase 12)
        Final Pose Verification

    Parameters
    ----------
    arm          : RobotArm (Phase 12 robot model, mutable state).
    scene        : SceneConfig (for obstacle geometry).
    ik_config    : IKConfig overrides (optional).
    rrt_max_iter : Max RRT iterations per motion.
    rrt_step     : Angular step size per RRT expansion (rad).
    exec_steps_per_wp : Simulation micro-steps executed per waypoint.
    """

    STATUS = "PHASE_13"

    def __init__(
        self,
        arm:              RobotArm,
        scene:            SceneConfig,
        ik_config:        Optional[IKConfig] = None,
        rrt_max_iter:     int   = 3000,
        rrt_step:         float = 0.15,
        exec_steps_per_wp: int  = 8,
    ) -> None:
        self.arm   = arm
        self.scene = scene

        # ── Phase 11: IK solver ───────────────────────────────────────────
        arm_cfg = ArmConfig.desk_arm_6dof()
        self._ik = JacobianIK(
            arm    = arm_cfg,
            config = ik_config or IKConfig(
                max_iterations=400,
                tolerance_m=0.004,
                damping=0.015,
                step_size=0.9,
            ),
        )
        self._arm_cfg = arm_cfg

        # ── Phase 10: Build 2-D Cartesian C-space from scene obstacles ───
        self._cspace_2d = self._build_cspace_2d()

        # ── Joint-space RRT ───────────────────────────────────────────────
        self._rrt = JointSpaceRRT(
            arm_config = arm_cfg,
            cspace_2d  = self._cspace_2d,
            max_iter   = rrt_max_iter,
            step_size  = rrt_step,
        )

        self.exec_steps_per_wp = exec_steps_per_wp

        # Pipeline history (accumulated across all motions)
        self.pipeline_history: List[PipelineResult] = []
        self.log_messages:     List[str] = []

    # ── Build Phase-10 C-space from scene ────────────────────────────────────

    def _build_cspace_2d(self) -> ConfigSpace:
        """
        Convert scene obstacles and table to a Phase-10 ConfigSpace.

        The workspace bounds cover the table surface in (x, y).
        Each cylindrical obstacle becomes a square bounding obstacle.
        """
        t = self.scene.table
        bounds = Bounds2D(
            x_min = min(float(t.position[0] - t.width  / 2), -0.80),
            x_max = max(float(t.position[0] + t.width  / 2),  0.80),
            y_min = min(float(t.position[1] - t.depth  / 2), -0.50),
            y_max = max(float(t.position[1] + t.depth  / 2),  0.90),
        )

        obs_list: List[Obstacle] = []
        for obs in self.scene.obstacles:
            r     = obs.radius
            ox, oy = obs.position[0], obs.position[1]
            obs_list.append(Obstacle(
                x      = ox - r,
                y      = oy - r,
                width  = 2 * r,
                height = 2 * r,
                margin = 0.04,
                label  = obs.obs_id,
            ))

        return ConfigSpace(bounds=bounds, obstacles=obs_list, seed=42)

    # ── Logging ───────────────────────────────────────────────────────────────

    def _log(self, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self.log_messages.append(f"[{ts}] {msg}")
        log.info(msg)

    # ── Main pipeline call ────────────────────────────────────────────────────

    def plan_and_execute(
        self,
        target_pos:      np.ndarray,
        gripper_action:  Optional[str] = None,
        active_book:     Optional[BookDef] = None,
    ) -> PipelineResult:
        """
        Run the full pipeline for one target Cartesian position.

        Steps
        -----
        1. Jacobian IK  → target joint configuration θ_goal.
        2. Joint-space RRT → collision-free path from θ_current to θ_goal.
        3. Execute path on virtual robot (update arm.angles step-by-step).
        4. Verify final pose.

        Parameters
        ----------
        target_pos     : (3,) target end-effector position [m].
        gripper_action : "open" | "close" | None — applied after execution.
        active_book    : Book being tracked (position updated during carry).

        Returns
        -------
        PipelineResult
        """
        result = PipelineResult(target_pos=target_pos.copy())
        self._log(f"Pipeline: target=({target_pos[0]:.3f},{target_pos[1]:.3f},{target_pos[2]:.3f})")

        # ── Stage 1: Jacobian IK ─────────────────────────────────────────
        t0 = time.perf_counter()
        ik_result = self._ik.solve(
            target_position = target_pos,
            initial_angles  = self.arm.angles,
        )
        ik_ms = (time.perf_counter() - t0) * 1000

        result.ik_result = ik_result
        result.stages.append(PipelineStep(
            stage       = "Jacobian IK",
            success     = ik_result.converged,
            duration_ms = ik_ms,
            details     = ik_result.to_dict(),
            error       = "" if ik_result.converged else
                          f"IK did not converge (error={ik_result.position_error_m*1000:.2f} mm)",
        ))

        if not ik_result.converged:
            self._log(f"IK failed — error {ik_result.position_error_m*1000:.2f} mm")
            self.pipeline_history.append(result)
            return result

        self._log(
            f"IK OK in {ik_result.iterations} iters, "
            f"err={ik_result.position_error_m*1000:.2f} mm, {ik_ms:.1f} ms"
        )

        q_start = self.arm.angles                   # current joint config
        q_goal  = ik_result.joint_angles_rad        # IK target

        # ── Stage 2: Joint-Space RRT ─────────────────────────────────────
        t1 = time.perf_counter()
        rrt_ok, path, n_nodes, n_iter, rrt_len, col_status = \
            self._rrt.plan(q_start, q_goal)
        rrt_ms = (time.perf_counter() - t1) * 1000

        result.rrt_n_nodes      = n_nodes
        result.rrt_n_waypoints  = len(path)
        result.rrt_length       = rrt_len
        result.rrt_planning_ms  = rrt_ms
        result.rrt_path_joints  = path
        result.collision_free   = (col_status == "clear")
        result.collision_status = col_status

        # Build Cartesian projection of RRT nodes for display
        result.rrt_path_cartesian = [
            self._rrt._ee_xy(q) for q in path
        ]

        result.stages.append(PipelineStep(
            stage       = "Joint-Space RRT",
            success     = rrt_ok,
            duration_ms = rrt_ms,
            details     = {
                "n_nodes":       n_nodes,
                "n_waypoints":   len(path),
                "path_length_rad": round(rrt_len, 4),
                "n_iterations":  n_iter,
                "collision":     col_status,
            },
            error = "" if rrt_ok else col_status,
        ))

        if not rrt_ok:
            # Fallback: direct interpolation (guarantees motion but may be non-optimal)
            self._log(f"RRT failed — using direct interpolation fallback")
            n_interp = 20
            path = [
                q_start + (q_goal - q_start) * (k / n_interp)
                for k in range(n_interp + 1)
            ]
            result.rrt_path_joints    = path
            result.rrt_path_cartesian = [
                self._rrt._ee_xy(q) for q in path
            ]
            result.rrt_n_waypoints    = len(path)
            result.collision_status   = "RRT failed — direct interpolation used"
            result.collision_free     = False
        else:
            self._log(
                f"RRT OK — {len(path)} waypoints, {n_nodes} nodes, "
                f"{rrt_len:.3f} rad, {rrt_ms:.1f} ms"
            )

        # ── Stage 3: Execute path on virtual robot ────────────────────────
        t2    = time.perf_counter()
        steps = self._execute_path(path, active_book)
        exec_ms = (time.perf_counter() - t2) * 1000

        result.executed_steps = steps

        # ── Stage 4: Verify final pose ────────────────────────────────────
        final_ee   = self.arm.end_effector_position()
        final_err  = float(np.linalg.norm(target_pos - final_ee))
        result.final_ee_pos   = final_ee.copy()
        result.final_error_m  = final_err
        result.success        = (
            ik_result.converged
            and result.rrt_n_waypoints > 0
            and final_err < 0.06      # 6 cm tolerance (generous for sim)
        )

        result.stages.append(PipelineStep(
            stage       = "Execute + Verify",
            success     = result.success,
            duration_ms = exec_ms,
            details     = {
                "steps_executed":   steps,
                "final_ee":         final_ee.tolist(),
                "final_error_mm":   round(final_err * 1000, 2),
            },
            error = "" if result.success else
                    f"Final error {final_err*1000:.1f} mm > 60 mm",
        ))

        self._log(
            f"Exec: {steps} steps, final_err={final_err*1000:.2f} mm, "
            f"{'OK' if result.success else 'WARN'}"
        )
        self.pipeline_history.append(result)
        return result

    # ── Execute path ─────────────────────────────────────────────────────────

    def _execute_path(
        self,
        path:         List[np.ndarray],
        active_book:  Optional[BookDef],
    ) -> int:
        """
        Interpolate each waypoint pair and set arm angles step-by-step.

        Returns total micro-steps executed.
        """
        total_steps = 0
        n = len(path)
        if n == 0:
            return 0

        for k in range(n - 1):
            q_from = path[k]
            q_to   = path[k + 1]
            for s in range(1, self.exec_steps_per_wp + 1):
                alpha = s / self.exec_steps_per_wp
                q_int = q_from + alpha * (q_to - q_from)
                self.arm.set_angles(q_int)
                total_steps += 1

                # If book is held, move it with the EE
                if active_book is not None and active_book.is_held:
                    ee = self.arm.end_effector_position()
                    new_pos = ee.copy()
                    new_pos[2] -= active_book.size[2] / 2 + 0.03
                    active_book.position = new_pos.tolist()

        # Set exactly the goal
        self.arm.set_angles(path[-1])
        return total_steps

    # ── Snapshot for UI ──────────────────────────────────────────────────────

    def latest_result_dict(self) -> Optional[dict]:
        """Return serialisable dict of the most recent PipelineResult."""
        if not self.pipeline_history:
            return None
        r = self.pipeline_history[-1]
        return {
            "target_pos":    r.target_pos.tolist(),
            "ik": {
                "converged":     r.ik_result.converged if r.ik_result else False,
                "iterations":    r.ik_result.iterations if r.ik_result else 0,
                "error_mm":      round(r.ik_result.position_error_m * 1000, 2) if r.ik_result else -1,
                "joint_angles_deg": (
                    [round(d, 1) for d in r.ik_result.joint_angles_deg.tolist()]
                    if r.ik_result else []
                ),
                "final_pos":     r.ik_result.final_position.tolist() if r.ik_result else [],
            },
            "rrt": {
                "success":       r.rrt_n_waypoints > 0,
                "n_nodes":       r.rrt_n_nodes,
                "n_waypoints":   r.rrt_n_waypoints,
                "path_length_rad": round(r.rrt_length, 4),
                "planning_ms":   round(r.rrt_planning_ms, 1),
                "collision_free": r.collision_free,
                "collision_status": r.collision_status,
                "path_cartesian": [list(p) for p in r.rrt_path_cartesian],
            },
            "execution": {
                "steps":       r.executed_steps,
                "final_ee":    r.final_ee_pos.tolist() if r.final_ee_pos is not None else [],
                "final_err_mm": round(r.final_error_m * 1000, 2),
            },
            "stages": [
                {
                    "stage":       s.stage,
                    "success":     s.success,
                    "duration_ms": round(s.duration_ms, 1),
                    "details":     s.details,
                    "error":       s.error,
                }
                for s in r.stages
            ],
            "success": r.success,
        }

    def all_history_dict(self) -> List[dict]:
        """Serialisable list of all past pipeline results."""
        out = []
        for r in self.pipeline_history:
            out.append({
                "target":    r.target_pos.tolist(),
                "success":   r.success,
                "ik_ok":     r.ik_result.converged if r.ik_result else False,
                "rrt_ok":    r.rrt_n_waypoints > 0,
                "collision_free": r.collision_free,
                "n_waypoints":   r.rrt_n_waypoints,
                "final_err_mm":  round(r.final_error_m * 1000, 2),
                "rrt_ms":        round(r.rrt_planning_ms, 1),
            })
        return out
