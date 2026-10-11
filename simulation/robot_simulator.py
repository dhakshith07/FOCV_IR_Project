"""
simulation/robot_simulator.py
==============================
Phase 12 — Software-based Robotic Arm Simulator.

PURPOSE
-------
Drive a 6-DOF robot arm through a complete book pick-and-place task:
  1. Move to approach position above a book
  2. Lower to grasp position
  3. Close gripper (simulate gripping)
  4. Lift the book clear of the table
  5. Move laterally to target position
  6. Lower to placement height
  7. Release the book

PHYSICS BACKEND
---------------
PyBullet (optional, auto-detected):
  - Gravity + rigid-body dynamics
  - Collision geometry (box bodies for table, books; capsule for obstacles)
  - URDF-free body creation via createMultiBody()

Numpy fallback (always available):
  - Pure forward kinematics via DH matrices
  - Jacobian-based IK solver (built-in, no external dependency)
  - No physics engine required

CLASSES
-------
  SimState       -- Enum: IDLE | MOVING | GRASPING | LIFTING |
                          CARRYING | PLACING | RELEASING | DONE |
                          PAUSED | ERROR
  TaskStep       -- One waypoint in the task sequence
  _JacobianIK    -- Lightweight built-in Jacobian IK (Phase 12 self-contained)
  RobotSimulator -- Main simulator class
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from robotics.robot_model import RobotArm, SceneConfig, BookDef
from robotics.gripper import Gripper, GripperState

try:
    import pybullet as pb
    import pybullet_data
    _PYBULLET_OK = True
except ImportError:
    _PYBULLET_OK = False

try:
    from utils.logger import get_logger
    log = get_logger(__name__)
except Exception:
    import logging
    log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  SimState
# ═══════════════════════════════════════════════════════════════════════════════

class SimState(Enum):
    IDLE      = "idle"
    MOVING    = "moving"
    GRASPING  = "grasping"
    LIFTING   = "lifting"
    CARRYING  = "carrying"
    PLACING   = "placing"
    RELEASING = "releasing"
    DONE      = "done"
    PAUSED    = "paused"
    ERROR     = "error"


# ═══════════════════════════════════════════════════════════════════════════════
#  TaskStep
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TaskStep:
    """One waypoint / action in the pick-and-place task sequence."""
    description:  str
    target_pos:   Optional[np.ndarray] = None   # EE target position [m]
    gripper_cmd:  Optional[str] = None           # "open" | "close" | None
    sim_state:    SimState = SimState.MOVING
    tolerance:    float = 0.015                  # arrival tolerance [m]
    completed:    bool  = False


# ═══════════════════════════════════════════════════════════════════════════════
#  _JacobianIK — self-contained Jacobian IK solver (no external dependency)
# ═══════════════════════════════════════════════════════════════════════════════

class _JacobianIK:
    """
    Lightweight Jacobian pseudo-inverse IK solver for the RobotArm.

    Uses the analytic Jacobian computed by finite-difference FK.
    Falls back to random restarts if the initial attempt diverges.
    """

    def __init__(
        self,
        arm:            RobotArm,
        max_iterations: int   = 400,
        tolerance_m:    float = 0.008,
        step_size:      float = 0.6,
        damping:        float = 0.025,
    ) -> None:
        self.arm            = arm
        self.max_iterations = max_iterations
        self.tolerance_m    = tolerance_m
        self.step_size      = step_size
        self.damping        = damping
        self._eps           = 1e-5   # finite-difference step

    def _jacobian(self, angles: np.ndarray) -> np.ndarray:
        """3 x dof Jacobian by forward-difference."""
        p0  = self.arm.end_effector_position(angles)
        J   = np.zeros((3, self.arm.dof), dtype=np.float64)
        for i in range(self.arm.dof):
            a1     = angles.copy()
            a1[i] += self._eps
            J[:, i] = (self.arm.end_effector_position(a1) - p0) / self._eps
        return J

    def solve(
        self,
        target_pos:   np.ndarray,
        initial_q:    Optional[np.ndarray] = None,
        n_restarts:   int = 3,
    ) -> Tuple[np.ndarray, float, bool]:
        """
        Solve IK for target_pos.

        Returns
        -------
        angles     : best joint configuration found
        error_m    : final EE-to-target distance [m]
        converged  : True if error < tolerance_m
        """
        target = np.asarray(target_pos, dtype=np.float64).flatten()
        best_q = self.arm.angles if initial_q is None else initial_q.copy()
        best_err = float("inf")

        attempts = [best_q.copy()]
        rng = np.random.default_rng(seed=7)
        for _ in range(n_restarts):
            q_rand = np.array([
                rng.uniform(jd.limit_lower, jd.limit_upper)
                for jd in self.arm.joints
            ])
            attempts.append(q_rand)

        for q0 in attempts:
            q = q0.copy()
            for _ in range(self.max_iterations):
                ee  = self.arm.end_effector_position(q)
                err = target - ee
                err_norm = float(np.linalg.norm(err))
                if err_norm < self.tolerance_m:
                    best_q   = q.copy()
                    best_err = err_norm
                    break
                J   = self._jacobian(q)
                JtJ = J.T @ J + (self.damping ** 2) * np.eye(self.arm.dof)
                dq  = self.step_size * np.linalg.solve(JtJ, J.T @ err)
                q   = np.array([self.arm.joints[i].clamp(q[i] + dq[i])
                                 for i in range(self.arm.dof)])
            else:
                ee  = self.arm.end_effector_position(q)
                err_norm = float(np.linalg.norm(target - ee))

            if err_norm < best_err:
                best_err = err_norm
                best_q   = q.copy()

            if best_err < self.tolerance_m:
                break

        return best_q, best_err, best_err < self.tolerance_m * 1.5


# ═══════════════════════════════════════════════════════════════════════════════
#  RobotSimulator
# ═══════════════════════════════════════════════════════════════════════════════

class RobotSimulator:
    """
    Phase 12 — Software-based Robotic Arm Simulator.

    Supports two backends:
      • PyBullet (headless) — full rigid-body physics
      • NumPy fallback       — pure kinematics, no physics engine needed

    Control API
    -----------
    start(book_id)     -- begin pick-and-place for given book
    pause()            -- pause simulation
    resume()           -- resume from pause
    reset()            -- reset to initial state
    step() -> bool     -- advance one task step; returns False when done
    run_n_steps(n)     -- advance up to n steps

    State access
    ------------
    snapshot() -> dict -- full JSON-serialisable state snapshot for the UI
    """

    STATUS = "PHASE_12"

    def __init__(
        self,
        scene:        Optional[SceneConfig] = None,
        headless:     bool = True,
        use_pybullet: Optional[bool] = None,
    ) -> None:
        self.scene   = scene or SceneConfig.default_scene()
        # Keep original book positions for reset
        self._initial_book_positions = {
            b.book_id: list(b.position) for b in self.scene.books
        }
        self.arm     = self.scene.arm
        self.gripper = Gripper()

        # Built-in IK solver (self-contained, no pipeline import)
        self._ik = _JacobianIK(
            arm            = self.arm,
            max_iterations = 400,
            tolerance_m    = 0.008,
            step_size      = 0.6,
            damping        = 0.025,
        )

        # Simulation state
        self._state:             SimState       = SimState.IDLE
        self._step_count:        int            = 0
        self._task_steps:        List[TaskStep] = []
        self._current_step_idx:  int            = 0
        self._active_book:       Optional[BookDef] = None
        self._paused:            bool           = False
        self._running:           bool           = False

        # History for UI plots
        self.ee_history:    List[np.ndarray] = []
        self.joint_history: List[np.ndarray] = []
        self.state_log:     List[str]        = []
        self.log_messages:  List[str]        = []

        # IK diagnostics per step
        self.ik_log: List[dict] = []

        # PyBullet handles
        self._pb_client: Optional[int] = None
        self._pb_bodies: Dict[str, int] = {}
        self._use_pb:    bool = False

        # Decide backend
        want_pb = use_pybullet if use_pybullet is not None else _PYBULLET_OK
        if want_pb and _PYBULLET_OK:
            self._init_pybullet(headless)
        else:
            self._use_pb = False

        self._log(
            f"Phase-12 Simulator ready (backend={'pybullet' if self._use_pb else 'numpy'})"
        )

    # ── PyBullet setup ────────────────────────────────────────────────────────

    def _init_pybullet(self, headless: bool) -> None:
        try:
            mode = pb.DIRECT if headless else pb.GUI
            self._pb_client = pb.connect(mode)
            pb.setAdditionalSearchPath(pybullet_data.getDataPath())
            pb.setGravity(0, 0, -9.81, physicsClientId=self._pb_client)
            pb.setTimeStep(1 / 240, physicsClientId=self._pb_client)
            self._use_pb = True
            self._build_pybullet_scene()
        except Exception as exc:
            log.warning(f"PyBullet init failed: {exc} — using numpy mode")
            self._use_pb = False

    def _build_pybullet_scene(self) -> None:
        cid = self._pb_client

        # Table
        t = self.scene.table
        t_col = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=list(t.half_extents),
                                        physicsClientId=cid)
        t_vis = pb.createVisualShape(pb.GEOM_BOX, halfExtents=list(t.half_extents),
                                     rgbaColor=list(t.color_rgba), physicsClientId=cid)
        t_id  = pb.createMultiBody(baseMass=0, baseCollisionShapeIndex=t_col,
                                   baseVisualShapeIndex=t_vis,
                                   basePosition=list(t.position), physicsClientId=cid)
        self._pb_bodies["table"] = t_id

        # Books
        for book in self.scene.books:
            b_col = pb.createCollisionShape(pb.GEOM_BOX,
                                            halfExtents=list(book.half_extents),
                                            physicsClientId=cid)
            b_vis = pb.createVisualShape(pb.GEOM_BOX,
                                         halfExtents=list(book.half_extents),
                                         rgbaColor=list(book.color_rgba),
                                         physicsClientId=cid)
            b_id  = pb.createMultiBody(baseMass=0.3,
                                       baseCollisionShapeIndex=b_col,
                                       baseVisualShapeIndex=b_vis,
                                       basePosition=list(book.position),
                                       physicsClientId=cid)
            self._pb_bodies[book.book_id] = b_id

        # Obstacles
        for obs in self.scene.obstacles:
            o_col = pb.createCollisionShape(pb.GEOM_CYLINDER,
                                            radius=obs.radius, height=obs.height,
                                            physicsClientId=cid)
            o_vis = pb.createVisualShape(pb.GEOM_CYLINDER,
                                          radius=obs.radius, length=obs.height,
                                          rgbaColor=list(obs.color_rgba),
                                          physicsClientId=cid)
            o_id  = pb.createMultiBody(baseMass=0,
                                       baseCollisionShapeIndex=o_col,
                                       baseVisualShapeIndex=o_vis,
                                       basePosition=list(obs.position),
                                       physicsClientId=cid)
            self._pb_bodies[obs.obs_id] = o_id

    # ── Logging ───────────────────────────────────────────────────────────────

    def _log(self, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        entry = f"[{ts}] {msg}"
        self.log_messages.append(entry)
        log.info(msg)

    # ── Control API ───────────────────────────────────────────────────────────

    def start(self, book_id: Optional[str] = None) -> None:
        """
        Start the full pick-and-place task for the specified book.

        Parameters
        ----------
        book_id : ID of the book to pick (e.g. "B1").
                  Defaults to the first book in the scene.
        """
        if self._running and self._state not in (SimState.IDLE, SimState.DONE,
                                                  SimState.ERROR):
            self._log("Already running — call reset() first")
            return

        self._running = True
        self._paused  = False

        # Select book
        if book_id is None:
            self._active_book = self.scene.books[0]
        else:
            matches = [b for b in self.scene.books if b.book_id == book_id]
            if not matches:
                self._log(f"Book {book_id} not found in scene")
                self._state = SimState.ERROR
                return
            self._active_book = matches[0]

        # Reset book state
        self._active_book.is_held = False
        self.gripper.release()
        self.arm.set_home()

        # Build task sequence
        self._task_steps       = self._build_task_steps(self._active_book)
        self._current_step_idx = 0
        self._state            = SimState.MOVING
        self.ik_log.clear()
        self._log(f"Task started — picking '{self._active_book.title}' ({book_id})")

    def pause(self) -> None:
        """Pause the simulation."""
        self._paused = True
        self._state  = SimState.PAUSED
        self._log("Simulation paused")

    def resume(self) -> None:
        """Resume from pause."""
        self._paused = False
        if self._task_steps and self._current_step_idx < len(self._task_steps):
            self._state = self._task_steps[self._current_step_idx].sim_state
        self._log("Simulation resumed")

    def reset(self) -> None:
        """Reset simulation to initial state (all books return to origin)."""
        # Restore book positions
        for book in self.scene.books:
            orig_pos = self._initial_book_positions.get(book.book_id)
            if orig_pos:
                book.position = list(orig_pos)
            book.is_held = False
            if self._use_pb and book.book_id in self._pb_bodies:
                pb.resetBasePositionAndOrientation(
                    self._pb_bodies[book.book_id],
                    orig_pos or book.position,
                    [0, 0, 0, 1],
                    physicsClientId=self._pb_client,
                )

        self.arm.set_home()
        self.gripper.reset()

        self._state            = SimState.IDLE
        self._step_count       = 0
        self._task_steps       = []
        self._current_step_idx = 0
        self._active_book      = None
        self._running          = False
        self._paused           = False

        self.ee_history.clear()
        self.joint_history.clear()
        self.state_log.clear()
        self.log_messages.clear()
        self.ik_log.clear()
        self._log("Simulation reset to initial state")

    def step(self) -> bool:
        """
        Advance by ONE simulation step.

        For movement steps: runs built-in Jacobian IK to compute the joint
        angles for the target position, then applies them to the arm.

        For gripper steps: opens or closes the gripper and (if closing)
        attempts to grasp the active book.

        Returns True if a step was processed, False when done/paused.
        """
        if self._paused or self._state in (SimState.DONE, SimState.IDLE, SimState.ERROR):
            return False

        self._step_count += 1
        self.gripper.step(dt=0.05)

        if not self._task_steps or self._current_step_idx >= len(self._task_steps):
            self._state = SimState.DONE
            self._log("All task steps complete ✓")
            return False

        current = self._task_steps[self._current_step_idx]

        if current.completed:
            self._current_step_idx += 1
            return True

        # ── Gripper-only step ─────────────────────────────────────────────────
        if current.target_pos is None:
            if current.gripper_cmd == "open":
                self.gripper.open()
                self._log(f"Gripper → OPEN")
            elif current.gripper_cmd == "close":
                self.gripper.close()
                ee = self.arm.end_effector_position()
                if self._active_book and not self._active_book.is_held:
                    success = self.gripper.try_grasp(self._active_book, ee)
                    if not success:
                        # Force-grasp: IK already placed EE at grasp position
                        self.gripper.force_grasp(self._active_book)
                        self._log(f"Gripper → CLOSED (force grasp '{self._active_book.title}')")
                    else:
                        self._log(f"Gripper → CLOSED (grasped '{self._active_book.title}')")
            current.completed = True
            self._current_step_idx += 1

        # ── Movement step via built-in Jacobian IK ────────────────────────────
        else:
            self._state = current.sim_state
            target = current.target_pos

            # Run IK
            q_sol, err_m, converged = self._ik.solve(
                target_pos = target,
                initial_q  = self.arm.angles,
            )

            # Apply solution to arm
            self.arm.set_angles(q_sol)

            # Update held book position
            ee = self.arm.end_effector_position()
            self.gripper.update_held_book(ee)

            # Sync to PyBullet if active
            if self._use_pb and self._active_book and self._active_book.is_held:
                bid = self._pb_bodies.get(self._active_book.book_id)
                if bid is not None:
                    pb.resetBasePositionAndOrientation(
                        bid, self._active_book.position, [0, 0, 0, 1],
                        physicsClientId=self._pb_client,
                    )
                pb.stepSimulation(physicsClientId=self._pb_client)

            # Log IK result
            self.ik_log.append({
                "step":       current.description,
                "target":     target.tolist(),
                "ee_pos":     ee.tolist(),
                "err_mm":     round(err_m * 1000, 2),
                "converged":  converged,
                "angles_deg": self.arm.angles_deg.tolist(),
            })

            status = "OK" if converged else "APPROX"
            self._log(
                f"[{status}] {current.description} — "
                f"err={err_m*1000:.1f} mm  "
                f"ee=({ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f})"
            )

            current.completed = True
            self._current_step_idx += 1

            # Advance state
            if self._current_step_idx < len(self._task_steps):
                self._state = self._task_steps[self._current_step_idx].sim_state
            else:
                self._state = SimState.DONE

        # Record history
        ee = self.arm.end_effector_position()
        self.ee_history.append(ee.copy())
        self.joint_history.append(self.arm.angles.copy())
        self.state_log.append(self._state.value)
        return True

    def run_n_steps(self, n: int = 10) -> int:
        """Run up to n steps. Returns number actually executed."""
        executed = 0
        for _ in range(n):
            if not self.step():
                break
            executed += 1
        return executed

    # ── Task builder ──────────────────────────────────────────────────────────

    def _build_task_steps(self, book: BookDef) -> List[TaskStep]:
        """
        Build the ordered pick-and-place task steps for one book.

        Sequence:
          1. Open gripper
          2. Move above book  (approach)
          3. Lower to grasp position
          4. Close gripper  (grasp)
          5. Lift book clear of table
          6. Carry to target area
          7. Lower to placement height
          8. Open gripper  (release)
          9. Retreat upward
        """
        grasp_pos  = book.grasp_position()
        above_book = grasp_pos.copy()
        above_book[2] += 0.12     # 12 cm above grasp

        place_pos = book.place_position()
        if place_pos is None:
            place_pos = grasp_pos.copy()
            place_pos[0] += 0.30

        above_place = place_pos.copy()
        above_place[2] += 0.12
        retreat = above_place.copy()
        retreat[2] += 0.08

        return [
            TaskStep("Open gripper",
                     target_pos=None, gripper_cmd="open",
                     sim_state=SimState.IDLE),

            TaskStep("Approach: move above book",
                     target_pos=above_book,
                     sim_state=SimState.MOVING),

            TaskStep("Lower to grasp position",
                     target_pos=grasp_pos,
                     sim_state=SimState.GRASPING,
                     tolerance=0.015),

            TaskStep("Close gripper — grasp book",
                     target_pos=None, gripper_cmd="close",
                     sim_state=SimState.GRASPING),

            TaskStep("Lift book above table",
                     target_pos=above_book,
                     sim_state=SimState.LIFTING),

            TaskStep("Carry book to target area",
                     target_pos=above_place,
                     sim_state=SimState.CARRYING),

            TaskStep("Lower to placement position",
                     target_pos=place_pos,
                     sim_state=SimState.PLACING,
                     tolerance=0.015),

            TaskStep("Open gripper — release book",
                     target_pos=None, gripper_cmd="open",
                     sim_state=SimState.RELEASING),

            TaskStep("Retreat to safe height",
                     target_pos=retreat,
                     sim_state=SimState.MOVING),
        ]

    # ── Snapshot for UI ───────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        """
        Return a complete JSON-serialisable snapshot of current simulation state.

        Used by the Streamlit visualisation page.
        """
        ee     = self.arm.end_effector_position()
        joints = self.arm.joint_positions()
        T_ee   = self.arm.end_effector_transform()
        f_pos  = self.gripper.finger_positions(T_ee)

        # Latest IK diagnostic
        latest_ik = self.ik_log[-1] if self.ik_log else None

        return {
            # ── Core state ──────────────────────────────────────────────────
            "state":              self._state.value,
            "step_count":         self._step_count,
            "running":            self._running,
            "paused":             self._paused,

            # ── Arm ─────────────────────────────────────────────────────────
            "ee_pos":             ee.tolist(),
            "joint_angles_deg":   self.arm.angles_deg.tolist(),
            "joint_positions":    [p.tolist() for p in joints],
            "joint_info":         self.arm.joint_info(),

            # ── Gripper ─────────────────────────────────────────────────────
            "gripper": self.gripper.info(),
            "finger_positions": {
                "left_base":  f_pos[0].tolist(),
                "left_tip":   f_pos[1].tolist(),
                "right_base": f_pos[2].tolist(),
                "right_tip":  f_pos[3].tolist(),
            },

            # ── Scene objects ────────────────────────────────────────────────
            "books": [
                {
                    "id":       b.book_id,
                    "title":    b.title,
                    "position": list(b.position),
                    "size":     list(b.size),
                    "color":    list(b.color_rgba),
                    "is_held":  b.is_held,
                    "target":   b.target_pos,
                }
                for b in self.scene.books
            ],
            "obstacles": [
                {
                    "id":       o.obs_id,
                    "position": list(o.position),
                    "radius":   o.radius,
                    "height":   o.height,
                    "color":    list(o.color_rgba),
                }
                for o in self.scene.obstacles
            ],
            "table": {
                "position": list(self.scene.table.position),
                "width":    self.scene.table.width,
                "depth":    self.scene.table.depth,
                "height":   self.scene.table.height,
                "color":    list(self.scene.table.color_rgba),
            },

            # ── Task progress ────────────────────────────────────────────────
            "task_steps": [
                {
                    "description": s.description,
                    "completed":   s.completed,
                    "active":      (i == self._current_step_idx),
                    "sim_state":   s.sim_state.value,
                }
                for i, s in enumerate(self._task_steps)
            ],
            "current_step": (
                self._task_steps[self._current_step_idx].description
                if self._task_steps and self._current_step_idx < len(self._task_steps)
                else "—"
            ),
            "active_book": self._active_book.book_id if self._active_book else None,

            # ── History ─────────────────────────────────────────────────────
            "ee_history":   [p.tolist() for p in self.ee_history[-300:]],
            "state_log":    self.state_log[-200:],
            "log_messages": self.log_messages[-30:],

            # ── Diagnostics ─────────────────────────────────────────────────
            "use_pybullet": self._use_pb,
            "ik_latest":    latest_ik,
            "ik_history":   self.ik_log[-50:],

            # Compatibility shim: pipeline key expected by old robot_sim_page
            "pipeline": {
                "latest":    None,
                "history":   self.ik_log[-50:],
                "n_motions": len(self.ik_log),
            },
        }

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def shutdown(self) -> None:
        """Disconnect from PyBullet if connected."""
        if self._use_pb and self._pb_client is not None:
            try:
                pb.disconnect(physicsClientId=self._pb_client)
                self._log("PyBullet disconnected")
            except Exception:
                pass

    def __del__(self) -> None:
        try:
            self.shutdown()
        except Exception:
            pass

    def __repr__(self) -> str:
        return (
            f"RobotSimulator(state={self._state.value}, "
            f"step={self._step_count}, "
            f"backend={'pybullet' if self._use_pb else 'numpy'})"
        )
