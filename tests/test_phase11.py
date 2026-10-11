"""
tests/test_phase11.py
=====================
Phase 11 — Jacobian-based Inverse Kinematics (6-DOF Desk Arm).

Test groups
-----------
  A. DHParam                        (fields, defaults)
  B. JointLimit                     (clamp, is_valid, bounds)
  C. ArmConfig.desk_arm_6dof        (dof=6, DH lengths, limits)
  D. ArmConfig.from_config_dict     (limits from degrees, dof)
  E. IKConfig                       (defaults, from_dict, tolerance conversion)
  F. ForwardKinematics.dh_matrix    (identity at zero angles, row/col count,
                                     known analytic value)
  G. ForwardKinematics.compute      (shape, zero-config position,
                                     frame count, last frame = T_ee)
  H. FK consistency                 (end_effector_position matches compute[:3,3])
  I. FK -- known analytic value     (single-joint, single-link arm)
  J. JacobianCalculator             (shape (6,n), position part (3,n))
  K. Jacobian -- zero rows absent   (not all zeros for generic config)
  L. Jacobian -- finite values      (no NaN/Inf)
  M. Jacobian -- numerical check    (finite differences ≈ analytic)
  N. IKResult                       (display is string, to_dict keys)
  O. IKResult.display               (contains all required fields)
  P. JacobianIK construction        (arm and cfg set, fk and jac exist)
  Q. JacobianIK.solve -- zero target reachable (close to origin)
  R. JacobianIK.solve -- returns IKResult
  S. JacobianIK.solve -- position error ≤ tolerance on convergence
  T. JacobianIK.solve -- joint angles within limits on convergence
  U. JacobianIK.solve -- per_iter_error decreasing (generally)
  V. JacobianIK.solve -- iterations ≤ max_iterations
  W. JacobianIK.solve -- unreachable target (does not crash, result.converged=False)
  X. JacobianIK.solve -- custom initial angles respected
  Y. JacobianIK.solve -- orientation mode (does not crash, returns result)
  Z. JacobianIK.reachable           (returns bool)
  AA. JacobianIK.workspace_sample   (shape, within reach of arm)
  AB. JacobianIK.from_config_dict   (builds successfully from dict)
  AC. IKResult.to_dict              (numeric types, all keys present)
  AD. Public imports                (robotics.ik namespace, __all__)

Run:
    pytest tests/test_phase11.py -v
    pytest tests/test_phase11.py -v -k "not slow"
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import List

import numpy as np
import pytest

# ── project root on path ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ══════════════════════════════════════════════════════════════════════════════
#  Fixtures / helpers
# ══════════════════════════════════════════════════════════════════════════════

def _arm():
    from robotics.ik.jacobian_ik import ArmConfig
    return ArmConfig.desk_arm_6dof()


def _fk(arm=None):
    from robotics.ik.jacobian_ik import ForwardKinematics
    return ForwardKinematics(arm or _arm())


def _jac(arm=None):
    from robotics.ik.jacobian_ik import JacobianCalculator
    arm = arm or _arm()
    return JacobianCalculator(arm, _fk(arm))


def _ik(max_iter=500, tol=1e-3, damping=0.05, step=1.0):
    from robotics.ik.jacobian_ik import JacobianIK, IKConfig, ArmConfig
    cfg = IKConfig(
        max_iterations=max_iter,
        tolerance_m=tol,
        damping=damping,
        step_size=step,
        solve_orientation=False,
    )
    return JacobianIK(arm=ArmConfig.desk_arm_6dof(), config=cfg)


def _zero_angles(n: int = 6) -> np.ndarray:
    return np.zeros(n, dtype=np.float64)


def _random_angles(arm=None, seed: int = 0) -> np.ndarray:
    """Random valid joint angles for the arm."""
    from robotics.ik.jacobian_ik import ArmConfig
    arm = arm or ArmConfig.desk_arm_6dof()
    rng = np.random.default_rng(seed)
    return np.array([
        rng.uniform(lim.lower * 0.8, lim.upper * 0.8)
        for lim in arm.joint_limits
    ], dtype=np.float64)


def _reachable_target(arm=None, seed: int = 42) -> np.ndarray:
    """Return a guaranteed-reachable FK position."""
    arm = arm or _arm()
    fk  = _fk(arm)
    angles = _random_angles(arm, seed)
    T, _ = fk.compute(angles)
    return T[:3, 3].copy()


# ══════════════════════════════════════════════════════════════════════════════
#  A. DHParam
# ══════════════════════════════════════════════════════════════════════════════

class TestDHParam:

    def test_fields(self):
        from robotics.ik.jacobian_ik import DHParam
        p = DHParam(a=0.25, alpha=1.57, d=0.15, theta0=0.0)
        assert p.a      == pytest.approx(0.25)
        assert p.alpha  == pytest.approx(1.57)
        assert p.d      == pytest.approx(0.15)
        assert p.theta0 == pytest.approx(0.0)

    def test_defaults(self):
        from robotics.ik.jacobian_ik import DHParam
        p = DHParam()
        assert p.a == 0.0
        assert p.d == 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  B. JointLimit
# ══════════════════════════════════════════════════════════════════════════════

class TestJointLimit:

    def test_clamp_in_range(self):
        from robotics.ik.jacobian_ik import JointLimit
        jl = JointLimit(lower=-1.0, upper=1.0)
        assert jl.clamp(0.5) == pytest.approx(0.5)

    def test_clamp_below_lower(self):
        from robotics.ik.jacobian_ik import JointLimit
        jl = JointLimit(lower=-1.0, upper=1.0)
        assert jl.clamp(-2.0) == pytest.approx(-1.0)

    def test_clamp_above_upper(self):
        from robotics.ik.jacobian_ik import JointLimit
        jl = JointLimit(lower=-1.0, upper=1.0)
        assert jl.clamp(3.0) == pytest.approx(1.0)

    def test_is_valid_inside(self):
        from robotics.ik.jacobian_ik import JointLimit
        jl = JointLimit(lower=-1.0, upper=1.0)
        assert jl.is_valid(0.0)

    def test_is_valid_outside(self):
        from robotics.ik.jacobian_ik import JointLimit
        jl = JointLimit(lower=-1.0, upper=1.0)
        assert not jl.is_valid(2.0)

    def test_boundary_valid(self):
        from robotics.ik.jacobian_ik import JointLimit
        jl = JointLimit(lower=-1.0, upper=1.0)
        assert jl.is_valid(-1.0)
        assert jl.is_valid(1.0)


# ══════════════════════════════════════════════════════════════════════════════
#  C. ArmConfig.desk_arm_6dof
# ══════════════════════════════════════════════════════════════════════════════

class TestArmConfigDesk6DOF:

    def test_dof_is_6(self):
        arm = _arm()
        assert arm.dof == 6

    def test_dh_params_count(self):
        arm = _arm()
        assert len(arm.dh_params) == 6

    def test_joint_limits_count(self):
        arm = _arm()
        assert len(arm.joint_limits) == 6

    def test_link_lengths_positive_for_arms(self):
        arm = _arm()
        # joints 2 and 3 should have positive a (upper arm, forearm)
        assert arm.dh_params[1].a > 0
        assert arm.dh_params[2].a > 0

    def test_limits_symmetric(self):
        arm = _arm()
        for lim in arm.joint_limits:
            # Each limit should be symmetric around 0
            assert lim.lower < 0
            assert lim.upper > 0

    def test_limits_in_radians(self):
        arm = _arm()
        for lim in arm.joint_limits:
            # Limits should be in radians (not degrees)
            assert abs(lim.lower) <= math.pi + 0.1
            assert abs(lim.upper) <= math.pi + 0.1


# ══════════════════════════════════════════════════════════════════════════════
#  D. ArmConfig.from_config_dict
# ══════════════════════════════════════════════════════════════════════════════

class TestArmConfigFromDict:

    def test_dof_respected(self):
        from robotics.ik.jacobian_ik import ArmConfig
        d = {"dof": 4, "joint_limits": [[-90, 90]] * 4}
        arm = ArmConfig.from_config_dict(d)
        assert arm.dof == 4

    def test_limits_converted_from_degrees(self):
        from robotics.ik.jacobian_ik import ArmConfig
        d = {"dof": 6, "joint_limits": [[-180, 180]] * 6}
        arm = ArmConfig.from_config_dict(d)
        assert arm.joint_limits[0].lower == pytest.approx(-math.pi, abs=1e-6)
        assert arm.joint_limits[0].upper == pytest.approx( math.pi, abs=1e-6)

    def test_partial_limits(self):
        from robotics.ik.jacobian_ik import ArmConfig
        d = {"dof": 3, "joint_limits": [[-90, 90], [-45, 45], [-60, 60]]}
        arm = ArmConfig.from_config_dict(d)
        assert arm.joint_limits[1].upper == pytest.approx(math.radians(45), abs=1e-5)


# ══════════════════════════════════════════════════════════════════════════════
#  E. IKConfig
# ══════════════════════════════════════════════════════════════════════════════

class TestIKConfig:

    def test_defaults(self):
        from robotics.ik.jacobian_ik import IKConfig
        cfg = IKConfig()
        assert cfg.max_iterations     == 200
        assert cfg.tolerance_m        == pytest.approx(0.001)
        assert cfg.damping            == pytest.approx(0.01)
        assert cfg.step_size          == pytest.approx(1.0)
        assert cfg.solve_orientation  is False

    def test_from_dict_tolerance_mm(self):
        from robotics.ik.jacobian_ik import IKConfig
        cfg = IKConfig.from_dict({"tolerance_mm": 2.0})
        assert cfg.tolerance_m == pytest.approx(0.002, abs=1e-7)

    def test_from_dict_damping_lambda(self):
        from robotics.ik.jacobian_ik import IKConfig
        cfg = IKConfig.from_dict({"damping_lambda": 0.05})
        assert cfg.damping == pytest.approx(0.05, abs=1e-7)

    def test_from_dict_max_iterations(self):
        from robotics.ik.jacobian_ik import IKConfig
        cfg = IKConfig.from_dict({"max_iterations": 50})
        assert cfg.max_iterations == 50


# ══════════════════════════════════════════════════════════════════════════════
#  F. ForwardKinematics.dh_matrix
# ══════════════════════════════════════════════════════════════════════════════

class TestDHMatrix:

    def test_shape(self):
        from robotics.ik.jacobian_ik import ForwardKinematics
        T = ForwardKinematics.dh_matrix(0.0, 0.0, 0.0, 0.0)
        assert T.shape == (4, 4)

    def test_all_zeros_is_identity(self):
        from robotics.ik.jacobian_ik import ForwardKinematics
        # a=0, alpha=0, d=0, theta=0 → identity
        T = ForwardKinematics.dh_matrix(0.0, 0.0, 0.0, 0.0)
        np.testing.assert_allclose(T, np.eye(4), atol=1e-12)

    def test_pure_rotation_about_z(self):
        from robotics.ik.jacobian_ik import ForwardKinematics
        # Only theta=π/2, rest zero → Rz(π/2)
        T = ForwardKinematics.dh_matrix(0.0, 0.0, 0.0, math.pi / 2)
        assert T[0, 0] == pytest.approx(0.0,  abs=1e-10)
        assert T[0, 1] == pytest.approx(-1.0, abs=1e-10)
        assert T[1, 0] == pytest.approx(1.0,  abs=1e-10)
        assert T[1, 1] == pytest.approx(0.0,  abs=1e-10)
        assert T[3, 3] == pytest.approx(1.0,  abs=1e-12)  # homogeneous

    def test_translation_along_z(self):
        from robotics.ik.jacobian_ik import ForwardKinematics
        # d=0.5, rest zero → Tz(0.5)
        T = ForwardKinematics.dh_matrix(0.0, 0.0, 0.5, 0.0)
        assert T[2, 3] == pytest.approx(0.5, abs=1e-12)

    def test_translation_along_x(self):
        from robotics.ik.jacobian_ik import ForwardKinematics
        # a=0.3, rest zero → Tx(0.3)
        T = ForwardKinematics.dh_matrix(0.3, 0.0, 0.0, 0.0)
        assert T[0, 3] == pytest.approx(0.3, abs=1e-12)

    def test_last_row(self):
        from robotics.ik.jacobian_ik import ForwardKinematics
        T = ForwardKinematics.dh_matrix(0.1, 0.5, 0.2, 0.3)
        np.testing.assert_allclose(T[3], [0, 0, 0, 1], atol=1e-12)

    def test_rotation_submatrix_orthogonal(self):
        from robotics.ik.jacobian_ik import ForwardKinematics
        T = ForwardKinematics.dh_matrix(0.2, math.pi / 4, 0.1, 1.2)
        R = T[:3, :3]
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-10)


# ══════════════════════════════════════════════════════════════════════════════
#  G. ForwardKinematics.compute
# ══════════════════════════════════════════════════════════════════════════════

class TestFKCompute:

    def test_returns_tuple(self):
        fk  = _fk()
        out = fk.compute(_zero_angles())
        assert isinstance(out, tuple)
        assert len(out) == 2

    def test_T_ee_shape(self):
        fk   = _fk()
        T_ee, frames = fk.compute(_zero_angles())
        assert T_ee.shape == (4, 4)

    def test_frames_count(self):
        # frames[0] = world; frames[1..6] = cumulative joints → total 7
        fk   = _fk()
        T_ee, frames = fk.compute(_zero_angles())
        assert len(frames) == 7   # n+1 for 6-DOF arm

    def test_last_frame_equals_T_ee(self):
        fk   = _fk()
        T_ee, frames = fk.compute(_zero_angles())
        np.testing.assert_allclose(T_ee, frames[-1], atol=1e-12)

    def test_first_frame_is_identity(self):
        fk   = _fk()
        _, frames = fk.compute(_zero_angles())
        np.testing.assert_allclose(frames[0], np.eye(4), atol=1e-12)

    def test_T_ee_last_row(self):
        fk   = _fk()
        T_ee, _ = fk.compute(_zero_angles())
        np.testing.assert_allclose(T_ee[3], [0, 0, 0, 1], atol=1e-12)

    def test_rotation_submatrix_orthogonal(self):
        fk   = _fk()
        T_ee, _ = fk.compute(_random_angles())
        R = T_ee[:3, :3]
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-9)


# ══════════════════════════════════════════════════════════════════════════════
#  H. FK consistency
# ══════════════════════════════════════════════════════════════════════════════

class TestFKConsistency:

    def test_end_effector_position_matches_compute(self):
        fk     = _fk()
        angles = _random_angles()
        T_ee, _ = fk.compute(angles)
        p_pos   = fk.end_effector_position(angles)
        np.testing.assert_allclose(p_pos, T_ee[:3, 3], atol=1e-12)

    def test_end_effector_transform_matches_compute(self):
        fk     = _fk()
        angles = _random_angles()
        T1, _  = fk.compute(angles)
        T2     = fk.end_effector_transform(angles)
        np.testing.assert_allclose(T1, T2, atol=1e-12)

    def test_different_angles_different_position(self):
        fk  = _fk()
        p1  = fk.end_effector_position(_zero_angles())
        p2  = fk.end_effector_position(_random_angles(seed=5))
        assert not np.allclose(p1, p2)


# ══════════════════════════════════════════════════════════════════════════════
#  I. FK -- analytic value for single-joint single-link arm
# ══════════════════════════════════════════════════════════════════════════════

class TestFKAnalytic:

    def test_single_link_rotation(self):
        """
        1-DOF arm: joint 1 rotates about z, link length a=1.0.
        At θ=0: end-effector at (1, 0, 0).
        At θ=π/2: end-effector at (0, 1, 0).
        """
        from robotics.ik.jacobian_ik import (
            ArmConfig, DHParam, JointLimit, ForwardKinematics
        )
        arm = ArmConfig(
            dh_params    = [DHParam(a=1.0, alpha=0.0, d=0.0, theta0=0.0)],
            joint_limits = [JointLimit(-math.pi, math.pi)],
        )
        fk = ForwardKinematics(arm)

        p0 = fk.end_effector_position(np.array([0.0]))
        np.testing.assert_allclose(p0, [1.0, 0.0, 0.0], atol=1e-10)

        p90 = fk.end_effector_position(np.array([math.pi / 2]))
        np.testing.assert_allclose(p90, [0.0, 1.0, 0.0], atol=1e-9)

    def test_two_link_planar_arm(self):
        """
        2-DOF planar arm: both joints rotate about z, L1=L2=0.5.
        At θ=[0,0]: end-effector at (1.0, 0, 0).
        At θ=[π/2, 0]: end-effector at (0, 0.5, 0) + (0, 0.5, 0) = (0, 1.0, 0).
        """
        from robotics.ik.jacobian_ik import (
            ArmConfig, DHParam, JointLimit, ForwardKinematics
        )
        arm = ArmConfig(
            dh_params = [
                DHParam(a=0.5, alpha=0.0, d=0.0, theta0=0.0),
                DHParam(a=0.5, alpha=0.0, d=0.0, theta0=0.0),
            ],
            joint_limits = [JointLimit(-math.pi, math.pi)] * 2,
        )
        fk = ForwardKinematics(arm)

        # Both zero: chain fully extended along x
        p00 = fk.end_effector_position(np.array([0.0, 0.0]))
        np.testing.assert_allclose(p00, [1.0, 0.0, 0.0], atol=1e-10)

        # First joint 90°, second 0°: both links point along y
        p90 = fk.end_effector_position(np.array([math.pi / 2, 0.0]))
        np.testing.assert_allclose(p90, [0.0, 1.0, 0.0], atol=1e-9)


# ══════════════════════════════════════════════════════════════════════════════
#  J. JacobianCalculator -- shape
# ══════════════════════════════════════════════════════════════════════════════

class TestJacobianShape:

    def test_full_jacobian_shape(self):
        jac    = _jac()
        angles = _random_angles()
        J = jac.compute(angles)
        assert J.shape == (6, 6)

    def test_position_jacobian_shape(self):
        jac    = _jac()
        angles = _random_angles()
        Jv = jac.position_jacobian(angles)
        assert Jv.shape == (3, 6)

    def test_zero_config_shape(self):
        jac = _jac()
        J   = jac.compute(_zero_angles())
        assert J.shape == (6, 6)


# ══════════════════════════════════════════════════════════════════════════════
#  K. Jacobian -- not all zeros
# ══════════════════════════════════════════════════════════════════════════════

class TestJacobianNonZero:

    def test_not_all_zero_generic_config(self):
        jac    = _jac()
        angles = _random_angles(seed=3)
        J = jac.compute(angles)
        assert not np.allclose(J, 0.0)

    def test_position_part_not_all_zero(self):
        jac    = _jac()
        angles = _random_angles(seed=4)
        Jv = jac.position_jacobian(angles)
        assert not np.allclose(Jv, 0.0)


# ══════════════════════════════════════════════════════════════════════════════
#  L. Jacobian -- finite values
# ══════════════════════════════════════════════════════════════════════════════

class TestJacobianFinite:

    def test_no_nan_or_inf(self):
        jac = _jac()
        for seed in range(10):
            angles = _random_angles(seed=seed)
            J = jac.compute(angles)
            assert np.all(np.isfinite(J)), f"Non-finite Jacobian at seed={seed}"


# ══════════════════════════════════════════════════════════════════════════════
#  M. Jacobian -- numerical finite-difference verification
# ══════════════════════════════════════════════════════════════════════════════

class TestJacobianNumerical:
    """
    Compare the analytic Jacobian (position part) with a central-difference
    numerical Jacobian to verify correctness.
    """

    def _numerical_jacobian_pos(self, fk, angles: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        n  = len(angles)
        Jn = np.zeros((3, n), dtype=np.float64)
        for i in range(n):
            a_plus  = angles.copy(); a_plus[i]  += eps
            a_minus = angles.copy(); a_minus[i] -= eps
            p_plus  = fk.end_effector_position(a_plus)
            p_minus = fk.end_effector_position(a_minus)
            Jn[:, i] = (p_plus - p_minus) / (2 * eps)
        return Jn

    def test_position_jacobian_matches_finite_diff(self):
        fk     = _fk()
        jac    = _jac()
        angles = _random_angles(seed=7)

        J_analytic  = jac.position_jacobian(angles)
        J_numerical = self._numerical_jacobian_pos(fk, angles)

        np.testing.assert_allclose(
            J_analytic, J_numerical, atol=1e-5,
            err_msg="Analytic and numerical position Jacobians disagree",
        )

    def test_finite_diff_multiple_configs(self):
        fk  = _fk()
        jac = _jac()
        for seed in [1, 5, 9, 13]:
            angles = _random_angles(seed=seed)
            J_a = jac.position_jacobian(angles)
            J_n = self._numerical_jacobian_pos(fk, angles)
            np.testing.assert_allclose(J_a, J_n, atol=1e-5,
                                       err_msg=f"Jacobian mismatch at seed={seed}")


# ══════════════════════════════════════════════════════════════════════════════
#  N. IKResult construction
# ══════════════════════════════════════════════════════════════════════════════

class TestIKResult:

    def _make_result(self):
        from robotics.ik.jacobian_ik import IKResult
        return IKResult(
            converged           = True,
            joint_angles_rad    = np.zeros(6),
            joint_angles_deg    = np.zeros(6),
            target_position     = np.array([0.3, 0.1, 0.4]),
            final_position      = np.array([0.3, 0.1, 0.4]),
            position_error_m    = 0.00045,
            iterations          = 35,
            solve_time_s        = 0.007,
            per_iter_error      = [0.1, 0.05, 0.001],
            within_joint_limits = True,
        )

    def test_display_is_string(self):
        r = self._make_result()
        assert isinstance(r.display(), str)

    def test_to_dict_returns_dict(self):
        r = self._make_result()
        assert isinstance(r.to_dict(), dict)

    def test_to_dict_keys(self):
        r = self._make_result()
        d = r.to_dict()
        required = (
            "converged", "iterations", "position_error_mm", "solve_time_ms",
            "joint_angles_deg", "joint_angles_rad", "target_position",
            "final_position", "within_joint_limits",
        )
        for k in required:
            assert k in d, f"Missing key: {k}"


# ══════════════════════════════════════════════════════════════════════════════
#  O. IKResult.display -- contains all required fields
# ══════════════════════════════════════════════════════════════════════════════

class TestIKResultDisplay:

    def test_display_contains_target_pose(self):
        ik     = _ik()
        target = _reachable_target(seed=1)
        result = ik.solve(target)
        s = result.display()
        assert "TARGET" in s

    def test_display_contains_achieved_pose(self):
        ik     = _ik()
        target = _reachable_target(seed=2)
        result = ik.solve(target)
        s = result.display()
        assert "END-EFFECTOR" in s

    def test_display_contains_joint_angles(self):
        ik     = _ik()
        target = _reachable_target(seed=3)
        result = ik.solve(target)
        s = result.display()
        assert "JOINT ANGLES" in s
        assert "θ1" in s

    def test_display_contains_position_error(self):
        ik     = _ik()
        target = _reachable_target(seed=4)
        result = ik.solve(target)
        s = result.display()
        assert "POSITION ERROR" in s

    def test_display_contains_iterations(self):
        ik     = _ik()
        target = _reachable_target(seed=5)
        result = ik.solve(target)
        s = result.display()
        assert "Iterations" in s

    def test_display_contains_convergence(self):
        ik     = _ik()
        target = _reachable_target(seed=6)
        result = ik.solve(target)
        s = result.display()
        assert "Convergence" in s


# ══════════════════════════════════════════════════════════════════════════════
#  P. JacobianIK construction
# ══════════════════════════════════════════════════════════════════════════════

class TestJacobianIKConstruct:

    def test_arm_stored(self):
        from robotics.ik.jacobian_ik import JacobianIK
        ik = JacobianIK()
        assert ik.arm is not None
        assert ik.arm.dof == 6

    def test_cfg_stored(self):
        from robotics.ik.jacobian_ik import JacobianIK
        ik = JacobianIK()
        assert ik.cfg is not None

    def test_fk_exists(self):
        from robotics.ik.jacobian_ik import JacobianIK
        ik = JacobianIK()
        assert ik.fk is not None

    def test_jac_exists(self):
        from robotics.ik.jacobian_ik import JacobianIK
        ik = JacobianIK()
        assert ik.jac is not None

    def test_status_field(self):
        from robotics.ik.jacobian_ik import JacobianIK
        assert JacobianIK.STATUS == "PHASE_11"


# ══════════════════════════════════════════════════════════════════════════════
#  Q. JacobianIK.solve -- close-to-origin reachable target
# ══════════════════════════════════════════════════════════════════════════════

class TestJacobianIKSolveBasic:

    def test_solve_returns_result(self):
        ik     = _ik()
        target = _reachable_target(seed=10)
        result = ik.solve(target)
        from robotics.ik.jacobian_ik import IKResult
        assert isinstance(result, IKResult)

    def test_solve_reachable_target_converges(self):
        ik     = _ik(max_iter=500, tol=1e-3, damping=0.05)
        target = _reachable_target(seed=11)
        result = ik.solve(target)
        assert result.converged, (
            f"Did not converge: error={result.position_error_m*1000:.3f} mm "
            f"after {result.iterations} iters"
        )

    def test_solve_correct_n_joint_angles(self):
        ik     = _ik()
        target = _reachable_target(seed=12)
        result = ik.solve(target)
        assert result.joint_angles_rad.shape == (6,)
        assert result.joint_angles_deg.shape == (6,)

    def test_solve_degrees_match_radians(self):
        ik     = _ik()
        target = _reachable_target(seed=13)
        result = ik.solve(target)
        np.testing.assert_allclose(
            result.joint_angles_deg,
            np.degrees(result.joint_angles_rad),
            atol=1e-10,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  R. IKResult structure from solve
# ══════════════════════════════════════════════════════════════════════════════

class TestIKResultFromSolve:

    def test_target_position_stored(self):
        ik     = _ik()
        target = np.array([0.30, 0.05, 0.40])
        result = ik.solve(target)
        np.testing.assert_allclose(result.target_position, target, atol=1e-12)

    def test_final_position_is_ndarray(self):
        ik     = _ik()
        result = ik.solve(_reachable_target(seed=20))
        assert isinstance(result.final_position, np.ndarray)
        assert result.final_position.shape == (3,)

    def test_per_iter_error_is_list(self):
        ik     = _ik()
        result = ik.solve(_reachable_target(seed=21))
        assert isinstance(result.per_iter_error, list)
        assert len(result.per_iter_error) >= 1

    def test_solve_time_positive(self):
        ik     = _ik()
        result = ik.solve(_reachable_target(seed=22))
        assert result.solve_time_s >= 0.0

    def test_iterations_positive(self):
        ik     = _ik()
        result = ik.solve(_reachable_target(seed=23))
        assert result.iterations >= 1


# ══════════════════════════════════════════════════════════════════════════════
#  S. Position error ≤ tolerance on convergence
# ══════════════════════════════════════════════════════════════════════════════

class TestIKPositionError:

    def test_error_below_tolerance_when_converged(self):
        tol  = 1e-3
        ik   = _ik(max_iter=500, tol=tol, damping=0.05)
        for seed in range(5):
            target = _reachable_target(seed=seed + 30)
            result = ik.solve(target)
            if result.converged:
                assert result.position_error_m <= tol + 1e-9, (
                    f"seed={seed}: error={result.position_error_m*1e3:.4f} mm > tol={tol*1e3} mm"
                )

    def test_final_position_matches_error(self):
        ik     = _ik()
        target = _reachable_target(seed=40)
        result = ik.solve(target)
        computed_error = float(np.linalg.norm(result.target_position - result.final_position))
        assert computed_error == pytest.approx(result.position_error_m, abs=1e-10)


# ══════════════════════════════════════════════════════════════════════════════
#  T. Joint angles within limits on convergence
# ══════════════════════════════════════════════════════════════════════════════

class TestJointLimitsRespected:

    def test_all_angles_within_bounds(self):
        ik  = _ik(max_iter=500, tol=1e-3)
        for seed in range(5):
            target = _reachable_target(seed=seed + 50)
            result = ik.solve(target)
            for j, (angle, lim) in enumerate(
                zip(result.joint_angles_rad, ik.arm.joint_limits)
            ):
                assert lim.lower - 1e-9 <= angle <= lim.upper + 1e-9, (
                    f"Joint {j+1}: angle={math.degrees(angle):.2f}° "
                    f"outside [{math.degrees(lim.lower):.1f}, {math.degrees(lim.upper):.1f}]°"
                )

    def test_within_joint_limits_flag_true_when_converged(self):
        ik     = _ik(max_iter=500)
        target = _reachable_target(seed=55)
        result = ik.solve(target)
        if result.converged:
            assert result.within_joint_limits


# ══════════════════════════════════════════════════════════════════════════════
#  U. per_iter_error is generally decreasing
# ══════════════════════════════════════════════════════════════════════════════

class TestIKConvergence:

    def test_error_generally_decreasing(self):
        """
        The error need not be strictly monotone (DLS can oscillate slightly)
        but the final error should be less than the initial error.
        """
        ik     = _ik(max_iter=200, tol=1e-4, damping=0.05)
        target = _reachable_target(seed=60)
        result = ik.solve(target)
        if result.converged and len(result.per_iter_error) > 1:
            assert result.per_iter_error[-1] < result.per_iter_error[0]

    def test_per_iter_error_length_equals_iterations(self):
        ik     = _ik(max_iter=50, tol=1e-3)
        target = _reachable_target(seed=61)
        result = ik.solve(target)
        assert len(result.per_iter_error) == result.iterations


# ══════════════════════════════════════════════════════════════════════════════
#  V. Iterations ≤ max_iterations
# ══════════════════════════════════════════════════════════════════════════════

class TestIterationBound:

    def test_iterations_bounded(self):
        max_iter = 30
        ik       = _ik(max_iter=max_iter, tol=1e-10)   # very tight tol — likely won't converge
        target   = np.array([0.60, 0.40, 0.30])
        result   = ik.solve(target)
        assert result.iterations <= max_iter


# ══════════════════════════════════════════════════════════════════════════════
#  W. Unreachable target — does not crash
# ══════════════════════════════════════════════════════════════════════════════

class TestIKUnreachable:

    def test_far_target_returns_result_gracefully(self):
        ik = _ik(max_iter=50, tol=1e-6)
        # Target 10 metres away — well beyond arm reach
        target = np.array([10.0, 10.0, 10.0])
        result = ik.solve(target)
        # Should not raise — may or may not converge
        from robotics.ik.jacobian_ik import IKResult
        assert isinstance(result, IKResult)

    def test_far_target_not_converged(self):
        ik     = _ik(max_iter=30, tol=1e-6)
        target = np.array([50.0, 50.0, 50.0])
        result = ik.solve(target)
        assert not result.converged

    def test_far_target_iterations_equals_max(self):
        max_it = 20
        ik     = _ik(max_iter=max_it, tol=1e-9)
        target = np.array([50.0, 50.0, 50.0])
        result = ik.solve(target)
        assert result.iterations == max_it


# ══════════════════════════════════════════════════════════════════════════════
#  X. Custom initial angles respected
# ══════════════════════════════════════════════════════════════════════════════

class TestIKInitialAngles:

    def test_different_initials_may_yield_different_solutions(self):
        ik     = _ik(max_iter=100, tol=1e-3, damping=0.05)
        target = _reachable_target(seed=70)
        init1  = _random_angles(seed=1)
        init2  = _random_angles(seed=99)
        r1 = ik.solve(target, initial_angles=init1)
        r2 = ik.solve(target, initial_angles=init2)
        # Both should be valid IKResult objects
        from robotics.ik.jacobian_ik import IKResult
        assert isinstance(r1, IKResult)
        assert isinstance(r2, IKResult)

    def test_bad_initial_angle_shape_raises(self):
        ik = _ik()
        with pytest.raises(ValueError, match="initial_angles"):
            ik.solve(np.array([0.3, 0.1, 0.4]), initial_angles=np.zeros(3))


# ══════════════════════════════════════════════════════════════════════════════
#  Y. Orientation mode — does not crash
# ══════════════════════════════════════════════════════════════════════════════

class TestIKOrientationMode:

    def test_orientation_mode_returns_result(self):
        from robotics.ik.jacobian_ik import JacobianIK, IKConfig, ArmConfig
        cfg = IKConfig(
            max_iterations=100, tolerance_m=1e-3,
            damping=0.05, solve_orientation=True,
        )
        ik = JacobianIK(arm=ArmConfig.desk_arm_6dof(), config=cfg)
        target_pos = _reachable_target(seed=80)
        target_ori = np.array([0.0, 0.0, 1.0])  # point up
        result = ik.solve(target_pos, target_orientation=target_ori)
        from robotics.ik.jacobian_ik import IKResult
        assert isinstance(result, IKResult)

    def test_orientation_disabled_skips_ori(self):
        ik     = _ik()   # solve_orientation=False
        target = _reachable_target(seed=81)
        # Providing orientation should be silently ignored
        result = ik.solve(target, target_orientation=np.array([0.0, 0.0, 1.0]))
        from robotics.ik.jacobian_ik import IKResult
        assert isinstance(result, IKResult)


# ══════════════════════════════════════════════════════════════════════════════
#  Z. JacobianIK.reachable
# ══════════════════════════════════════════════════════════════════════════════

class TestJacobianIKReachable:

    def test_reachable_returns_bool(self):
        ik = _ik()
        assert isinstance(ik.reachable(_reachable_target(seed=90)), bool)

    def test_reachable_for_fk_position(self):
        ik     = _ik(max_iter=500, tol=1e-3, damping=0.05)
        target = _reachable_target(seed=91)
        assert ik.reachable(target)

    def test_unreachable_for_far_target(self):
        ik = _ik(max_iter=30, tol=1e-6)
        assert not ik.reachable(np.array([50.0, 50.0, 50.0]))


# ══════════════════════════════════════════════════════════════════════════════
#  AA. JacobianIK.workspace_sample
# ══════════════════════════════════════════════════════════════════════════════

class TestWorkspaceSample:

    def test_shape(self):
        ik  = _ik()
        pts = ik.workspace_sample(n_samples=50, seed=0)
        assert pts.shape == (50, 3)

    def test_finite_values(self):
        ik  = _ik()
        pts = ik.workspace_sample(n_samples=20, seed=1)
        assert np.all(np.isfinite(pts))

    def test_within_rough_reach(self):
        """
        All sampled positions should be within rough arm reach (~70 cm from base).
        """
        ik  = _ik()
        pts = ik.workspace_sample(n_samples=200, seed=2)
        distances = np.linalg.norm(pts, axis=1)
        # No point should be farther than total arm length (~61 cm) + some slack
        total_reach = sum(dh.a for dh in ik.arm.dh_params) + \
                      sum(dh.d for dh in ik.arm.dh_params)
        assert np.all(distances <= total_reach + 0.05), (
            f"Some sampled points exceed arm reach: max={distances.max():.3f}m"
        )

    def test_seed_reproducible(self):
        ik   = _ik()
        pts1 = ik.workspace_sample(n_samples=30, seed=42)
        pts2 = ik.workspace_sample(n_samples=30, seed=42)
        np.testing.assert_array_equal(pts1, pts2)


# ══════════════════════════════════════════════════════════════════════════════
#  AB. JacobianIK.from_config_dict
# ══════════════════════════════════════════════════════════════════════════════

class TestFromConfigDict:

    def test_builds_without_error(self):
        from robotics.ik.jacobian_ik import JacobianIK
        d = {
            "dof": 6,
            "max_iterations": 100,
            "tolerance_mm": 1.0,
            "damping_lambda": 0.01,
            "joint_limits": [
                [-180, 180], [-90, 90], [-150, 150],
                [-180, 180], [-90, 90], [-180, 180],
            ],
        }
        ik = JacobianIK.from_config_dict(d)
        assert ik.arm.dof == 6
        assert ik.cfg.max_iterations == 100

    def test_tolerance_converted(self):
        from robotics.ik.jacobian_ik import JacobianIK
        ik = JacobianIK.from_config_dict({"tolerance_mm": 2.5})
        assert ik.cfg.tolerance_m == pytest.approx(0.0025, abs=1e-7)


# ══════════════════════════════════════════════════════════════════════════════
#  AC. IKResult.to_dict — numeric types
# ══════════════════════════════════════════════════════════════════════════════

class TestIKResultToDict:

    def test_converged_is_bool(self):
        ik = _ik()
        d = ik.solve(_reachable_target(seed=100)).to_dict()
        assert isinstance(d["converged"], bool)

    def test_position_error_mm_is_float(self):
        ik = _ik()
        d = ik.solve(_reachable_target(seed=101)).to_dict()
        assert isinstance(d["position_error_mm"], float)

    def test_joint_angles_deg_is_list(self):
        ik = _ik()
        d = ik.solve(_reachable_target(seed=102)).to_dict()
        assert isinstance(d["joint_angles_deg"], list)
        assert len(d["joint_angles_deg"]) == 6

    def test_target_position_is_list(self):
        ik = _ik()
        d = ik.solve(_reachable_target(seed=103)).to_dict()
        assert isinstance(d["target_position"], list)
        assert len(d["target_position"]) == 3


# ══════════════════════════════════════════════════════════════════════════════
#  AD. Public imports (robotics.ik namespace)
# ══════════════════════════════════════════════════════════════════════════════

class TestPublicImports:

    def test_jacobian_ik_import(self):
        from robotics.ik import JacobianIK
        assert JacobianIK is not None

    def test_fk_import(self):
        from robotics.ik import ForwardKinematics
        assert ForwardKinematics is not None

    def test_jac_import(self):
        from robotics.ik import JacobianCalculator
        assert JacobianCalculator is not None

    def test_arm_config_import(self):
        from robotics.ik import ArmConfig, DHParam, JointLimit
        assert ArmConfig is not None
        assert DHParam   is not None
        assert JointLimit is not None

    def test_config_and_result_import(self):
        from robotics.ik import IKConfig, IKResult
        assert IKConfig is not None
        assert IKResult is not None

    def test_all_exports_accessible(self):
        import robotics.ik as ik_mod
        for name in ik_mod.__all__:
            assert hasattr(ik_mod, name), f"Missing export: {name}"
