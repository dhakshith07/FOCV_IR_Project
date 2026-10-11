"""
robotics/ik/ — Jacobian-based Inverse Kinematics (Phase 11).

Exports
-------
JacobianIK          : Iterative Damped Least Squares IK solver.
ForwardKinematics   : Explicit DH-matrix FK engine.
JacobianCalculator  : Geometric (cross-product) Jacobian builder.
ArmConfig           : Full arm definition (DH table + joint limits).
DHParam             : One DH row (a, alpha, d, theta0).
JointLimit          : Per-joint angle bounds (lower, upper).
IKConfig            : Solver hyper-parameters.
IKResult            : Rich solve result (angles, pose, error, stats).
"""

from robotics.ik.jacobian_ik import (
    JacobianIK,
    ForwardKinematics,
    JacobianCalculator,
    ArmConfig,
    DHParam,
    JointLimit,
    IKConfig,
    IKResult,
)

__all__ = [
    "JacobianIK",
    "ForwardKinematics",
    "JacobianCalculator",
    "ArmConfig",
    "DHParam",
    "JointLimit",
    "IKConfig",
    "IKResult",
]
