"""
robotics/rrt/ -- Rapidly-exploring Random Tree path planner.

Phase 10 exports
----------------
RRTPlanner      : Main planning algorithm (plan, smooth, reconstruct).
RRTConfig       : Algorithm hyper-parameter dataclass.
RRTNode         : Tree node (config + parent pointer).
PlanResult      : Rich result (path, length, nodes, time, collision status).
RRTVisualizer   : Matplotlib + ASCII visualiser.
ConfigSpace     : 2-D workspace C-space (bounds + obstacles).
Bounds2D        : Axis-aligned workspace boundary.
Obstacle        : Axis-aligned rectangular obstacle with margin.
CollisionChecker: Point / segment / path collision queries.
CollisionReport : Structured collision detection result.
SafetyValidator : Start / goal reachability validator.
"""

from robotics.rrt.configuration_space import (
    ConfigSpace,
    Bounds2D,
    Obstacle,
    Config,
)
from robotics.rrt.collision_checker import (
    CollisionChecker,
    CollisionReport,
    SafetyValidator,
)
from robotics.rrt.rrt_planner import (
    RRTPlanner,
    RRTConfig,
    RRTNode,
    PlanResult,
    RRTVisualizer,
)

__all__ = [
    "RRTPlanner",
    "RRTConfig",
    "RRTNode",
    "PlanResult",
    "RRTVisualizer",
    "ConfigSpace",
    "Bounds2D",
    "Obstacle",
    "Config",
    "CollisionChecker",
    "CollisionReport",
    "SafetyValidator",
]
