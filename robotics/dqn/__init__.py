"""
robotics/dqn/ -- Deep Q-Network agent for robot action selection.

Phase 9 exports
---------------
DQNAgent        : Double-DQN agent (online/target networks, epsilon-greedy).
DQNConfig       : Agent hyper-parameter dataclass.
QNetwork        : The underlying MLP Q-network.
WorkspaceEnv    : Local offline training environment.
ReplayBuffer    : Circular experience replay buffer.
Transition      : Named-tuple for (s, a, r, s', done).
Trainer         : Training loop orchestrator.
TrainConfig     : Training loop configuration.
EpisodeResult   : Per-episode training statistics.
Evaluator       : Greedy policy evaluator.
EvalResult      : Per-episode evaluation result.
StepRecord      : Per-step evaluation record.
"""

from robotics.dqn.replay_buffer import ReplayBuffer, Transition
from robotics.dqn.dqn_agent     import DQNAgent, DQNConfig, QNetwork
from robotics.dqn.environment   import WorkspaceEnv
from robotics.dqn.train         import Trainer, TrainConfig, EpisodeResult
from robotics.dqn.evaluate      import Evaluator, EvalResult, StepRecord

__all__ = [
    "DQNAgent",
    "DQNConfig",
    "QNetwork",
    "WorkspaceEnv",
    "ReplayBuffer",
    "Transition",
    "Trainer",
    "TrainConfig",
    "EpisodeResult",
    "Evaluator",
    "EvalResult",
    "StepRecord",
]
