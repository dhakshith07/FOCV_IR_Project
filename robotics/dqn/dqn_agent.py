"""
robotics/dqn/dqn_agent.py
==========================
Phase 9 — Deep Q-Network (DQN) agent for robot action selection.

Architecture
------------
  QNetwork   : 3-layer MLP  (obs_size -> 256 -> 256 -> n_actions)
  DQNAgent   : Double-DQN with:
                 - Online network  (trained each step)
                 - Target network  (copied every target_update_freq steps)
                 - Epsilon-greedy exploration with exponential decay
                 - Experience replay via ReplayBuffer

Device policy
-------------
  - CPU by default (no GPU required).
  - If a CUDA-capable GPU is detected it is used automatically unless
    force_cpu=True is set.

Checkpoint
----------
  save()  : saves model weights + config to <checkpoint_path>.
  load()  : restores from checkpoint; safe to call before training.
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from robotics.dqn.replay_buffer import ReplayBuffer


# ---------------------------------------------------------------------------
# DQNConfig
# ---------------------------------------------------------------------------

@dataclass
class DQNConfig:
    """Hyper-parameters for the DQN agent."""
    obs_size:           int   = 64
    n_actions:          int   = 8
    hidden_size:        int   = 256
    learning_rate:      float = 3e-4
    gamma:              float = 0.99
    epsilon_start:      float = 1.0
    epsilon_end:        float = 0.05
    epsilon_decay:      float = 0.995
    replay_buffer_size: int   = 50_000
    batch_size:         int   = 64
    target_update_freq: int   = 10
    checkpoint_path:    str   = "models/dqn/dqn_checkpoint.pth"
    force_cpu:          bool  = False
    seed:               Optional[int] = None

    @classmethod
    def from_dict(cls, d: dict) -> "DQNConfig":
        fields = {k for k in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in fields})


# ---------------------------------------------------------------------------
# QNetwork (MLP)
# ---------------------------------------------------------------------------

class QNetwork(nn.Module):
    """
    3-layer MLP approximating Q(s, a) for all actions simultaneously.

    Input  : state vector (obs_size,)
    Output : Q-values     (n_actions,)
    """

    def __init__(self, obs_size: int, n_actions: int, hidden_size: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# DQNAgent
# ---------------------------------------------------------------------------

class DQNAgent:
    """
    Double-DQN agent with experience replay.

    Parameters
    ----------
    config : DQNConfig instance.
    """

    STATUS = "PHASE_9"

    def __init__(self, config: Optional[DQNConfig] = None) -> None:
        self.cfg = config or DQNConfig()

        # Seed
        if self.cfg.seed is not None:
            random.seed(self.cfg.seed)
            np.random.seed(self.cfg.seed)
            torch.manual_seed(self.cfg.seed)

        # Device selection (CPU default, GPU if available and not forced off)
        if self.cfg.force_cpu or not torch.cuda.is_available():
            self._device = torch.device("cpu")
        else:
            self._device = torch.device("cuda")

        # Networks
        self._online = QNetwork(
            self.cfg.obs_size, self.cfg.n_actions, self.cfg.hidden_size
        ).to(self._device)
        self._target = QNetwork(
            self.cfg.obs_size, self.cfg.n_actions, self.cfg.hidden_size
        ).to(self._device)
        self._target.load_state_dict(self._online.state_dict())
        self._target.eval()

        # Optimizer
        self._optimizer = optim.Adam(
            self._online.parameters(), lr=self.cfg.learning_rate
        )

        # Replay buffer
        self._buffer = ReplayBuffer(
            capacity = self.cfg.replay_buffer_size,
            seed     = self.cfg.seed,
        )

        # Counters
        self._epsilon:    float = self.cfg.epsilon_start
        self._steps_done: int   = 0
        self._updates:    int   = 0

    @property
    def config(self) -> DQNConfig:
        """Alias for cfg."""
        return self.cfg

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        """
        Epsilon-greedy action selection. If evaluate=True, takes greedy action.

        Parameters
        ----------
        state    : Flat float32 numpy array (obs_size,).
        evaluate : If True, selects argmax Q(s, a) greedily without random exploration.

        Returns
        -------
        int — chosen action index.
        """
        if not evaluate and random.random() < self._epsilon:
            return random.randrange(self.cfg.n_actions)

        self._online.eval()
        with torch.no_grad():
            t = torch.tensor(state, dtype=torch.float32, device=self._device).unsqueeze(0)
            q_values = self._online(t)
        self._online.train()
        return int(q_values.argmax(dim=1).item())

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Compute Q-values for a state vector as 1D numpy array."""
        self._online.eval()
        with torch.no_grad():
            t = torch.tensor(state, dtype=torch.float32, device=self._device).unsqueeze(0)
            q = self._online(t)
        return q.cpu().numpy().flatten()

    @property
    def q_network(self) -> QNetwork:
        """Access the online QNetwork."""
        return self._online

    def push(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ) -> None:
        """Store one transition in the replay buffer."""
        self._buffer.push(state, action, reward, next_state, done)

    def train_step(self) -> Optional[float]:
        """
        Sample a mini-batch and perform one gradient update.

        Returns
        -------
        float   — TD loss value, or None if buffer is not yet ready.
        """
        if not self._buffer.is_ready(self.cfg.batch_size):
            return None

        states, actions, rewards, next_states, dones = self._buffer.sample(
            self.cfg.batch_size
        )

        s  = torch.tensor(states,      dtype=torch.float32, device=self._device)
        a  = torch.tensor(actions,     dtype=torch.int64,   device=self._device)
        r  = torch.tensor(rewards,     dtype=torch.float32, device=self._device)
        ns = torch.tensor(next_states, dtype=torch.float32, device=self._device)
        d  = torch.tensor(dones,       dtype=torch.float32, device=self._device)

        # Current Q-values for taken actions
        q_current = self._online(s).gather(1, a.unsqueeze(1)).squeeze(1)

        # Double-DQN target:
        #   action* = argmax_a  Q_online(s', a)
        #   target  = r + gamma * Q_target(s', action*)  * (1 - done)
        with torch.no_grad():
            best_actions   = self._online(ns).argmax(dim=1, keepdim=True)
            q_next         = self._target(ns).gather(1, best_actions).squeeze(1)
            q_target       = r + self.cfg.gamma * q_next * (1.0 - d)

        loss = F.smooth_l1_loss(q_current, q_target)

        self._optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(self._online.parameters(), max_norm=10.0)
        self._optimizer.step()

        self._updates += 1

        # Periodically sync target network
        if self._updates % self.cfg.target_update_freq == 0:
            self._target.load_state_dict(self._online.state_dict())

        return float(loss.item())

    def decay_epsilon(self) -> None:
        """Apply one step of epsilon decay (call once per episode)."""
        self._epsilon = max(
            self.cfg.epsilon_end,
            self._epsilon * self.cfg.epsilon_decay,
        )

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save(self, path: Optional[str] = None) -> str:
        """
        Save model weights and config to disk.

        Returns the path where the checkpoint was written.
        """
        path = path or self.cfg.checkpoint_path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "online_state_dict":  self._online.state_dict(),
                "target_state_dict":  self._target.state_dict(),
                "optimizer_state":    self._optimizer.state_dict(),
                "epsilon":            self._epsilon,
                "steps_done":         self._steps_done,
                "updates":            self._updates,
                "config":             asdict(self.cfg),
            },
            path,
        )
        return str(path)

    def load(self, path: Optional[str] = None) -> None:
        """
        Load a previously saved checkpoint.

        Safe to call with a non-existent path (logs a warning and skips).
        """
        path = path or self.cfg.checkpoint_path
        if not Path(path).exists():
            return   # First run — no checkpoint yet
        ckpt = torch.load(path, map_location=self._device)
        self._online.load_state_dict(ckpt["online_state_dict"])
        self._target.load_state_dict(ckpt["target_state_dict"])
        self._optimizer.load_state_dict(ckpt["optimizer_state"])
        self._epsilon    = float(ckpt.get("epsilon",    self.cfg.epsilon_end))
        self._steps_done = int(ckpt.get("steps_done",  0))
        self._updates    = int(ckpt.get("updates",     0))

    # ------------------------------------------------------------------
    # Properties / diagnostics
    # ------------------------------------------------------------------

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @property
    def device(self) -> str:
        return str(self._device)

    @property
    def buffer_size(self) -> int:
        return len(self._buffer)

    @property
    def updates(self) -> int:
        return self._updates

    def q_values(self, state: np.ndarray) -> np.ndarray:
        """Return Q-values for all actions given a state (for evaluation)."""
        self._online.eval()
        with torch.no_grad():
            t = torch.tensor(state, dtype=torch.float32, device=self._device).unsqueeze(0)
            qv = self._online(t).squeeze(0).cpu().numpy()
        self._online.train()
        return qv
