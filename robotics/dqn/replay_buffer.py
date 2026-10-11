"""
robotics/dqn/replay_buffer.py
==============================
Phase 9 — Experience Replay Buffer for DQN training.

Stores (state, action, reward, next_state, done) transitions and
provides random mini-batch sampling for off-policy learning.

Two implementations are provided:
  ReplayBuffer       : Standard uniform random sampling.
  PrioritizedBuffer  : Lightweight alias kept for future PER extension.

Both are pure Python / NumPy — no GPU dependency.
"""

from __future__ import annotations

import random
from collections import deque
from typing import List, NamedTuple, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Transition container
# ---------------------------------------------------------------------------

class Transition(NamedTuple):
    """One SARSD experience tuple stored in the buffer."""
    state:      np.ndarray
    action:     int
    reward:     float
    next_state: np.ndarray
    done:       bool


# ---------------------------------------------------------------------------
# ReplayBuffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """
    Circular experience replay buffer with uniform random sampling.

    Parameters
    ----------
    capacity    : Maximum number of transitions to store.
    seed        : Optional random seed for reproducibility.
    """

    def __init__(self, capacity: int = 50_000, seed: Optional[int] = None) -> None:
        self._capacity = capacity
        self._buffer: deque[Transition] = deque(maxlen=capacity)
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ) -> None:
        """Add one transition to the buffer (oldest is discarded when full)."""
        self._buffer.append(
            Transition(
                state      = np.array(state,      dtype=np.float32),
                action     = int(action),
                reward     = float(reward),
                next_state = np.array(next_state, dtype=np.float32),
                done       = bool(done),
            )
        )

    def sample(self, batch_size: int) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        """
        Draw a random mini-batch.

        Returns
        -------
        states, actions, rewards, next_states, dones — each as a
        numpy array with shape (batch_size, ...).

        Raises
        ------
        ValueError if buffer contains fewer than batch_size transitions.
        """
        if len(self._buffer) < batch_size:
            raise ValueError(
                f"Buffer has {len(self._buffer)} transitions, "
                f"need at least {batch_size}."
            )
        batch = self._rng.sample(list(self._buffer), batch_size)

        states      = np.stack([t.state      for t in batch])
        actions     = np.array([t.action     for t in batch], dtype=np.int64)
        rewards     = np.array([t.reward     for t in batch], dtype=np.float32)
        next_states = np.stack([t.next_state for t in batch])
        dones       = np.array([t.done       for t in batch], dtype=np.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self._buffer)

    @property
    def capacity(self) -> int:
        return self._capacity

    def is_ready(self, batch_size: int) -> bool:
        """Return True if enough transitions exist to sample a full batch."""
        return len(self._buffer) >= batch_size

    def clear(self) -> None:
        """Remove all stored transitions."""
        self._buffer.clear()
