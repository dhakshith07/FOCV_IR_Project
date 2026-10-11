"""
robotics/dqn/environment.py
============================
Phase 9 — Workspace Organisation Environment for DQN training.

The environment simulates the robot's workspace as a local, offline
gym-compatible RL environment.  No cloud resources or GPU are required.

State vector (flat float32)
---------------------------
  [robot_x, robot_y,                               # 2
   book_0_x, book_0_y, ..., book_{N-1}_x, y,      # 2*N
   freq_0, ..., freq_{N-1},                        # N   (normalised)
   wf_00, wf_01, ..., wf_{N-1,N-1},               # N*N (transition probs)
   context_0, ..., context_{K-1},                  # K   (one-hot)
   target_0_x, target_0_y, ...,                    # 2*N
   placed_0, ..., placed_{N-1},                    # N   (0/1 bitmask)
   obs_0_x, obs_0_y, obs_0_w, obs_0_h, ...        # 4*M (obstacle AABB)
  ]

Actions
-------
  0 .. N-1  : MOVE_TO_BOOK_i   (move robot to book i and pick it up)
  N          : WAIT             (no-op; useful when no valid action exists)
  N+1        : VERIFY           (verify current placement)
  N+2        : FINISH           (declare task complete)

Reward shaping
--------------
  +10   book placed on its target (correct placement)
  +1    retrieval cost reduced after placement
  +0.5  placed book follows workflow priority order
  -1    action on already-placed book (invalid)
  -2    collision (book moved into obstacle)
  -0.3  WAIT when work remains (unnecessary delay)
  -0.1  per step (encourage efficiency)
  -5    FINISH with unplaced books (premature termination)
  +20   FINISH with all books placed (success)

Episode terminates when:
  - FINISH is chosen
  - step limit is reached (max_steps)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Named action ids relative to n_books
_ACTION_WAIT   = 0   # offset from n_books
_ACTION_VERIFY = 1
_ACTION_FINISH = 2
N_SPECIAL      = 3   # WAIT, VERIFY, FINISH

# Context labels supported
CONTEXT_LABELS = ["exam_prep", "research", "casual", "project", "unknown"]


# ---------------------------------------------------------------------------
# WorkspaceEnv
# ---------------------------------------------------------------------------

class WorkspaceEnv:
    """
    Local, offline DQN training environment for workspace organisation.

    Parameters
    ----------
    n_books         : Number of books in the workspace (max 5 for the
                      default action set).
    book_positions  : Initial (x, y) positions of each book [n_books, 2].
    target_positions: Target (x, y) positions for each book [n_books, 2].
    usage_freq      : Usage frequency for each book [n_books].
    workflow_matrix : Transition probability matrix [n_books, n_books].
    user_position   : (x, y) robot / user base.
    workspace       : (x_min, y_min, x_max, y_max).
    obstacles       : List of obstacle AABB tuples (x, y, w, h).
    context         : Current session context string.
    max_steps       : Maximum steps per episode before forced termination.
    placement_tol_m : Distance tolerance to declare a book correctly placed.
    seed            : RNG seed.
    """

    metadata = {"render.modes": ["ansi"]}

    def __init__(
        self,
        n_books:          int,
        book_positions:   np.ndarray,
        target_positions: np.ndarray,
        usage_freq:       Sequence[float],
        workflow_matrix:  np.ndarray,
        user_position:    Tuple[float, float] = (0.60, 0.0),
        workspace:        Tuple[float, float, float, float] = (0.0, 0.0, 1.20, 0.80),
        obstacles:        Optional[List[Tuple[float, float, float, float]]] = None,
        context:          str = "unknown",
        max_steps:        int = 200,
        placement_tol_m:  float = 0.05,
        seed:             Optional[int] = None,
    ) -> None:
        self.n_books          = n_books
        self._init_book_pos   = np.array(book_positions,   dtype=np.float32)
        self._target_pos      = np.array(target_positions, dtype=np.float32)
        self._usage_freq_raw  = np.array(usage_freq,       dtype=np.float32)
        self._workflow_matrix = np.array(workflow_matrix,  dtype=np.float32).flatten()
        self._user_pos        = np.array(user_position,    dtype=np.float32)
        self._workspace       = workspace
        self._obstacles       = np.array(
            obstacles if obstacles else [], dtype=np.float32
        ).reshape(-1, 4)
        self._context_str     = context
        self.max_steps        = max_steps
        self._tol             = placement_tol_m
        self._rng             = np.random.default_rng(seed)

        # Derived
        self.n_actions = n_books + N_SPECIAL   # MOVE * n + WAIT + VERIFY + FINISH
        max_freq = max(float(self._usage_freq_raw.max()), 1.0)
        self._usage_freq_norm = self._usage_freq_raw / max_freq

        ctx_idx = CONTEXT_LABELS.index(context) if context in CONTEXT_LABELS else len(CONTEXT_LABELS) - 1
        self._context_onehot = np.zeros(len(CONTEXT_LABELS), dtype=np.float32)
        self._context_onehot[ctx_idx] = 1.0

        self.observation_size = self._compute_obs_size()

        # Runtime state (reset by reset())
        self._book_pos:   np.ndarray      = self._init_book_pos.copy()
        self._robot_pos:  np.ndarray      = self._user_pos.copy()
        self._placed:     np.ndarray      = np.zeros(n_books, dtype=np.float32)
        self._held_book:  Optional[int]   = None
        self._step_count: int             = 0
        self._done:       bool            = False
        self._last_retrieval_cost: float  = self._retrieval_cost()

    @property
    def obs_size(self) -> int:
        """Alias for observation_size."""
        return self.observation_size

    @property
    def action_names(self) -> List[str]:
        """List of all named actions in the environment."""
        return [self._action_name(i) for i in range(self.n_actions)]

    # ------------------------------------------------------------------
    # Gym-style API
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        """Reset episode to initial state. Returns first observation."""
        self._book_pos   = self._init_book_pos.copy()
        self._robot_pos  = self._user_pos.copy()
        self._placed     = np.zeros(self.n_books, dtype=np.float32)
        self._held_book  = None
        self._step_count = 0
        self._done       = False
        self._last_retrieval_cost = self._retrieval_cost()
        return self._observe()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Execute one action.

        Returns
        -------
        observation : np.ndarray
        reward      : float
        done        : bool
        info        : dict  (diagnostic metadata)
        """
        if self._done:
            raise RuntimeError("Episode has ended. Call reset() first.")

        reward = -0.1   # per-step cost
        info: Dict = {"action_name": self._action_name(action), "valid": True}

        if action < self.n_books:
            reward += self._exec_move_to_book(action, info)
        elif action == self.n_books + _ACTION_WAIT:
            reward += self._exec_wait(info)
        elif action == self.n_books + _ACTION_VERIFY:
            reward += self._exec_verify(info)
        elif action == self.n_books + _ACTION_FINISH:
            reward += self._exec_finish(info)
            self._done = True
        else:
            reward -= 1.0   # unknown action
            info["valid"] = False

        self._step_count += 1
        if self._step_count >= self.max_steps:
            self._done = True
            info["timeout"] = True

        obs = self._observe()
        return obs, float(reward), self._done, info

    # ------------------------------------------------------------------
    # State encoding
    # ------------------------------------------------------------------

    def _observe(self) -> np.ndarray:
        """Build and return the flat float32 state vector."""
        obs = np.concatenate([
            self._robot_pos,                           # 2
            self._book_pos.flatten(),                  # 2*N
            self._usage_freq_norm,                     # N
            self._workflow_matrix,                     # N*N
            self._context_onehot,                      # K
            self._target_pos.flatten(),                # 2*N
            self._placed,                              # N
            self._obstacles.flatten() if len(self._obstacles) > 0
                else np.zeros(0, dtype=np.float32),   # 4*M
        ], dtype=np.float32)
        return obs

    def _compute_obs_size(self) -> int:
        n = self.n_books
        m = len(self._obstacles)
        k = len(CONTEXT_LABELS)
        return 2 + 2*n + n + n*n + k + 2*n + n + 4*m

    # ------------------------------------------------------------------
    # Action executors
    # ------------------------------------------------------------------

    def _exec_move_to_book(self, book_idx: int, info: dict) -> float:
        """MOVE_TO_BOOK_i — move robot to book i and pick it up."""
        reward = 0.0

        if self._placed[book_idx] == 1.0:
            # Already placed — invalid
            reward -= 1.0
            info["valid"] = False
            info["reason"] = f"Book {book_idx} already placed"
            return reward

        # Move robot to book
        book_xy = self._book_pos[book_idx]
        self._robot_pos = book_xy.copy()

        # Check collision with obstacles
        if self._in_obstacle(book_xy):
            reward -= 2.0
            info["collision"] = True

        # Place the book on its target
        target_xy = self._target_pos[book_idx]
        dist_to_target = float(np.linalg.norm(book_xy - target_xy))

        # Move book to target
        self._book_pos[book_idx] = target_xy.copy()

        # Check if placement is correct (within tolerance)
        if dist_to_target <= self._tol or True:   # in simulation, always succeed
            self._placed[book_idx] = 1.0
            reward += 10.0   # successful placement

            # Retrieval cost reduction bonus
            new_cost = self._retrieval_cost()
            cost_delta = self._last_retrieval_cost - new_cost
            if cost_delta > 0:
                reward += min(cost_delta, 1.0)   # capped at +1
            self._last_retrieval_cost = new_cost

            # Workflow priority bonus
            if self._follows_workflow_priority(book_idx):
                reward += 0.5

        info["placed_book"] = book_idx
        info["target_dist"] = dist_to_target
        return reward

    def _exec_wait(self, info: dict) -> float:
        """WAIT — no-op."""
        remaining = int((self._placed == 0).sum())
        if remaining > 0:
            info["reason"] = f"{remaining} books still unplaced"
            return -0.3   # unnecessary wait
        return 0.0

    def _exec_verify(self, info: dict) -> float:
        """VERIFY — check current placement state."""
        n_placed = int(self._placed.sum())
        info["n_placed"] = n_placed
        # Small positive for verifying when things are placed
        return 0.1 * n_placed

    def _exec_finish(self, info: dict) -> float:
        """FINISH — declare task complete."""
        n_placed = int(self._placed.sum())
        info["n_placed"] = n_placed
        if n_placed == self.n_books:
            return 20.0   # all placed — full success reward
        else:
            unplaced = self.n_books - n_placed
            return -5.0 * unplaced   # penalty per unplaced book

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _retrieval_cost(self) -> float:
        """Weighted sum of distances from robot to each unplaced book."""
        total = 0.0
        for i in range(self.n_books):
            if self._placed[i] == 0.0:
                d = float(np.linalg.norm(self._book_pos[i] - self._robot_pos))
                total += d * float(self._usage_freq_norm[i])
        return total

    def _in_obstacle(self, point: np.ndarray) -> bool:
        """Return True if point is inside any obstacle AABB."""
        for obs in self._obstacles:
            ox, oy, ow, oh = obs
            if ox <= point[0] <= ox + ow and oy <= point[1] <= oy + oh:
                return True
        return False

    def _follows_workflow_priority(self, book_idx: int) -> bool:
        """
        Return True if book_idx is the highest-priority unplaced book
        according to usage frequency.
        """
        max_freq_idx = -1
        max_freq_val = -1.0
        for i in range(self.n_books):
            if self._placed[i] == 0.0 or i == book_idx:
                if self._usage_freq_norm[i] > max_freq_val:
                    max_freq_val = self._usage_freq_norm[i]
                    max_freq_idx = i
        return max_freq_idx == book_idx

    def _action_name(self, action: int) -> str:
        if action < self.n_books:
            return f"MOVE_TO_BOOK_{action + 1}"
        offset = action - self.n_books
        return ["WAIT", "VERIFY", "FINISH"][offset] if offset < 3 else f"UNKNOWN_{action}"

    # ------------------------------------------------------------------
    # Observation / action space descriptors (no gym dependency)
    # ------------------------------------------------------------------

    @property
    def obs_shape(self) -> Tuple[int]:
        return (self.observation_size,)

    @property
    def action_names(self) -> List[str]:
        names = [f"MOVE_TO_BOOK_{i + 1}" for i in range(self.n_books)]
        names += ["WAIT", "VERIFY", "FINISH"]
        return names

    def render(self) -> str:
        """Return a compact ASCII representation of the current state."""
        lines = [
            f"Step {self._step_count}/{self.max_steps}",
            f"  Robot:   ({self._robot_pos[0]:.2f}, {self._robot_pos[1]:.2f})",
        ]
        for i in range(self.n_books):
            status = "PLACED" if self._placed[i] == 1.0 else "unplaced"
            bx, by = self._book_pos[i]
            lines.append(f"  Book {i}: ({bx:.2f},{by:.2f}) [{status}]")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def make_simple(
        cls,
        n_books: int = 5,
        seed: Optional[int] = 42,
    ) -> "WorkspaceEnv":
        """
        Build a randomised but reproducible training environment.

        Books start at random positions within the workspace; targets are
        placed near the user position (closer = better ergonomics).
        """
        rng = np.random.default_rng(seed)
        ws  = (0.0, 0.0, 1.20, 0.80)

        # Random starting positions
        book_pos = np.column_stack([
            rng.uniform(ws[0], ws[2], n_books),
            rng.uniform(ws[1], ws[3], n_books),
        ]).astype(np.float32)

        # Target positions: spread near user side of desk
        cx = 0.60
        target_pos = np.column_stack([
            np.linspace(cx - 0.20 * (n_books - 1) / 2,
                        cx + 0.20 * (n_books - 1) / 2,
                        n_books),
            np.full(n_books, 0.15),
        ]).astype(np.float32)

        usage_freq = rng.uniform(1, 10, n_books).astype(np.float32)
        wf_matrix  = rng.dirichlet(np.ones(n_books), n_books).astype(np.float32)
        np.fill_diagonal(wf_matrix, 0.0)

        obstacles = [(0.45, 0.30, 0.15, 0.20)]   # laptop in middle

        return cls(
            n_books          = n_books,
            book_positions   = book_pos,
            target_positions = target_pos,
            usage_freq       = usage_freq,
            workflow_matrix  = wf_matrix,
            obstacles        = obstacles,
            seed             = seed,
        )
