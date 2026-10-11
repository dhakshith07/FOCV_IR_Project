"""
robotics/dqn/evaluate.py
=========================
Phase 9 — DQN Evaluation (greedy policy rollout).

Loads a saved checkpoint, runs the agent in greedy mode (epsilon=0),
and reports per-step and summary statistics.

Usage
-----
    python -m robotics.dqn.evaluate --checkpoint models/dqn/dqn_checkpoint.pth

Output per step
---------------
    Step  1 | Action: MOVE_TO_BOOK_2  | Reward:  9.90 | Placed: 1/5
      State: robot=(0.60,0.15) placed=1/5 remaining=4

Summary
-------
    Total Reward : 65.30
    Steps taken  : 12
    Books placed : 5 / 5
    Success      : YES
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from robotics.dqn.environment import WorkspaceEnv
from robotics.dqn.dqn_agent   import DQNAgent, DQNConfig
from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# StepRecord
# ---------------------------------------------------------------------------

@dataclass
class StepRecord:
    """Result of one evaluation step."""
    step:        int
    action_idx:  int
    action_name: str
    reward:      float
    n_placed:    int
    n_total:     int
    done:        bool
    state_repr:  str
    q_values:    np.ndarray


# ---------------------------------------------------------------------------
# EvalResult
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    """Full evaluation result for one episode."""
    total_reward:  float
    steps_taken:   int
    books_placed:  int
    n_books:       int
    success:       bool
    step_records:  List[StepRecord]

    def summary(self) -> str:
        lines = [
            "",
            "=" * 55,
            "  DQN EVALUATION RESULT",
            "=" * 55,
            f"  Total Reward  : {self.total_reward:.4f}",
            f"  Steps Taken   : {self.steps_taken}",
            f"  Books Placed  : {self.books_placed} / {self.n_books}",
            f"  Success       : {'YES' if self.success else 'NO'}",
            "=" * 55,
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "total_reward": self.total_reward,
            "steps_taken":  self.steps_taken,
            "books_placed": self.books_placed,
            "n_books":      self.n_books,
            "success":      self.success,
        }


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    """
    Runs the trained DQN agent in greedy mode and captures per-step data.

    Parameters
    ----------
    env             : WorkspaceEnv instance.
    agent           : DQNAgent (checkpoint should already be loaded).
    verbose         : Whether to print per-step info to stdout.
    """

    def __init__(
        self,
        env:     WorkspaceEnv,
        agent:   DQNAgent,
        verbose: bool = True,
    ) -> None:
        self.env     = env
        self.agent   = agent
        self.verbose = verbose

    def run(self, n_episodes: int = 1) -> List[EvalResult]:
        """
        Run n_episodes of greedy evaluation.

        Returns
        -------
        List[EvalResult] — one per episode.
        """
        results = []
        for ep in range(1, n_episodes + 1):
            if self.verbose and n_episodes > 1:
                print(f"\n{'─' * 40}")
                print(f"  Episode {ep}/{n_episodes}")
                print(f"{'─' * 40}")
            result = self._run_episode()
            results.append(result)
            if self.verbose:
                print(result.summary())
        return results

    def _run_episode(self) -> EvalResult:
        """Run one greedy episode."""
        state = self.env.reset()
        total_reward = 0.0
        step_records: List[StepRecord] = []
        done = False
        step = 0

        # Force greedy: save and override epsilon
        saved_eps = self.agent.epsilon
        self.agent._epsilon = 0.0   # pure greedy

        if self.verbose:
            print(f"\n  Initial State:\n{self.env.render()}\n")

        while not done:
            step += 1
            q_vals  = self.agent.q_values(state)
            action  = int(q_vals.argmax())
            a_name  = self.env.action_names[action]

            next_state, reward, done, info = self.env.step(action)
            total_reward += reward

            n_placed = int(self.env._placed.sum())
            rx, ry   = self.env._robot_pos
            state_repr = (
                f"robot=({rx:.2f},{ry:.2f}) "
                f"placed={n_placed}/{self.env.n_books} "
                f"remaining={self.env.n_books - n_placed}"
            )

            rec = StepRecord(
                step        = step,
                action_idx  = action,
                action_name = a_name,
                reward      = reward,
                n_placed    = n_placed,
                n_total     = self.env.n_books,
                done        = done,
                state_repr  = state_repr,
                q_values    = q_vals,
            )
            step_records.append(rec)

            if self.verbose:
                print(
                    f"  Step {step:3d} | "
                    f"Action: {a_name:<20s} | "
                    f"Reward: {reward:7.2f} | "
                    f"Placed: {n_placed}/{self.env.n_books}"
                )
                print(f"    State: {state_repr}")
                # Show top-3 Q-values
                top3_idx = q_vals.argsort()[::-1][:3]
                qv_str = "  |  ".join(
                    f"{self.env.action_names[i]}={q_vals[i]:.3f}"
                    for i in top3_idx
                )
                print(f"    Top Q: {qv_str}")

            state = next_state

        # Restore epsilon
        self.agent._epsilon = saved_eps

        n_placed = int(self.env._placed.sum())
        return EvalResult(
            total_reward = total_reward,
            steps_taken  = step,
            books_placed = n_placed,
            n_books      = self.env.n_books,
            success      = (n_placed == self.env.n_books),
            step_records = step_records,
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Evaluate a trained DQN agent for workspace organisation."
    )
    p.add_argument("--checkpoint", type=str, default="models/dqn/dqn_checkpoint.pth",
                   help="Path to saved DQN checkpoint.")
    p.add_argument("--n-books",    type=int, default=5,
                   help="Number of books in the workspace.")
    p.add_argument("--episodes",   type=int, default=1,
                   help="Number of evaluation episodes.")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--quiet",      action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> List[EvalResult]:
    args = _parse_args(argv)

    env = WorkspaceEnv.make_simple(n_books=args.n_books, seed=args.seed)

    # Build agent with correct sizes, then load checkpoint
    cfg = DQNConfig(
        obs_size  = env.observation_size,
        n_actions = env.n_actions,
        checkpoint_path = args.checkpoint,
    )
    agent = DQNAgent(cfg)
    agent.load()
    log.info(f"Loaded checkpoint: {args.checkpoint}")

    evaluator = Evaluator(env=env, agent=agent, verbose=not args.quiet)
    results   = evaluator.run(n_episodes=args.episodes)
    return results


if __name__ == "__main__":
    main()
