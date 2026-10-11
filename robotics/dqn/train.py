"""
robotics/dqn/train.py
======================
Phase 9 — DQN Training loop.

Runs the full training loop locally — no cloud, no GPU required
(GPU used automatically if available).

Usage
-----
    # Quick demo (50 episodes)
    python -m robotics.dqn.train --episodes 50 --n-books 3

    # Full training run
    python -m robotics.dqn.train

Console output per episode
--------------------------
    Ep   10/200 | Reward:  25.40 | Loss:  0.0432 | Eps: 0.951 | Steps:  23 | Action: MOVE_TO_BOOK_1
    State: robot=(0.60,0.00) | placed=0/3 | remaining=3

Checkpoint is saved every --save-interval episodes and at the end.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

# ── project root on path ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from robotics.dqn.environment import WorkspaceEnv
from robotics.dqn.dqn_agent   import DQNAgent, DQNConfig
from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# TrainConfig
# ---------------------------------------------------------------------------

class TrainConfig:
    """Hyper-parameters for the training loop (separate from DQNConfig)."""

    def __init__(
        self,
        n_books:        int   = 5,
        max_episodes:   int   = 200,
        max_steps:      int   = 100,
        save_interval:  int   = 50,
        checkpoint_path: str  = "models/dqn/dqn_checkpoint.pth",
        resume:         bool  = False,
        verbose:        bool  = True,
        seed:           Optional[int] = 42,
        # DQN hypers
        learning_rate:  float = 3e-4,
        gamma:          float = 0.99,
        epsilon_start:  float = 1.0,
        epsilon_end:    float = 0.05,
        epsilon_decay:  float = 0.995,
        buffer_size:    int   = 50_000,
        batch_size:     int   = 64,
        target_update:  int   = 10,
        hidden_size:    int   = 256,
    ) -> None:
        self.n_books         = n_books
        self.max_episodes    = max_episodes
        self.max_steps       = max_steps
        self.save_interval   = save_interval
        self.checkpoint_path = checkpoint_path
        self.resume          = resume
        self.verbose         = verbose
        self.seed            = seed
        self.learning_rate   = learning_rate
        self.gamma           = gamma
        self.epsilon_start   = epsilon_start
        self.epsilon_end     = epsilon_end
        self.epsilon_decay   = epsilon_decay
        self.buffer_size     = buffer_size
        self.batch_size      = batch_size
        self.target_update   = target_update
        self.hidden_size     = hidden_size


# ---------------------------------------------------------------------------
# EpisodeResult
# ---------------------------------------------------------------------------

class EpisodeResult:
    """Statistics from one training episode."""
    __slots__ = (
        "episode", "total_reward", "mean_loss", "steps",
        "n_placed", "epsilon", "last_action",
    )

    def __init__(
        self,
        episode:      int,
        total_reward: float,
        mean_loss:    float,
        steps:        int,
        n_placed:     int,
        epsilon:      float,
        last_action:  str,
    ) -> None:
        self.episode      = episode
        self.total_reward = total_reward
        self.mean_loss    = mean_loss
        self.steps        = steps
        self.n_placed     = n_placed
        self.epsilon      = epsilon
        self.last_action  = last_action

    def __str__(self) -> str:
        return (
            f"Ep {self.episode:4d} | "
            f"Reward: {self.total_reward:7.2f} | "
            f"Loss: {self.mean_loss:7.4f} | "
            f"Eps: {self.epsilon:.3f} | "
            f"Steps: {self.steps:4d} | "
            f"Placed: {self.n_placed} | "
            f"Action: {self.last_action}"
        )


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Orchestrates the DQN training loop.

    Parameters
    ----------
    cfg : TrainConfig instance.
    """

    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg = cfg

        # Build environment
        self.env = WorkspaceEnv.make_simple(
            n_books = cfg.n_books,
            seed    = cfg.seed,
        )

        # Build agent
        dqn_cfg = DQNConfig(
            obs_size           = self.env.observation_size,
            n_actions          = self.env.n_actions,
            hidden_size        = cfg.hidden_size,
            learning_rate      = cfg.learning_rate,
            gamma              = cfg.gamma,
            epsilon_start      = cfg.epsilon_start,
            epsilon_end        = cfg.epsilon_end,
            epsilon_decay      = cfg.epsilon_decay,
            replay_buffer_size = cfg.buffer_size,
            batch_size         = cfg.batch_size,
            target_update_freq = cfg.target_update,
            checkpoint_path    = cfg.checkpoint_path,
            seed               = cfg.seed,
        )
        self.agent = DQNAgent(dqn_cfg)

        if cfg.resume:
            self.agent.load()
            log.info(f"Resumed from checkpoint: {cfg.checkpoint_path}")

        self._history: List[EpisodeResult] = []

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def run(self) -> List[EpisodeResult]:
        """
        Execute the full training loop.

        Returns
        -------
        List[EpisodeResult] — one entry per episode.
        """
        log.info(
            f"Training: {self.cfg.max_episodes} episodes | "
            f"{self.cfg.n_books} books | "
            f"device={self.agent.device}"
        )

        for ep in range(1, self.cfg.max_episodes + 1):
            result = self._run_episode(ep)
            self._history.append(result)

            if self.cfg.verbose:
                print(str(result))

            # Epsilon decay once per episode
            self.agent.decay_epsilon()

            # Periodic checkpoint save
            if ep % self.cfg.save_interval == 0:
                path = self.agent.save()
                log.info(f"Checkpoint saved: {path}")

        # Final save
        path = self.agent.save()
        log.info(f"Training complete. Final checkpoint: {path}")

        if self.cfg.verbose:
            self._print_summary()

        return self._history

    def _run_episode(self, episode: int) -> EpisodeResult:
        """Run one complete episode. Returns EpisodeResult."""
        state = self.env.reset()
        total_reward = 0.0
        losses: List[float] = []
        last_action_name = "N/A"
        last_info: dict = {}
        n_placed = 0

        for step in range(self.cfg.max_steps):
            action = self.agent.select_action(state)
            last_action_name = self.env.action_names[action]

            next_state, reward, done, info = self.env.step(action)
            last_info = info
            n_placed  = int(self.env._placed.sum())

            self.agent.push(state, action, reward, next_state, done)
            loss = self.agent.train_step()
            if loss is not None:
                losses.append(loss)

            total_reward += reward
            state = next_state

            if self.cfg.verbose and step == 0:
                # Print state header for first step of every episode
                rx, ry = self.env._robot_pos
                remaining = self.env.n_books - n_placed
                print(
                    f"  State: robot=({rx:.2f},{ry:.2f}) | "
                    f"placed={n_placed}/{self.env.n_books} | "
                    f"remaining={remaining}"
                )

            if done:
                break

        return EpisodeResult(
            episode      = episode,
            total_reward = total_reward,
            mean_loss    = float(np.mean(losses)) if losses else 0.0,
            steps        = self.env._step_count,
            n_placed     = n_placed,
            epsilon      = self.agent.epsilon,
            last_action  = last_action_name,
        )

    def _print_summary(self) -> None:
        if not self._history:
            return
        rewards = [r.total_reward for r in self._history]
        last20  = rewards[-20:] if len(rewards) >= 20 else rewards
        print("\n" + "=" * 60)
        print("  TRAINING SUMMARY")
        print("=" * 60)
        print(f"  Episodes        : {len(self._history)}")
        print(f"  Best reward     : {max(rewards):.2f}")
        print(f"  Mean reward     : {float(np.mean(rewards)):.2f}")
        print(f"  Last-20 mean    : {float(np.mean(last20)):.2f}")
        print(f"  Final epsilon   : {self._history[-1].epsilon:.4f}")
        print(f"  Buffer size     : {self.agent.buffer_size}")
        print(f"  Gradient updates: {self.agent.updates}")
        print("=" * 60)

    @property
    def history(self) -> List[EpisodeResult]:
        return list(self._history)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Train DQN agent for workspace book organisation."
    )
    p.add_argument("--episodes",      type=int,   default=200,
                   help="Number of training episodes (default 200).")
    p.add_argument("--n-books",       type=int,   default=5,
                   help="Number of books in the workspace (default 5).")
    p.add_argument("--max-steps",     type=int,   default=100,
                   help="Max steps per episode (default 100).")
    p.add_argument("--save-interval", type=int,   default=50,
                   help="Save checkpoint every N episodes (default 50).")
    p.add_argument("--checkpoint",    type=str,   default="models/dqn/dqn_checkpoint.pth",
                   help="Checkpoint file path.")
    p.add_argument("--resume",        action="store_true",
                   help="Resume from an existing checkpoint.")
    p.add_argument("--quiet",         action="store_true",
                   help="Suppress per-episode output.")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--gamma",         type=float, default=0.99)
    p.add_argument("--batch-size",    type=int,   default=64)
    p.add_argument("--buffer-size",   type=int,   default=50_000)
    return p.parse_args(argv)


def main(argv=None) -> List[EpisodeResult]:
    args = _parse_args(argv)
    cfg  = TrainConfig(
        n_books         = args.n_books,
        max_episodes    = args.episodes,
        max_steps       = args.max_steps,
        save_interval   = args.save_interval,
        checkpoint_path = args.checkpoint,
        resume          = args.resume,
        verbose         = not args.quiet,
        seed            = args.seed,
        learning_rate   = args.lr,
        gamma           = args.gamma,
        batch_size      = args.batch_size,
        buffer_size     = args.buffer_size,
    )
    trainer = Trainer(cfg)
    return trainer.run()


if __name__ == "__main__":
    main()
