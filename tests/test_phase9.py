"""
tests/test_phase9.py
====================
Phase 9 -- Deep Q-Network (DQN) Robot Action Selection.

Test groups
-----------
  A. Transition NamedTuple          (fields, types)
  B. ReplayBuffer construction      (capacity, seed)
  C. ReplayBuffer.push              (single, overflow, type coercion)
  D. ReplayBuffer.sample            (shape, not-ready error)
  E. ReplayBuffer helpers           (is_ready, clear, len)
  F. WorkspaceEnv construction      (obs size, action count)
  G. WorkspaceEnv.reset             (returns ndarray, correct shape)
  H. WorkspaceEnv.step              (valid actions, reward sign, done flag)
  I. WorkspaceEnv actions           (MOVE_TO_BOOK places book,
                                     WAIT penalty, VERIFY, FINISH)
  J. WorkspaceEnv.render            (returns string)
  K. WorkspaceEnv.make_simple       (factory, obs shape consistent)
  L. WorkspaceEnv action names      (correct count, correct labels)
  M. DQNConfig                      (defaults, from_dict)
  N. QNetwork                       (forward shape, device)
  O. DQNAgent construction          (device, networks exist)
  P. DQNAgent.select_action         (range, greedy when eps=0, random when eps=1)
  Q. DQNAgent.push + train_step     (buffer grows, loss returned after warmup)
  R. DQNAgent.decay_epsilon         (monotone decrease, floor)
  S. DQNAgent.save + load           (round-trip, file created)
  T. DQNAgent.q_values              (shape matches n_actions)
  U. TrainConfig                    (defaults, fields)
  V. EpisodeResult                  (__str__ contains key fields)
  W. Trainer construction           (env and agent built)
  X. Trainer._run_episode           (returns EpisodeResult, reward finite)
  Y. Trainer.run (short)            (history length, checkpoint saved)
  Z. EvalResult                     (summary, to_dict)
  AA. Evaluator                     (returns EvalResult list, success field)
  AB. Public imports                (robotics.dqn namespace, __all__)

Run:
    pytest tests/test_phase9.py -v
    pytest tests/test_phase9.py -v -k "not Slow"
"""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import pytest
import torch

# ── project root on path ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers / fixtures
# ══════════════════════════════════════════════════════════════════════════════

N_BOOKS = 3   # keep small for fast tests

def _make_env(n: int = N_BOOKS, seed: int = 0) -> "WorkspaceEnv":
    from robotics.dqn.environment import WorkspaceEnv
    return WorkspaceEnv.make_simple(n_books=n, seed=seed)


def _make_agent(env, seed: int = 0) -> "DQNAgent":
    from robotics.dqn.dqn_agent import DQNAgent, DQNConfig
    cfg = DQNConfig(
        obs_size           = env.observation_size,
        n_actions          = env.n_actions,
        hidden_size        = 64,
        learning_rate      = 1e-3,
        replay_buffer_size = 500,
        batch_size         = 16,
        target_update_freq = 5,
        force_cpu          = True,
        seed               = seed,
    )
    return DQNAgent(cfg)


def _fake_state(env) -> np.ndarray:
    return np.zeros(env.observation_size, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  A. Transition NamedTuple
# ══════════════════════════════════════════════════════════════════════════════

class TestTransition:

    def test_fields_exist(self):
        from robotics.dqn.replay_buffer import Transition
        t = Transition(
            state      = np.zeros(4, dtype=np.float32),
            action     = 1,
            reward     = 0.5,
            next_state = np.ones(4, dtype=np.float32),
            done       = False,
        )
        assert t.action == 1
        assert t.reward == pytest.approx(0.5)
        assert t.done   is False

    def test_types(self):
        from robotics.dqn.replay_buffer import Transition
        t = Transition(np.zeros(3, np.float32), 0, 1.0, np.zeros(3, np.float32), True)
        assert isinstance(t.state, np.ndarray)
        assert isinstance(t.action, int)


# ══════════════════════════════════════════════════════════════════════════════
#  B. ReplayBuffer construction
# ══════════════════════════════════════════════════════════════════════════════

class TestReplayBufferConstruct:

    def test_initial_len_zero(self):
        from robotics.dqn.replay_buffer import ReplayBuffer
        buf = ReplayBuffer(capacity=100)
        assert len(buf) == 0

    def test_capacity_stored(self):
        from robotics.dqn.replay_buffer import ReplayBuffer
        buf = ReplayBuffer(capacity=999)
        assert buf.capacity == 999

    def test_not_ready_when_empty(self):
        from robotics.dqn.replay_buffer import ReplayBuffer
        buf = ReplayBuffer(capacity=100)
        assert not buf.is_ready(32)


# ══════════════════════════════════════════════════════════════════════════════
#  C. ReplayBuffer.push
# ══════════════════════════════════════════════════════════════════════════════

class TestReplayBufferPush:

    def test_len_increases(self):
        from robotics.dqn.replay_buffer import ReplayBuffer
        buf = ReplayBuffer(capacity=50)
        for i in range(5):
            buf.push(np.zeros(4, np.float32), 0, 0.0, np.zeros(4, np.float32), False)
        assert len(buf) == 5

    def test_overflow_capped_at_capacity(self):
        from robotics.dqn.replay_buffer import ReplayBuffer
        buf = ReplayBuffer(capacity=10)
        for i in range(25):
            buf.push(np.zeros(4, np.float32), 0, float(i), np.zeros(4, np.float32), False)
        assert len(buf) == 10

    def test_type_coercion(self):
        from robotics.dqn.replay_buffer import ReplayBuffer, Transition
        buf = ReplayBuffer(capacity=10)
        buf.push([1.0, 2.0], 2, 3, [4.0, 5.0], 0)   # pass lists and ints
        t = list(buf._buffer)[0]
        assert t.state.dtype  == np.float32
        assert isinstance(t.action, int)
        assert isinstance(t.done, bool)


# ══════════════════════════════════════════════════════════════════════════════
#  D. ReplayBuffer.sample
# ══════════════════════════════════════════════════════════════════════════════

class TestReplayBufferSample:

    def _fill(self, buf, n: int, obs_size: int = 4):
        for _ in range(n):
            buf.push(
                np.random.rand(obs_size).astype(np.float32),
                int(np.random.randint(0, 3)),
                float(np.random.randn()),
                np.random.rand(obs_size).astype(np.float32),
                bool(np.random.randint(0, 2)),
            )

    def test_sample_shapes(self):
        from robotics.dqn.replay_buffer import ReplayBuffer
        buf = ReplayBuffer(capacity=200, seed=0)
        self._fill(buf, 100, obs_size=8)
        s, a, r, ns, d = buf.sample(32)
        assert s.shape  == (32, 8)
        assert a.shape  == (32,)
        assert r.shape  == (32,)
        assert ns.shape == (32, 8)
        assert d.shape  == (32,)

    def test_sample_not_ready_raises(self):
        from robotics.dqn.replay_buffer import ReplayBuffer
        buf = ReplayBuffer(capacity=100)
        self._fill(buf, 10)
        with pytest.raises(ValueError, match="Buffer has"):
            buf.sample(32)

    def test_sample_dtypes(self):
        from robotics.dqn.replay_buffer import ReplayBuffer
        buf = ReplayBuffer(capacity=100, seed=1)
        self._fill(buf, 50, obs_size=4)
        s, a, r, ns, d = buf.sample(16)
        assert s.dtype  == np.float32
        assert a.dtype  == np.int64
        assert r.dtype  == np.float32
        assert d.dtype  == np.float32


# ══════════════════════════════════════════════════════════════════════════════
#  E. ReplayBuffer helpers
# ══════════════════════════════════════════════════════════════════════════════

class TestReplayBufferHelpers:

    def test_is_ready_true(self):
        from robotics.dqn.replay_buffer import ReplayBuffer
        buf = ReplayBuffer(capacity=100, seed=0)
        for _ in range(32):
            buf.push(np.zeros(4, np.float32), 0, 0.0, np.zeros(4, np.float32), False)
        assert buf.is_ready(32)

    def test_clear(self):
        from robotics.dqn.replay_buffer import ReplayBuffer
        buf = ReplayBuffer(capacity=100)
        for _ in range(50):
            buf.push(np.zeros(4, np.float32), 0, 0.0, np.zeros(4, np.float32), False)
        buf.clear()
        assert len(buf) == 0


# ══════════════════════════════════════════════════════════════════════════════
#  F. WorkspaceEnv construction
# ══════════════════════════════════════════════════════════════════════════════

class TestWorkspaceEnvConstruct:

    def test_n_actions(self):
        env = _make_env(n=3)
        # 3 MOVE + WAIT + VERIFY + FINISH = 6
        assert env.n_actions == 6

    def test_obs_size_positive(self):
        env = _make_env(n=3)
        assert env.observation_size > 0

    def test_action_names_count(self):
        env = _make_env(n=5)
        assert len(env.action_names) == env.n_actions


# ══════════════════════════════════════════════════════════════════════════════
#  G. WorkspaceEnv.reset
# ══════════════════════════════════════════════════════════════════════════════

class TestWorkspaceEnvReset:

    def test_returns_ndarray(self):
        env = _make_env()
        obs = env.reset()
        assert isinstance(obs, np.ndarray)

    def test_correct_shape(self):
        env = _make_env()
        obs = env.reset()
        assert obs.shape == (env.observation_size,)

    def test_dtype_float32(self):
        env = _make_env()
        obs = env.reset()
        assert obs.dtype == np.float32

    def test_placed_reset_to_zero(self):
        env = _make_env()
        # Manually mark something placed
        env._placed[0] = 1.0
        env.reset()
        assert env._placed.sum() == 0.0

    def test_step_count_reset(self):
        env = _make_env()
        for _ in range(5):
            env.step(0)
        env.reset()
        assert env._step_count == 0


# ══════════════════════════════════════════════════════════════════════════════
#  H. WorkspaceEnv.step (general)
# ══════════════════════════════════════════════════════════════════════════════

class TestWorkspaceEnvStep:

    def test_returns_tuple_of_four(self):
        env = _make_env()
        env.reset()
        result = env.step(0)
        assert len(result) == 4

    def test_obs_shape_after_step(self):
        env = _make_env()
        env.reset()
        obs, _, _, _ = env.step(0)
        assert obs.shape == (env.observation_size,)

    def test_reward_is_float(self):
        env = _make_env()
        env.reset()
        _, reward, _, _ = env.step(0)
        assert isinstance(reward, float)

    def test_done_is_bool(self):
        env = _make_env()
        env.reset()
        _, _, done, _ = env.step(0)
        assert isinstance(done, bool)

    def test_info_is_dict(self):
        env = _make_env()
        env.reset()
        _, _, _, info = env.step(0)
        assert isinstance(info, dict)

    def test_raises_after_done(self):
        env = _make_env(n=1)
        env.reset()
        # FINISH action
        action_finish = env.n_books + 2
        env.step(action_finish)   # should set done=True
        with pytest.raises(RuntimeError, match="reset"):
            env.step(0)


# ══════════════════════════════════════════════════════════════════════════════
#  I. WorkspaceEnv specific actions
# ══════════════════════════════════════════════════════════════════════════════

class TestWorkspaceEnvActions:

    def test_move_to_book_marks_placed(self):
        env = _make_env(n=2)
        env.reset()
        env.step(0)   # MOVE_TO_BOOK_1
        assert env._placed[0] == 1.0

    def test_move_to_placed_book_is_invalid(self):
        env = _make_env(n=2)
        env.reset()
        env.step(0)   # place book 0
        _, reward, _, info = env.step(0)   # again — invalid
        assert not info["valid"]
        assert reward < 0

    def test_wait_penalty_when_unplaced(self):
        env = _make_env(n=3)
        env.reset()
        action_wait = env.n_books   # WAIT
        _, reward, _, _ = env.step(action_wait)
        # Should incur both per-step cost and wait penalty
        assert reward < 0

    def test_verify_returns_info(self):
        env = _make_env(n=2)
        env.reset()
        action_verify = env.n_books + 1
        _, _, _, info = env.step(action_verify)
        assert "n_placed" in info

    def test_finish_with_all_placed_positive(self):
        env = _make_env(n=2)
        env.reset()
        env.step(0)   # place book 0
        env.step(1)   # place book 1
        action_finish = env.n_books + 2
        _, reward, done, info = env.step(action_finish)
        assert done
        assert reward > 0   # success reward
        assert info["n_placed"] == 2

    def test_finish_with_unplaced_negative(self):
        env = _make_env(n=3)
        env.reset()
        # Only place 1 of 3 books
        env.step(0)
        action_finish = env.n_books + 2
        _, reward, done, _ = env.step(action_finish)
        assert done
        assert reward < 0   # penalty for unplaced

    def test_step_limit_terminates(self):
        from robotics.dqn.environment import WorkspaceEnv
        env = WorkspaceEnv.make_simple(n_books=3, seed=1)
        env.max_steps = 3
        env.reset()
        done = False
        for _ in range(5):
            _, _, done, info = env.step(env.n_books)   # WAIT forever
            if done:
                break
        assert done
        assert info.get("timeout", False)


# ══════════════════════════════════════════════════════════════════════════════
#  J. WorkspaceEnv.render
# ══════════════════════════════════════════════════════════════════════════════

class TestWorkspaceEnvRender:

    def test_returns_string(self):
        env = _make_env()
        env.reset()
        s = env.render()
        assert isinstance(s, str)

    def test_contains_robot_line(self):
        env = _make_env()
        env.reset()
        assert "Robot" in env.render()

    def test_contains_book_lines(self):
        env = _make_env(n=3)
        env.reset()
        text = env.render()
        assert "Book 0" in text
        assert "Book 2" in text


# ══════════════════════════════════════════════════════════════════════════════
#  K. WorkspaceEnv.make_simple
# ══════════════════════════════════════════════════════════════════════════════

class TestWorkspaceEnvMakeSimple:

    def test_returns_env_instance(self):
        from robotics.dqn.environment import WorkspaceEnv
        env = WorkspaceEnv.make_simple(n_books=3)
        assert isinstance(env, WorkspaceEnv)

    def test_obs_shape_consistent(self):
        env = _make_env(n=4)
        obs = env.reset()
        assert obs.shape[0] == env.observation_size

    def test_different_seeds_different_books(self):
        env1 = _make_env(n=3, seed=1)
        env2 = _make_env(n=3, seed=2)
        obs1 = env1.reset()
        obs2 = env2.reset()
        assert not np.allclose(obs1, obs2)


# ══════════════════════════════════════════════════════════════════════════════
#  L. WorkspaceEnv action names
# ══════════════════════════════════════════════════════════════════════════════

class TestWorkspaceEnvActionNames:

    def test_move_labels(self):
        env = _make_env(n=3)
        names = env.action_names
        assert names[0] == "MOVE_TO_BOOK_1"
        assert names[1] == "MOVE_TO_BOOK_2"
        assert names[2] == "MOVE_TO_BOOK_3"

    def test_special_labels(self):
        env = _make_env(n=3)
        names = env.action_names
        assert names[3] == "WAIT"
        assert names[4] == "VERIFY"
        assert names[5] == "FINISH"

    def test_count_matches_n_actions(self):
        for n in (2, 3, 5):
            env = _make_env(n=n)
            assert len(env.action_names) == env.n_actions


# ══════════════════════════════════════════════════════════════════════════════
#  M. DQNConfig
# ══════════════════════════════════════════════════════════════════════════════

class TestDQNConfig:

    def test_defaults(self):
        from robotics.dqn.dqn_agent import DQNConfig
        cfg = DQNConfig()
        assert cfg.gamma           == pytest.approx(0.99)
        assert cfg.epsilon_start   == pytest.approx(1.0)
        assert cfg.epsilon_end     == pytest.approx(0.05)
        assert cfg.batch_size      == 64
        assert cfg.target_update_freq == 10

    def test_from_dict(self):
        from robotics.dqn.dqn_agent import DQNConfig
        d = {
            "obs_size": 32, "n_actions": 6,
            "learning_rate": 1e-3, "gamma": 0.95,
        }
        cfg = DQNConfig.from_dict(d)
        assert cfg.obs_size == 32
        assert cfg.gamma    == pytest.approx(0.95)

    def test_from_dict_ignores_unknown_keys(self):
        from robotics.dqn.dqn_agent import DQNConfig
        d = {"obs_size": 8, "unknown_key": 999}
        cfg = DQNConfig.from_dict(d)
        assert cfg.obs_size == 8


# ══════════════════════════════════════════════════════════════════════════════
#  N. QNetwork
# ══════════════════════════════════════════════════════════════════════════════

class TestQNetwork:

    def test_output_shape(self):
        from robotics.dqn.dqn_agent import QNetwork
        net = QNetwork(obs_size=16, n_actions=6, hidden_size=32)
        x   = torch.zeros(1, 16)
        out = net(x)
        assert out.shape == (1, 6)

    def test_batch_output_shape(self):
        from robotics.dqn.dqn_agent import QNetwork
        net = QNetwork(obs_size=32, n_actions=8, hidden_size=64)
        x   = torch.randn(16, 32)
        out = net(x)
        assert out.shape == (16, 8)

    def test_output_not_all_zero(self):
        from robotics.dqn.dqn_agent import QNetwork
        net = QNetwork(obs_size=16, n_actions=6, hidden_size=32)
        x   = torch.randn(1, 16)
        out = net(x)
        assert not torch.all(out == 0)


# ══════════════════════════════════════════════════════════════════════════════
#  O. DQNAgent construction
# ══════════════════════════════════════════════════════════════════════════════

class TestDQNAgentConstruct:

    def test_device_is_cpu(self):
        env   = _make_env()
        agent = _make_agent(env)
        assert "cpu" in agent.device

    def test_online_and_target_exist(self):
        env   = _make_env()
        agent = _make_agent(env)
        assert agent._online is not None
        assert agent._target is not None

    def test_initial_epsilon(self):
        env   = _make_env()
        agent = _make_agent(env)
        assert agent.epsilon == pytest.approx(1.0)

    def test_buffer_empty(self):
        env   = _make_env()
        agent = _make_agent(env)
        assert agent.buffer_size == 0


# ══════════════════════════════════════════════════════════════════════════════
#  P. DQNAgent.select_action
# ══════════════════════════════════════════════════════════════════════════════

class TestDQNAgentSelectAction:

    def test_action_in_valid_range(self):
        env   = _make_env()
        agent = _make_agent(env)
        state = _fake_state(env)
        for _ in range(50):
            a = agent.select_action(state)
            assert 0 <= a < env.n_actions

    def test_greedy_when_epsilon_zero(self):
        env   = _make_env()
        agent = _make_agent(env)
        agent._epsilon = 0.0
        state = _fake_state(env)
        # Same state should produce same action
        a1 = agent.select_action(state)
        a2 = agent.select_action(state)
        assert a1 == a2

    def test_returns_int(self):
        env   = _make_env()
        agent = _make_agent(env)
        a = agent.select_action(_fake_state(env))
        assert isinstance(a, int)


# ══════════════════════════════════════════════════════════════════════════════
#  Q. DQNAgent.push + train_step
# ══════════════════════════════════════════════════════════════════════════════

class TestDQNAgentTrainStep:

    def _warmup(self, agent, env, n: int = 20):
        state = env.reset()
        for _ in range(n):
            a  = agent.select_action(state)
            ns, r, done, _ = env.step(a)
            agent.push(state, a, r, ns, done)
            if done:
                state = env.reset()
            else:
                state = ns

    def test_none_before_warmup(self):
        env   = _make_env()
        agent = _make_agent(env)
        assert agent.train_step() is None

    def test_float_after_warmup(self):
        env   = _make_env()
        agent = _make_agent(env)
        self._warmup(agent, env, n=20)
        loss = agent.train_step()
        assert isinstance(loss, float)
        assert math.isfinite(loss)
        assert loss >= 0.0

    def test_buffer_grows(self):
        env   = _make_env()
        agent = _make_agent(env)
        state = env.reset()
        for _ in range(10):
            a = 0
            ns, r, done, _ = env.step(a)
            agent.push(state, a, r, ns, done)
            state = ns if not done else env.reset()
        assert agent.buffer_size == 10

    def test_updates_counter_increments(self):
        env   = _make_env()
        agent = _make_agent(env)
        self._warmup(agent, env, n=20)
        before = agent.updates
        agent.train_step()
        assert agent.updates == before + 1


# ══════════════════════════════════════════════════════════════════════════════
#  R. DQNAgent.decay_epsilon
# ══════════════════════════════════════════════════════════════════════════════

class TestDQNAgentDecayEpsilon:

    def test_epsilon_decreases(self):
        env   = _make_env()
        agent = _make_agent(env)
        eps_before = agent.epsilon
        agent.decay_epsilon()
        assert agent.epsilon < eps_before

    def test_epsilon_floor_respected(self):
        from robotics.dqn.dqn_agent import DQNConfig, DQNAgent
        cfg = DQNConfig(
            obs_size=4, n_actions=3,
            epsilon_start=0.06, epsilon_end=0.05,
            epsilon_decay=0.1,   # very aggressive decay
            force_cpu=True,
        )
        agent = DQNAgent(cfg)
        for _ in range(100):
            agent.decay_epsilon()
        assert agent.epsilon >= 0.05 - 1e-9

    def test_epsilon_monotone_decreasing(self):
        env   = _make_env()
        agent = _make_agent(env)
        prev = agent.epsilon
        for _ in range(20):
            agent.decay_epsilon()
            assert agent.epsilon <= prev + 1e-9
            prev = agent.epsilon


# ══════════════════════════════════════════════════════════════════════════════
#  S. DQNAgent.save + load
# ══════════════════════════════════════════════════════════════════════════════

class TestDQNAgentCheckpoint:

    def test_save_creates_file(self):
        env   = _make_env()
        agent = _make_agent(env)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "ckpt.pth")
            saved_path = agent.save(path)
            assert Path(saved_path).exists()

    def test_load_restores_epsilon(self):
        env   = _make_env()
        agent = _make_agent(env)
        agent._epsilon = 0.42
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "ckpt.pth")
            agent.save(path)

            # Build a fresh agent and load
            agent2 = _make_agent(env)
            agent2.load(path)
            assert agent2.epsilon == pytest.approx(0.42, abs=1e-5)

    def test_load_nonexistent_is_safe(self):
        env   = _make_env()
        agent = _make_agent(env)
        # Should not raise
        agent.load("/tmp/__nonexistent_phase9_test__.pth")

    def test_round_trip_q_values_match(self):
        env   = _make_env()
        agent = _make_agent(env)
        state = _fake_state(env)
        q_before = agent.q_values(state)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "ckpt.pth")
            agent.save(path)
            agent2 = _make_agent(env)
            agent2.load(path)
            q_after = agent2.q_values(state)
        np.testing.assert_allclose(q_before, q_after, atol=1e-5)


# ══════════════════════════════════════════════════════════════════════════════
#  T. DQNAgent.q_values
# ══════════════════════════════════════════════════════════════════════════════

class TestDQNAgentQValues:

    def test_shape_matches_n_actions(self):
        env   = _make_env()
        agent = _make_agent(env)
        qv = agent.q_values(_fake_state(env))
        assert qv.shape == (env.n_actions,)

    def test_dtype_float32(self):
        env   = _make_env()
        agent = _make_agent(env)
        qv = agent.q_values(_fake_state(env))
        assert qv.dtype == np.float32


# ══════════════════════════════════════════════════════════════════════════════
#  U. TrainConfig
# ══════════════════════════════════════════════════════════════════════════════

class TestTrainConfig:

    def test_defaults(self):
        from robotics.dqn.train import TrainConfig
        cfg = TrainConfig()
        assert cfg.max_episodes == 200
        assert cfg.n_books      == 5
        assert cfg.gamma        == pytest.approx(0.99)

    def test_custom_values(self):
        from robotics.dqn.train import TrainConfig
        cfg = TrainConfig(n_books=3, max_episodes=50, batch_size=32)
        assert cfg.n_books      == 3
        assert cfg.max_episodes == 50
        assert cfg.batch_size   == 32


# ══════════════════════════════════════════════════════════════════════════════
#  V. EpisodeResult
# ══════════════════════════════════════════════════════════════════════════════

class TestEpisodeResult:

    def _make(self):
        from robotics.dqn.train import EpisodeResult
        return EpisodeResult(
            episode      = 42,
            total_reward = 35.7,
            mean_loss    = 0.0312,
            steps        = 15,
            n_placed     = 4,
            epsilon      = 0.72,
            last_action  = "MOVE_TO_BOOK_3",
        )

    def test_str_contains_episode(self):
        r = self._make()
        assert "42" in str(r)

    def test_str_contains_reward(self):
        r = self._make()
        assert "35.70" in str(r)

    def test_str_contains_action(self):
        r = self._make()
        assert "MOVE_TO_BOOK_3" in str(r)

    def test_str_contains_loss(self):
        r = self._make()
        assert "0.0312" in str(r)


# ══════════════════════════════════════════════════════════════════════════════
#  W. Trainer construction
# ══════════════════════════════════════════════════════════════════════════════

class TestTrainerConstruct:

    def test_env_and_agent_built(self):
        from robotics.dqn.train import Trainer, TrainConfig
        from robotics.dqn.environment import WorkspaceEnv
        from robotics.dqn.dqn_agent import DQNAgent
        cfg = TrainConfig(n_books=2, max_episodes=1, seed=0)
        t = Trainer(cfg)
        assert isinstance(t.env, WorkspaceEnv)
        assert isinstance(t.agent, DQNAgent)

    def test_history_empty_before_run(self):
        from robotics.dqn.train import Trainer, TrainConfig
        cfg = TrainConfig(n_books=2, max_episodes=1, seed=0)
        t = Trainer(cfg)
        assert t.history == []


# ══════════════════════════════════════════════════════════════════════════════
#  X. Trainer._run_episode
# ══════════════════════════════════════════════════════════════════════════════

class TestTrainerRunEpisode:

    def test_returns_episode_result(self):
        from robotics.dqn.train import Trainer, TrainConfig, EpisodeResult
        cfg = TrainConfig(n_books=2, max_episodes=1, max_steps=20,
                          batch_size=8, buffer_size=50, verbose=False, seed=7)
        t = Trainer(cfg)
        result = t._run_episode(episode=1)
        assert isinstance(result, EpisodeResult)

    def test_reward_is_finite(self):
        from robotics.dqn.train import Trainer, TrainConfig
        cfg = TrainConfig(n_books=2, max_episodes=1, max_steps=20,
                          batch_size=8, buffer_size=50, verbose=False, seed=7)
        t = Trainer(cfg)
        result = t._run_episode(episode=1)
        assert math.isfinite(result.total_reward)

    def test_steps_positive(self):
        from robotics.dqn.train import Trainer, TrainConfig
        cfg = TrainConfig(n_books=2, max_episodes=1, max_steps=20,
                          batch_size=8, buffer_size=50, verbose=False, seed=7)
        t = Trainer(cfg)
        result = t._run_episode(episode=1)
        assert result.steps >= 1


# ══════════════════════════════════════════════════════════════════════════════
#  Y. Trainer.run (short run)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestTrainerRun:

    def test_history_length(self):
        from robotics.dqn.train import Trainer, TrainConfig
        cfg = TrainConfig(
            n_books=2, max_episodes=5, max_steps=20,
            batch_size=8, buffer_size=50,
            save_interval=10,   # no saves during run
            verbose=False, seed=3,
        )
        t = Trainer(cfg)
        history = t.run()
        assert len(history) == 5

    def test_checkpoint_saved(self):
        from robotics.dqn.train import Trainer, TrainConfig
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = str(Path(tmpdir) / "test.pth")
            cfg = TrainConfig(
                n_books=2, max_episodes=3, max_steps=10,
                batch_size=8, buffer_size=50,
                checkpoint_path=ckpt,
                save_interval=10,
                verbose=False, seed=5,
            )
            t = Trainer(cfg)
            t.run()
            assert Path(ckpt).exists()


# ══════════════════════════════════════════════════════════════════════════════
#  Z. EvalResult
# ══════════════════════════════════════════════════════════════════════════════

class TestEvalResult:

    def _make(self):
        from robotics.dqn.evaluate import EvalResult
        return EvalResult(
            total_reward = 55.0,
            steps_taken  = 7,
            books_placed = 5,
            n_books      = 5,
            success      = True,
            step_records = [],
        )

    def test_summary_is_string(self):
        r = self._make()
        assert isinstance(r.summary(), str)

    def test_summary_contains_reward(self):
        r = self._make()
        assert "55" in r.summary()

    def test_summary_contains_success(self):
        r = self._make()
        assert "YES" in r.summary()

    def test_to_dict_keys(self):
        r = self._make()
        d = r.to_dict()
        for key in ("total_reward", "steps_taken", "books_placed", "n_books", "success"):
            assert key in d

    def test_success_false_shows_no(self):
        from robotics.dqn.evaluate import EvalResult
        r = EvalResult(20.0, 30, 3, 5, False, [])
        assert "NO" in r.summary()


# ══════════════════════════════════════════════════════════════════════════════
#  AA. Evaluator
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestEvaluator:

    def test_returns_list_of_eval_results(self):
        from robotics.dqn.evaluate import Evaluator, EvalResult
        env   = _make_env(n=2)
        agent = _make_agent(env)
        evaluator = Evaluator(env=env, agent=agent, verbose=False)
        results = evaluator.run(n_episodes=2)
        assert len(results) == 2
        assert all(isinstance(r, EvalResult) for r in results)

    def test_epsilon_restored_after_eval(self):
        from robotics.dqn.evaluate import Evaluator
        env   = _make_env(n=2)
        agent = _make_agent(env)
        agent._epsilon = 0.77
        evaluator = Evaluator(env=env, agent=agent, verbose=False)
        evaluator.run(n_episodes=1)
        assert agent.epsilon == pytest.approx(0.77)

    def test_step_records_populated(self):
        from robotics.dqn.evaluate import Evaluator
        env   = _make_env(n=2)
        agent = _make_agent(env)
        evaluator = Evaluator(env=env, agent=agent, verbose=False)
        result = evaluator.run(n_episodes=1)[0]
        assert len(result.step_records) >= 1

    def test_success_field_consistent(self):
        from robotics.dqn.evaluate import Evaluator
        env   = _make_env(n=2)
        agent = _make_agent(env)
        evaluator = Evaluator(env=env, agent=agent, verbose=False)
        result = evaluator.run(n_episodes=1)[0]
        expected_success = result.books_placed == result.n_books
        assert result.success == expected_success


# ══════════════════════════════════════════════════════════════════════════════
#  AB. Public imports (robotics.dqn namespace)
# ══════════════════════════════════════════════════════════════════════════════

class TestPublicImports:

    def test_dqn_agent_import(self):
        from robotics.dqn import DQNAgent, DQNConfig, QNetwork
        assert DQNAgent  is not None
        assert DQNConfig is not None
        assert QNetwork  is not None

    def test_environment_import(self):
        from robotics.dqn import WorkspaceEnv
        assert WorkspaceEnv is not None

    def test_buffer_import(self):
        from robotics.dqn import ReplayBuffer, Transition
        assert ReplayBuffer is not None
        assert Transition   is not None

    def test_train_imports(self):
        from robotics.dqn import Trainer, TrainConfig, EpisodeResult
        assert Trainer       is not None
        assert TrainConfig   is not None
        assert EpisodeResult is not None

    def test_eval_imports(self):
        from robotics.dqn import Evaluator, EvalResult, StepRecord
        assert Evaluator  is not None
        assert EvalResult is not None
        assert StepRecord is not None

    def test_all_exports_accessible(self):
        import robotics.dqn as dqn_mod
        for name in dqn_mod.__all__:
            assert hasattr(dqn_mod, name), f"Missing export: {name}"
