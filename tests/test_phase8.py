"""
tests/test_phase8.py
====================
Phase 8 -- Genetic Algorithm Workspace Placement Optimization.

Test groups
-----------
  A. FitnessConfig             (defaults, custom values)
  B. FitnessFunction construct (valid, invalid matrix shape)
  C. _retrieval_cost           (zero distance, weighted)
  D. _workflow_cost            (zero matrix, identity, real weights)
  E. _movement_cost            (single book, multi-book ordering)
  F. _constraint_penalty       (inside, outside, boundary edge)
  G. _overlap_penalty          (no overlap, touching, overlapping pairs)
  H. _obstacle_penalty         (no obstacles, inside, partial, no-touch)
  I. FitnessFunction.evaluate  (positive scalar, ranking two chromosomes)
  J. FitnessFunction.breakdown (all keys present, sum matches evaluate)
  K. GAConfig                  (defaults, from_dict)
  L. GeneticAlgorithm.initialise (population size, seed chromosomes,
                                   costs computed, best tracked)
  M. GeneticAlgorithm._tournament_select (returns valid chromosome)
  N. GeneticAlgorithm._crossover         (child length, gene source)
  O. GeneticAlgorithm._mutate            (clamped to workspace)
  P. GeneticAlgorithm.evolve             (cost non-increasing on average,
                                          history length, best is best)
  Q. GA reproducibility                  (same seed -> same result)
  R. BookInfo                            (field access)
  S. OptimizationResult                  (summary, to_dict)
  T. PlacementOptimizer construct        (empty books error, valid)
  U. PlacementOptimizer.optimize         (cost_before/after computed,
                                          improvement = before - after,
                                          improvement_pct honest,
                                          result fields complete)
  V. PlacementOptimizer.from_workflow_analyzer
                                         (builds from WorkflowAnalyzer)
  W. End-to-end integration              (5-book scenario, cost decreases)
  X. Public imports                      (optimization namespace)

Run:
    pytest tests/test_phase8.py -v
    pytest tests/test_phase8.py -v -k "not Integration"
"""

from __future__ import annotations

import sys
import math
from pathlib import Path
from typing import List, Tuple
from unittest.mock import MagicMock

import numpy as np
import pytest

# ── path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers / fixtures
# ══════════════════════════════════════════════════════════════════════════════

WORKSPACE = (0.0, 0.0, 1.20, 0.80)   # (x_min, y_min, x_max, y_max)
USER_POS  = (0.60, 0.0)               # robot / user base

def _uniform_matrix(n: int, value: float = 0.5) -> np.ndarray:
    m = np.full((n, n), value, dtype=float)
    np.fill_diagonal(m, 0.0)
    return m


def _zero_matrix(n: int) -> np.ndarray:
    return np.zeros((n, n), dtype=float)


def _make_fitness(
    n: int = 3,
    usage_freq=None,
    wf_matrix=None,
    obstacles=None,
):
    from optimization.fitness_function import FitnessFunction, FitnessConfig
    usage_freq = usage_freq or [1.0] * n
    wf_matrix  = wf_matrix  if wf_matrix is not None else _zero_matrix(n)
    return FitnessFunction(
        usage_freq      = usage_freq,
        workflow_matrix = wf_matrix,
        user_position   = USER_POS,
        workspace       = WORKSPACE,
        obstacles       = obstacles or [],
        config          = FitnessConfig(),
    )


def _make_chromosome(n: int, x: float = 0.5, y: float = 0.4) -> list:
    """All books at the same point (worst case for overlap)."""
    return [(x + i * 0.25, y) for i in range(n)]


def _make_ga(fitness_fn, n_books: int = 3, seed: int = 42, generations: int = 10):
    from optimization.genetic_algorithm import GeneticAlgorithm, GAConfig
    cfg = GAConfig(
        population_size = 20,
        generations     = generations,
        crossover_prob  = 0.80,
        mutation_prob   = 0.20,
        tournament_size = 3,
        elite_count     = 2,
        seed            = seed,
    )
    return GeneticAlgorithm(
        fitness_fn = fitness_fn,
        workspace  = WORKSPACE,
        n_books    = n_books,
        config     = cfg,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  A. FitnessConfig
# ══════════════════════════════════════════════════════════════════════════════

class TestFitnessConfig:

    def test_defaults(self):
        from optimization.fitness_function import FitnessConfig
        cfg = FitnessConfig()
        assert cfg.w_retrieval   == pytest.approx(1.0)
        assert cfg.w_workflow    == pytest.approx(1.5)
        assert cfg.w_movement    == pytest.approx(0.8)
        assert cfg.w_constraint  == pytest.approx(10.0)
        assert cfg.w_overlap     == pytest.approx(8.0)
        assert cfg.w_obstacle    == pytest.approx(10.0)
        assert cfg.book_radius_m == pytest.approx(0.10)

    def test_custom_values(self):
        from optimization.fitness_function import FitnessConfig
        cfg = FitnessConfig(w_retrieval=2.0, book_radius_m=0.05)
        assert cfg.w_retrieval   == pytest.approx(2.0)
        assert cfg.book_radius_m == pytest.approx(0.05)


# ══════════════════════════════════════════════════════════════════════════════
#  B. FitnessFunction construction
# ══════════════════════════════════════════════════════════════════════════════

class TestFitnessFunctionConstruct:

    def test_valid_construction(self):
        ff = _make_fitness(n=3)
        assert ff is not None
        assert len(ff.usage_freq) == 3

    def test_wrong_matrix_shape_raises(self):
        from optimization.fitness_function import FitnessFunction
        with pytest.raises(ValueError, match="workflow_matrix shape"):
            FitnessFunction(
                usage_freq      = [1.0, 1.0, 1.0],
                workflow_matrix = np.zeros((2, 2)),    # wrong: 2x2 for 3 books
                user_position   = USER_POS,
                workspace       = WORKSPACE,
            )

    def test_obstacles_default_empty(self):
        ff = _make_fitness(n=2)
        assert ff.obstacles == []

    def test_obstacles_stored(self):
        obs = [(0.1, 0.1, 0.3, 0.3)]
        ff = _make_fitness(n=2, obstacles=obs)
        assert ff.obstacles == obs


# ══════════════════════════════════════════════════════════════════════════════
#  C. _retrieval_cost
# ══════════════════════════════════════════════════════════════════════════════

class TestRetrievalCost:

    def test_user_at_book_zero_distance(self):
        ff = _make_fitness(n=1, usage_freq=[5.0])
        # Book placed exactly at user position
        chromosome = [USER_POS]
        assert ff._retrieval_cost(chromosome) == pytest.approx(0.0)

    def test_single_book_distance(self):
        ff = _make_fitness(n=1, usage_freq=[4.0])
        # Book at (0.60, 0.50) — user at (0.60, 0.0)
        chromosome = [(0.60, 0.50)]
        expected = 1.0 * 0.50   # freq normalised to 1.0, dist = 0.50
        assert ff._retrieval_cost(chromosome) == pytest.approx(expected, abs=1e-6)

    def test_higher_freq_increases_cost(self):
        # Two scenarios: one book at same distance, one with higher frequency
        ff_low  = _make_fitness(n=1, usage_freq=[1.0])
        ff_high = _make_fitness(n=1, usage_freq=[10.0])
        chromosome = [(0.60, 0.50)]
        # Both normalise to 1.0 — cost should be equal
        assert ff_low._retrieval_cost(chromosome) == pytest.approx(
            ff_high._retrieval_cost(chromosome), abs=1e-6
        )

    def test_two_books_sum(self):
        ff = _make_fitness(n=2, usage_freq=[2.0, 1.0])
        c = [(USER_POS[0], USER_POS[1] + 0.4),    # dist=0.4, weight=1.0
             (USER_POS[0], USER_POS[1] + 0.2)]     # dist=0.2, weight=0.5
        # max_freq=2 -> norms: [1.0, 0.5]
        expected = 1.0 * 0.4 + 0.5 * 0.2
        assert ff._retrieval_cost(c) == pytest.approx(expected, abs=1e-6)

    def test_zero_freq_gives_zero_weight(self):
        ff = _make_fitness(n=2, usage_freq=[0.0, 0.0])
        c = [(0.5, 0.5), (0.2, 0.2)]
        assert ff._retrieval_cost(c) == pytest.approx(0.0)


# ══════════════════════════════════════════════════════════════════════════════
#  D. _workflow_cost
# ══════════════════════════════════════════════════════════════════════════════

class TestWorkflowCost:

    def test_zero_matrix_gives_zero_cost(self):
        ff = _make_fitness(n=3, wf_matrix=_zero_matrix(3))
        c = _make_chromosome(3)
        assert ff._workflow_cost(c) == pytest.approx(0.0)

    def test_same_position_gives_zero_cost(self):
        ff = _make_fitness(n=2, wf_matrix=_uniform_matrix(2, 0.5))
        c = [(0.5, 0.4), (0.5, 0.4)]   # both at same point
        assert ff._workflow_cost(c) == pytest.approx(0.0)

    def test_workflow_increases_with_distance(self):
        n = 2
        wf = np.array([[0.0, 1.0], [1.0, 0.0]])
        ff = _make_fitness(n=n, wf_matrix=wf)
        c_close = [(0.5, 0.4), (0.5, 0.5)]    # 0.1 apart
        c_far   = [(0.1, 0.4), (1.1, 0.4)]    # 1.0 apart
        assert ff._workflow_cost(c_far) > ff._workflow_cost(c_close)


# ══════════════════════════════════════════════════════════════════════════════
#  E. _movement_cost
# ══════════════════════════════════════════════════════════════════════════════

class TestMovementCost:

    def test_single_book(self):
        ff = _make_fitness(n=1, usage_freq=[3.0])
        c = [(0.60, 0.50)]
        # user=(0.60, 0.0) -> book=(0.60, 0.50), dist=0.50
        assert ff._movement_cost(c) == pytest.approx(0.50, abs=1e-6)

    def test_empty_chromosome(self):
        ff = _make_fitness(n=1)
        assert ff._movement_cost([]) == pytest.approx(0.0)

    def test_order_by_frequency(self):
        # Two books: book 0 high freq, book 1 low freq
        # Path should visit book 0 first
        ff = _make_fitness(n=2, usage_freq=[10.0, 1.0])
        book0_pos = (0.60, 0.30)   # near user
        book1_pos = (1.10, 0.70)   # far from user
        c = [book0_pos, book1_pos]

        # Visit order: book0 first (high freq), then book1
        d_user_to_book0 = math.hypot(0.60 - 0.60, 0.30 - 0.0)
        d_book0_to_book1 = math.hypot(1.10 - 0.60, 0.70 - 0.30)
        expected = d_user_to_book0 + d_book0_to_book1
        assert ff._movement_cost(c) == pytest.approx(expected, abs=1e-6)


# ══════════════════════════════════════════════════════════════════════════════
#  F. _constraint_penalty
# ══════════════════════════════════════════════════════════════════════════════

class TestConstraintPenalty:

    def test_inside_workspace_zero_penalty(self):
        ff = _make_fitness(n=2)
        c = [(0.3, 0.3), (0.9, 0.6)]
        assert ff._constraint_penalty(c) == pytest.approx(0.0)

    def test_outside_x_axis(self):
        ff = _make_fitness(n=1)
        # Book at x=-0.2 -> dx = 0.2, dy = 0
        c = [(-0.2, 0.4)]
        assert ff._constraint_penalty(c) == pytest.approx(0.2, abs=1e-6)

    def test_outside_y_axis(self):
        ff = _make_fitness(n=1)
        # Book at y=0.9 -> x_max=1.20, y_max=0.80 -> dy=0.1
        c = [(0.5, 0.90)]
        assert ff._constraint_penalty(c) == pytest.approx(0.10, abs=1e-6)

    def test_corner_violation(self):
        ff = _make_fitness(n=1)
        # (1.30, 0.90) -> dx=0.10, dy=0.10 -> hypot
        c = [(1.30, 0.90)]
        expected = math.hypot(0.10, 0.10)
        assert ff._constraint_penalty(c) == pytest.approx(expected, abs=1e-6)

    def test_boundary_edge_no_penalty(self):
        ff = _make_fitness(n=1)
        c = [(0.0, 0.0)]   # exactly on boundary
        assert ff._constraint_penalty(c) == pytest.approx(0.0, abs=1e-9)


# ══════════════════════════════════════════════════════════════════════════════
#  G. _overlap_penalty
# ══════════════════════════════════════════════════════════════════════════════

class TestOverlapPenalty:

    def test_no_overlap_far_apart(self):
        ff = _make_fitness(n=2)
        # 0.50 apart > 2 * 0.10 = 0.20
        c = [(0.2, 0.4), (0.7, 0.4)]
        assert ff._overlap_penalty(c) == pytest.approx(0.0)

    def test_same_position_max_overlap(self):
        ff = _make_fitness(n=2)
        c = [(0.5, 0.4), (0.5, 0.4)]
        # dist=0, overlap = 2*0.10 - 0 = 0.20
        assert ff._overlap_penalty(c) == pytest.approx(0.20, abs=1e-6)

    def test_three_books_all_overlapping(self):
        ff = _make_fitness(n=3)
        # All at same point: 3 pairs, each overlap = 0.20
        c = [(0.5, 0.4)] * 3
        expected = 3 * 0.20
        assert ff._overlap_penalty(c) == pytest.approx(expected, abs=1e-6)

    def test_partial_overlap(self):
        from optimization.fitness_function import FitnessConfig
        cfg = FitnessConfig(book_radius_m=0.10)
        ff  = _make_fitness(n=2)
        # dist = 0.15, min_dist = 0.20 -> overlap = 0.05
        c = [(0.5, 0.4), (0.65, 0.4)]
        assert ff._overlap_penalty(c) == pytest.approx(0.05, abs=1e-6)


# ══════════════════════════════════════════════════════════════════════════════
#  H. _obstacle_penalty
# ══════════════════════════════════════════════════════════════════════════════

class TestObstaclePenalty:

    def test_no_obstacles_zero_penalty(self):
        ff = _make_fitness(n=2, obstacles=[])
        c = [(0.5, 0.4), (0.7, 0.4)]
        assert ff._obstacle_penalty(c) == pytest.approx(0.0)

    def test_book_inside_obstacle(self):
        # obstacle covers (0.4, 0.3, 0.7, 0.5)
        # book at (0.5, 0.4), radius=0.10 -> book box (0.4, 0.3, 0.6, 0.5)
        obs = [(0.4, 0.3, 0.7, 0.5)]
        ff = _make_fitness(n=1, obstacles=obs)
        c = [(0.5, 0.4)]
        penalty = ff._obstacle_penalty(c)
        assert penalty > 0.0

    def test_book_far_from_obstacle_zero_penalty(self):
        obs = [(0.9, 0.6, 1.1, 0.8)]
        ff = _make_fitness(n=1, obstacles=obs)
        # Book well away from obstacle
        c = [(0.2, 0.2)]
        assert ff._obstacle_penalty(c) == pytest.approx(0.0)

    def test_multiple_obstacles_accumulate(self):
        obs = [
            (0.4, 0.3, 0.7, 0.5),
            (0.4, 0.3, 0.7, 0.5),   # duplicate -> double penalty
        ]
        ff_single = _make_fitness(n=1, obstacles=[(0.4, 0.3, 0.7, 0.5)])
        ff_double = _make_fitness(n=1, obstacles=obs)
        c = [(0.5, 0.4)]
        assert ff_double._obstacle_penalty(c) == pytest.approx(
            2 * ff_single._obstacle_penalty(c), abs=1e-9
        )


# ══════════════════════════════════════════════════════════════════════════════
#  I. FitnessFunction.evaluate
# ══════════════════════════════════════════════════════════════════════════════

class TestFitnessFunctionEvaluate:

    def test_returns_positive_float(self):
        ff = _make_fitness(n=3)
        c = _make_chromosome(3)
        cost = ff.evaluate(c)
        assert isinstance(cost, float)
        assert cost >= 0.0

    def test_wrong_chromosome_length_raises(self):
        ff = _make_fitness(n=3)
        with pytest.raises(ValueError, match="Chromosome length"):
            ff.evaluate([(0.5, 0.4), (0.6, 0.4)])   # only 2 genes for 3 books

    def test_better_arrangement_lower_cost(self):
        """Books near user and spread apart should cost less than far & overlapping."""
        ff = _make_fitness(n=2, usage_freq=[5.0, 3.0])

        # Good: books near user, spread
        c_good = [(0.60, 0.10), (0.65, 0.10)]
        # Bad: books far from user, heavily overlapping
        c_bad  = [(1.10, 0.70), (1.10, 0.70)]

        assert ff.evaluate(c_good) < ff.evaluate(c_bad)

    def test_deterministic(self):
        ff = _make_fitness(n=3)
        c = _make_chromosome(3)
        assert ff.evaluate(c) == ff.evaluate(c)


# ══════════════════════════════════════════════════════════════════════════════
#  J. FitnessFunction.breakdown
# ══════════════════════════════════════════════════════════════════════════════

class TestFitnessFunctionBreakdown:

    EXPECTED_KEYS = {
        "retrieval_distance",
        "workflow_distance",
        "movement_cost",
        "constraint_penalty",
        "overlap_penalty",
        "obstacle_penalty",
        "total_cost",
    }

    def test_all_keys_present(self):
        ff = _make_fitness(n=3)
        bd = ff.breakdown(_make_chromosome(3))
        assert set(bd.keys()) == self.EXPECTED_KEYS

    def test_total_matches_evaluate(self):
        ff = _make_fitness(n=3)
        c  = _make_chromosome(3)
        bd = ff.breakdown(c)
        assert bd["total_cost"] == pytest.approx(ff.evaluate(c), abs=1e-9)

    def test_all_values_non_negative(self):
        ff = _make_fitness(n=4)
        c  = _make_chromosome(4)
        bd = ff.breakdown(c)
        for k, v in bd.items():
            assert v >= 0.0, f"Negative cost component: {k}={v}"


# ══════════════════════════════════════════════════════════════════════════════
#  K. GAConfig
# ══════════════════════════════════════════════════════════════════════════════

class TestGAConfig:

    def test_defaults(self):
        from optimization.genetic_algorithm import GAConfig
        cfg = GAConfig()
        assert cfg.population_size == 100
        assert cfg.generations     == 150
        assert cfg.crossover_prob  == pytest.approx(0.80)
        assert cfg.mutation_prob   == pytest.approx(0.15)
        assert cfg.tournament_size == 5
        assert cfg.elite_count     == 5

    def test_from_dict(self):
        from optimization.genetic_algorithm import GAConfig
        d = {
            "population_size": 50,
            "generations":     20,
            "crossover_prob":  0.70,
            "mutation_prob":   0.10,
            "tournament_size": 3,
            "elite_count":     2,
            "mutation_sigma":  0.03,
        }
        cfg = GAConfig.from_dict(d)
        assert cfg.population_size == 50
        assert cfg.generations     == 20
        assert cfg.crossover_prob  == pytest.approx(0.70)
        assert cfg.mutation_sigma  == pytest.approx(0.03)

    def test_from_dict_uses_defaults_for_missing_keys(self):
        from optimization.genetic_algorithm import GAConfig
        cfg = GAConfig.from_dict({})
        assert cfg.population_size == 100


# ══════════════════════════════════════════════════════════════════════════════
#  L. GeneticAlgorithm.initialise
# ══════════════════════════════════════════════════════════════════════════════

class TestGAInitialise:

    def test_population_size(self):
        ff = _make_fitness(n=3)
        ga = _make_ga(ff, n_books=3)
        ga.initialise()
        assert len(ga._population) == 20   # population_size in _make_ga

    def test_seed_chromosome_included(self):
        ff = _make_fitness(n=2)
        ga = _make_ga(ff, n_books=2)
        seed = [(0.5, 0.4), (0.7, 0.4)]
        ga.initialise(seed_chromosomes=[seed])
        assert seed in ga._population

    def test_costs_computed(self):
        ff = _make_fitness(n=3)
        ga = _make_ga(ff, n_books=3)
        ga.initialise()
        assert len(ga._costs) == len(ga._population)
        assert all(c >= 0 for c in ga._costs)

    def test_best_cost_set(self):
        ff = _make_fitness(n=3)
        ga = _make_ga(ff, n_books=3)
        ga.initialise()
        assert ga.best_cost < float("inf")

    def test_all_positions_within_workspace(self):
        ff = _make_fitness(n=4)
        ga = _make_ga(ff, n_books=4)
        ga.initialise()
        x_min, y_min, x_max, y_max = WORKSPACE
        for ch in ga._population:
            for (x, y) in ch:
                assert x_min <= x <= x_max
                assert y_min <= y <= y_max


# ══════════════════════════════════════════════════════════════════════════════
#  M. GeneticAlgorithm._tournament_select
# ══════════════════════════════════════════════════════════════════════════════

class TestTournamentSelect:

    def test_returns_chromosome_of_correct_length(self):
        ff = _make_fitness(n=3)
        ga = _make_ga(ff, n_books=3)
        ga.initialise()
        selected = ga._tournament_select()
        assert len(selected) == 3

    def test_returns_a_valid_chromosome(self):
        ff = _make_fitness(n=2)
        ga = _make_ga(ff, n_books=2)
        ga.initialise()
        for _ in range(20):
            sel = ga._tournament_select()
            assert len(sel) == 2
            for (x, y) in sel:
                assert isinstance(x, float)
                assert isinstance(y, float)


# ══════════════════════════════════════════════════════════════════════════════
#  N. GeneticAlgorithm._crossover
# ══════════════════════════════════════════════════════════════════════════════

class TestCrossover:

    def test_children_have_correct_length(self):
        ff = _make_fitness(n=4)
        ga = _make_ga(ff, n_books=4)
        ga.initialise()
        p1 = _make_chromosome(4, 0.3, 0.3)
        p2 = _make_chromosome(4, 0.8, 0.6)
        c1, c2 = ga._crossover(p1, p2)
        assert len(c1) == 4
        assert len(c2) == 4

    def test_genes_come_from_parents(self):
        """Each gene in a child must come from one of the two parents."""
        ff = _make_fitness(n=3)
        ga = _make_ga(ff, n_books=3)
        ga.initialise()
        p1 = [(0.1, 0.1), (0.2, 0.2), (0.3, 0.3)]
        p2 = [(0.7, 0.7), (0.8, 0.8), (0.9, 0.9)]
        for _ in range(50):
            c1, c2 = ga._crossover(p1, p2)
            for i in range(3):
                assert c1[i] in (p1[i], p2[i])
                assert c2[i] in (p1[i], p2[i])

    def test_complementary_swap(self):
        """If c1[i] = p1[i] then c2[i] = p2[i] and vice-versa."""
        ff = _make_fitness(n=3)
        ga = _make_ga(ff, n_books=3)
        ga.initialise()
        p1 = [(0.1, 0.1), (0.2, 0.2), (0.3, 0.3)]
        p2 = [(0.7, 0.7), (0.8, 0.8), (0.9, 0.9)]
        for _ in range(50):
            c1, c2 = ga._crossover(p1, p2)
            for i in range(3):
                if c1[i] == p1[i]:
                    assert c2[i] == p2[i]
                else:
                    assert c1[i] == p2[i]
                    assert c2[i] == p1[i]


# ══════════════════════════════════════════════════════════════════════════════
#  O. GeneticAlgorithm._mutate
# ══════════════════════════════════════════════════════════════════════════════

class TestMutate:

    def test_output_length_unchanged(self):
        ff = _make_fitness(n=4)
        ga = _make_ga(ff, n_books=4)
        ga.initialise()
        ch = _make_chromosome(4)
        mutated = ga._mutate(ch)
        assert len(mutated) == 4

    def test_positions_clamped_to_workspace(self):
        ff = _make_fitness(n=3)
        ga = _make_ga(ff, n_books=3)
        ga.initialise()
        x_min, y_min, x_max, y_max = WORKSPACE
        # Run many mutations to trigger boundary clamping
        ch = [(0.0, 0.0)] * 3   # on lower boundary
        for _ in range(200):
            mutated = ga._mutate(ch)
            for (x, y) in mutated:
                assert x_min <= x <= x_max
                assert y_min <= y <= y_max

    def test_high_mutation_changes_some_genes(self):
        from optimization.genetic_algorithm import GeneticAlgorithm, GAConfig
        cfg = GAConfig(mutation_prob=1.0, mutation_sigma=0.05, seed=99)
        ff  = _make_fitness(n=5)
        ga  = GeneticAlgorithm(fitness_fn=ff, workspace=WORKSPACE, n_books=5, config=cfg)
        ga.initialise()
        ch  = [(0.5, 0.4)] * 5
        mutated = ga._mutate(ch)
        # With prob=1.0 at least some genes should differ (sigma>0)
        diffs = [m != o for m, o in zip(mutated, ch)]
        assert any(diffs), "Mutation with prob=1.0 should change at least one gene."


# ══════════════════════════════════════════════════════════════════════════════
#  P. GeneticAlgorithm.evolve
# ══════════════════════════════════════════════════════════════════════════════

class TestGAEvolve:

    def test_history_length_equals_generations(self):
        ff = _make_fitness(n=3)
        ga = _make_ga(ff, n_books=3, generations=5)
        ga.initialise()
        ga.evolve()
        assert len(ga.history) == 5

    def test_history_generation_numbers(self):
        ff = _make_fitness(n=2)
        ga = _make_ga(ff, n_books=2, generations=4)
        ga.initialise()
        ga.evolve()
        for i, stats in enumerate(ga.history):
            assert stats.generation == i + 1

    def test_best_cost_non_negative(self):
        ff = _make_fitness(n=3)
        ga = _make_ga(ff, n_books=3, generations=5)
        ga.initialise()
        ga.evolve()
        assert ga.best_cost >= 0.0

    def test_best_chromosome_length(self):
        ff = _make_fitness(n=3)
        ga = _make_ga(ff, n_books=3, generations=5)
        ga.initialise()
        best = ga.evolve()
        assert len(best) == 3

    def test_best_cost_is_actual_evaluated_cost(self):
        """ga.best_cost must match ff.evaluate(best_chromosome)."""
        ff = _make_fitness(n=3)
        ga = _make_ga(ff, n_books=3, generations=10)
        ga.initialise()
        best_ch = ga.evolve()
        actual  = ff.evaluate(best_ch)
        assert ga.best_cost == pytest.approx(actual, abs=1e-9)

    def test_best_cost_does_not_increase_over_generations(self):
        """Best cost in later generations must be <= best cost in earlier ones."""
        ff = _make_fitness(n=4)
        ga = _make_ga(ff, n_books=4, generations=20)
        ga.initialise()
        ga.evolve()
        best_costs = [s.best_cost for s in ga.history]
        for i in range(1, len(best_costs)):
            assert best_costs[i] <= best_costs[i - 1] + 1e-9, (
                f"Cost increased at generation {i + 1}: "
                f"{best_costs[i-1]:.6f} -> {best_costs[i]:.6f}"
            )

    def test_raises_if_not_initialised(self):
        ff = _make_fitness(n=2)
        ga = _make_ga(ff, n_books=2, generations=5)
        with pytest.raises(RuntimeError, match="initialise"):
            ga.evolve()


# ══════════════════════════════════════════════════════════════════════════════
#  Q. Reproducibility
# ══════════════════════════════════════════════════════════════════════════════

class TestReproducibility:

    def test_same_seed_same_result(self):
        n = 3

        def _run(seed):
            ff = _make_fitness(n=n)
            ga = _make_ga(ff, n_books=n, seed=seed, generations=5)
            ga.initialise()
            ga.evolve()
            return ga.best_cost

        cost1 = _run(seed=7)
        cost2 = _run(seed=7)
        assert cost1 == pytest.approx(cost2, abs=1e-9)


# ══════════════════════════════════════════════════════════════════════════════
#  R. BookInfo
# ══════════════════════════════════════════════════════════════════════════════

class TestBookInfo:

    def test_field_access(self):
        from optimization.placement_optimizer import BookInfo
        b = BookInfo(book_id=42, title="Algebra", position=(0.3, 0.5), usage_freq=7.0)
        assert b.book_id    == 42
        assert b.title      == "Algebra"
        assert b.position   == (0.3, 0.5)
        assert b.usage_freq == pytest.approx(7.0)

    def test_default_usage_freq(self):
        from optimization.placement_optimizer import BookInfo
        b = BookInfo(book_id=1, title="Test", position=(0.5, 0.4))
        assert b.usage_freq == pytest.approx(1.0)


# ══════════════════════════════════════════════════════════════════════════════
#  S. OptimizationResult
# ══════════════════════════════════════════════════════════════════════════════

class TestOptimizationResult:

    def _make_result(self):
        from optimization.placement_optimizer import OptimizationResult
        return OptimizationResult(
            current_arrangement   = [(0.5, 0.4), (0.7, 0.4)],
            optimized_arrangement = [(0.6, 0.1), (0.65, 0.1)],
            cost_before           = 5.0,
            cost_after            = 3.5,
            improvement           = 1.5,
            improvement_pct       = 30.0,
            book_titles           = ["ML", "Stats"],
            book_ids              = [1, 2],
        )

    def test_summary_is_string(self):
        r = self._make_result()
        s = r.summary()
        assert isinstance(s, str)
        assert "Cost Before" in s
        assert "Cost After"  in s
        assert "Improvement" in s

    def test_summary_contains_book_titles(self):
        r = self._make_result()
        s = r.summary()
        assert "ML"    in s
        assert "Stats" in s

    def test_to_dict_keys(self):
        r  = self._make_result()
        d  = r.to_dict()
        for key in (
            "cost_before", "cost_after", "improvement", "improvement_pct",
            "elapsed_s", "book_titles", "book_ids",
            "current_positions", "optimal_positions",
        ):
            assert key in d, f"Missing key: {key}"

    def test_to_dict_cost_values(self):
        r = self._make_result()
        d = r.to_dict()
        assert d["cost_before"]     == pytest.approx(5.0)
        assert d["cost_after"]      == pytest.approx(3.5)
        assert d["improvement"]     == pytest.approx(1.5)
        assert d["improvement_pct"] == pytest.approx(30.0)


# ══════════════════════════════════════════════════════════════════════════════
#  T. PlacementOptimizer construction
# ══════════════════════════════════════════════════════════════════════════════

class TestPlacementOptimizerConstruct:

    def test_empty_books_raises(self):
        from optimization.placement_optimizer import PlacementOptimizer
        with pytest.raises(ValueError, match="empty"):
            PlacementOptimizer(
                books           = [],
                workflow_matrix = np.zeros((0, 0)),
                user_position   = USER_POS,
                workspace       = WORKSPACE,
            )

    def test_valid_construction(self):
        from optimization.placement_optimizer import PlacementOptimizer, BookInfo
        books = [
            BookInfo(1, "A", (0.3, 0.3), 5.0),
            BookInfo(2, "B", (0.7, 0.5), 2.0),
        ]
        opt = PlacementOptimizer(
            books           = books,
            workflow_matrix = _zero_matrix(2),
            user_position   = USER_POS,
            workspace       = WORKSPACE,
        )
        assert opt.n_books == 2

    def test_last_result_none_before_optimize(self):
        from optimization.placement_optimizer import PlacementOptimizer, BookInfo
        books = [BookInfo(1, "A", (0.3, 0.3), 3.0)]
        opt = PlacementOptimizer(
            books           = books,
            workflow_matrix = _zero_matrix(1),
            user_position   = USER_POS,
            workspace       = WORKSPACE,
        )
        assert opt.last_result is None


# ══════════════════════════════════════════════════════════════════════════════
#  U. PlacementOptimizer.optimize
# ══════════════════════════════════════════════════════════════════════════════

class TestPlacementOptimizerOptimize:

    def _make_optimizer(self, n: int = 3):
        from optimization.placement_optimizer import PlacementOptimizer, BookInfo
        from optimization.genetic_algorithm import GAConfig
        books = [
            BookInfo(i + 1, f"Book_{i}", (0.9 + i * 0.05, 0.7), float(n - i))
            for i in range(n)
        ]
        wf = _uniform_matrix(n, 0.3)
        ga_cfg = GAConfig(
            population_size = 20,
            generations     = 15,
            elite_count     = 2,
            seed            = 42,
        )
        return PlacementOptimizer(
            books           = books,
            workflow_matrix = wf,
            user_position   = USER_POS,
            workspace       = WORKSPACE,
            ga_config       = ga_cfg,
        )

    def test_returns_optimization_result(self):
        from optimization.placement_optimizer import OptimizationResult
        opt    = self._make_optimizer(n=3)
        result = opt.optimize()
        assert isinstance(result, OptimizationResult)

    def test_cost_before_is_actual(self):
        """cost_before must equal ff.evaluate(current_arrangement)."""
        opt    = self._make_optimizer(n=3)
        result = opt.optimize()
        actual = opt.fitness_fn.evaluate(result.current_arrangement)
        assert result.cost_before == pytest.approx(actual, abs=1e-9)

    def test_cost_after_is_actual(self):
        """cost_after must equal ff.evaluate(optimized_arrangement)."""
        opt    = self._make_optimizer(n=3)
        result = opt.optimize()
        actual = opt.fitness_fn.evaluate(result.optimized_arrangement)
        assert result.cost_after == pytest.approx(actual, abs=1e-9)

    def test_improvement_equals_before_minus_after(self):
        opt    = self._make_optimizer(n=3)
        result = opt.optimize()
        expected = result.cost_before - result.cost_after
        assert result.improvement == pytest.approx(expected, abs=1e-9)

    def test_improvement_pct_consistent(self):
        opt    = self._make_optimizer(n=3)
        result = opt.optimize()
        if result.improvement_pct is not None and result.cost_before > 0:
            expected_pct = 100.0 * result.improvement / result.cost_before
            assert result.improvement_pct == pytest.approx(expected_pct, abs=1e-6)

    def test_no_fake_improvement_pct(self):
        """improvement_pct must be None when cost_before == 0."""
        from optimization.placement_optimizer import PlacementOptimizer, BookInfo
        from optimization.fitness_function import FitnessConfig
        from optimization.genetic_algorithm import GAConfig
        # Create a scenario where books are right at user and no workflow
        books = [BookInfo(1, "A", USER_POS, 0.0)]
        ga_cfg = GAConfig(population_size=10, generations=5, seed=1)
        # Zero weights so cost = 0
        fc = FitnessConfig(
            w_retrieval=0.0, w_workflow=0.0, w_movement=0.0,
            w_constraint=0.0, w_overlap=0.0, w_obstacle=0.0
        )
        opt = PlacementOptimizer(
            books           = books,
            workflow_matrix = _zero_matrix(1),
            user_position   = USER_POS,
            workspace       = WORKSPACE,
            fitness_config  = fc,
            ga_config       = ga_cfg,
        )
        result = opt.optimize()
        assert result.cost_before == pytest.approx(0.0)
        assert result.improvement_pct is None

    def test_book_titles_and_ids_match(self):
        opt    = self._make_optimizer(n=3)
        result = opt.optimize()
        assert len(result.book_titles) == 3
        assert len(result.book_ids)    == 3
        assert result.book_ids   == [1, 2, 3]
        assert result.book_titles == ["Book_0", "Book_1", "Book_2"]

    def test_history_populated(self):
        opt    = self._make_optimizer(n=3)
        result = opt.optimize()
        assert len(result.history) == 15   # generations=15 in _make_optimizer

    def test_elapsed_positive(self):
        opt    = self._make_optimizer(n=3)
        result = opt.optimize()
        assert result.elapsed_s >= 0.0

    def test_last_result_set_after_optimize(self):
        opt = self._make_optimizer(n=3)
        opt.optimize()
        assert opt.last_result is not None

    def test_cost_breakdown_before_keys(self):
        opt    = self._make_optimizer(n=3)
        result = opt.optimize()
        assert "total_cost" in result.cost_breakdown_before

    def test_cost_breakdown_after_keys(self):
        opt    = self._make_optimizer(n=3)
        result = opt.optimize()
        assert "total_cost" in result.cost_breakdown_after


# ══════════════════════════════════════════════════════════════════════════════
#  V. PlacementOptimizer.from_workflow_analyzer
# ══════════════════════════════════════════════════════════════════════════════

class TestFromWorkflowAnalyzer:

    def _make_mock_analyzer(self):
        """Return a minimal WorkflowAnalyzer mock with 3 books."""
        from workflow.workflow_analyzer import WorkflowAnalyzer, BookMetrics
        wa = WorkflowAnalyzer(db_manager=None)
        # Manually inject 3 books
        wa._books = {1: "ML", 2: "Stats", 3: "Algorithms"}
        # Add some events to populate metrics
        from datetime import datetime, timezone
        from workflow.workflow_analyzer import InteractionEvent
        now = datetime.now(timezone.utc)
        for bid, title in wa._books.items():
            for _ in range(bid * 2):
                ev = InteractionEvent(
                    book_id=bid, book_title=title,
                    interaction="pickup", duration_s=10.0
                )
                wa._events.append(ev)
        # Add transitions
        wa._transitions[1][2] = 5
        wa._transitions[2][3] = 3
        return wa

    def test_builds_correctly(self):
        from optimization.placement_optimizer import PlacementOptimizer
        wa = self._make_mock_analyzer()
        book_positions = {1: (0.3, 0.3), 2: (0.7, 0.5), 3: (1.0, 0.6)}
        opt = PlacementOptimizer.from_workflow_analyzer(
            book_positions   = book_positions,
            workflow_analyzer = wa,
            user_position    = USER_POS,
            workspace        = WORKSPACE,
        )
        assert opt.n_books == 3

    def test_skips_books_without_positions(self):
        from optimization.placement_optimizer import PlacementOptimizer
        wa = self._make_mock_analyzer()
        # Only provide position for 2 of the 3 books
        book_positions = {1: (0.3, 0.3), 2: (0.7, 0.5)}
        opt = PlacementOptimizer.from_workflow_analyzer(
            book_positions   = book_positions,
            workflow_analyzer = wa,
            user_position    = USER_POS,
            workspace        = WORKSPACE,
        )
        assert opt.n_books == 2

    def test_raises_when_no_matching_positions(self):
        from optimization.placement_optimizer import PlacementOptimizer
        wa = self._make_mock_analyzer()
        with pytest.raises(ValueError, match="No books with known positions"):
            PlacementOptimizer.from_workflow_analyzer(
                book_positions   = {},   # empty
                workflow_analyzer = wa,
                user_position    = USER_POS,
                workspace        = WORKSPACE,
            )


# ══════════════════════════════════════════════════════════════════════════════
#  W. End-to-end integration (5-book scenario)
# ══════════════════════════════════════════════════════════════════════════════

class TestEndToEndIntegration:
    """
    5-book desk scenario.

    Books start in a poor arrangement (far from user, heavily overlapping).
    The GA should produce a better arrangement.
    """

    @pytest.mark.integration
    def test_five_book_cost_decreases(self):
        from optimization.placement_optimizer import PlacementOptimizer, BookInfo
        from optimization.fitness_function import FitnessConfig
        from optimization.genetic_algorithm import GAConfig

        # Poor starting positions: far from user, clustered at top-right
        books = [
            BookInfo(1, "Machine Learning",    (1.10, 0.75), usage_freq=10.0),
            BookInfo(2, "Deep Learning",        (1.12, 0.75), usage_freq=8.0),
            BookInfo(3, "Statistics",           (1.11, 0.74), usage_freq=6.0),
            BookInfo(4, "Linear Algebra",       (1.09, 0.76), usage_freq=4.0),
            BookInfo(5, "Calculus",             (1.10, 0.73), usage_freq=2.0),
        ]

        # Workflow: ML -> DL strong, DL -> Stats moderate
        wf = np.array([
            [0.0, 0.7, 0.1, 0.1, 0.1],
            [0.3, 0.0, 0.5, 0.1, 0.1],
            [0.1, 0.2, 0.0, 0.4, 0.3],
            [0.1, 0.1, 0.3, 0.0, 0.5],
            [0.1, 0.1, 0.2, 0.3, 0.0],
        ])

        obstacles = [
            (0.40, 0.35, 0.60, 0.55),   # laptop obstacle in centre
        ]

        ga_cfg = GAConfig(
            population_size = 60,
            generations     = 80,
            crossover_prob  = 0.80,
            mutation_prob   = 0.20,
            elite_count     = 3,
            seed            = 42,
        )

        opt = PlacementOptimizer(
            books           = books,
            workflow_matrix = wf,
            user_position   = USER_POS,
            workspace       = WORKSPACE,
            obstacles       = obstacles,
            ga_config       = ga_cfg,
        )

        result = opt.optimize(verbose=False)

        # cost_before is the ACTUAL cost of the clustered, far-away arrangement
        assert result.cost_before > 0.0, "Cost before must be positive"

        # The GA must find a strictly better arrangement
        assert result.cost_after  <= result.cost_before, (
            f"Optimized cost ({result.cost_after:.4f}) > "
            f"current cost ({result.cost_before:.4f})"
        )

        # Improvement is honest
        assert result.improvement == pytest.approx(
            result.cost_before - result.cost_after, abs=1e-9
        )
        assert result.improvement >= 0.0

        # All 5 books have positions
        assert len(result.optimized_arrangement) == 5

    @pytest.mark.integration
    def test_summary_output_is_complete(self):
        from optimization.placement_optimizer import PlacementOptimizer, BookInfo
        from optimization.genetic_algorithm import GAConfig

        books = [
            BookInfo(1, "Book A", (1.1, 0.7), 5.0),
            BookInfo(2, "Book B", (1.1, 0.7), 3.0),
        ]
        ga_cfg = GAConfig(population_size=10, generations=5, seed=1)
        opt = PlacementOptimizer(
            books           = books,
            workflow_matrix = _zero_matrix(2),
            user_position   = USER_POS,
            workspace       = WORKSPACE,
            ga_config       = ga_cfg,
        )
        result = opt.optimize()
        summary = result.summary()
        assert "CURRENT ARRANGEMENT"   in summary
        assert "OPTIMIZED ARRANGEMENT" in summary
        assert "Cost Before"           in summary
        assert "Cost After"            in summary
        assert "Improvement"           in summary
        assert "Book A"                in summary
        assert "Book B"                in summary


# ══════════════════════════════════════════════════════════════════════════════
#  X. Public imports (optimization namespace)
# ══════════════════════════════════════════════════════════════════════════════

class TestPublicImports:

    def test_fitness_function_import(self):
        from optimization import FitnessFunction, FitnessConfig
        assert FitnessFunction is not None
        assert FitnessConfig   is not None

    def test_genetic_algorithm_import(self):
        from optimization import GeneticAlgorithm, GAConfig, GenerationStats
        assert GeneticAlgorithm is not None
        assert GAConfig          is not None
        assert GenerationStats   is not None

    def test_placement_optimizer_import(self):
        from optimization import PlacementOptimizer, BookInfo, OptimizationResult
        assert PlacementOptimizer  is not None
        assert BookInfo            is not None
        assert OptimizationResult  is not None

    def test_type_aliases_import(self):
        from optimization import Chromosome, Position
        assert Chromosome is not None
        assert Position    is not None

    def test_all_exports_defined(self):
        import optimization
        for name in optimization.__all__:
            assert hasattr(optimization, name), f"Missing export: {name}"
