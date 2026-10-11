"""
optimization/placement_optimizer.py
=====================================
Phase 8 — High-level PlacementOptimizer facade.

Orchestrates the full optimization pipeline:

    1. Accepts book positions, usage frequency, workflow matrix,
       user position, workspace bounds, and obstacle map.
    2. Evaluates the ACTUAL cost of the current arrangement.
    3. Runs the Genetic Algorithm to find the optimal arrangement.
    4. Evaluates the ACTUAL cost of the optimized arrangement.
    5. Returns a rich OptimizationResult with:
         - Current arrangement (chromosome)
         - Optimized arrangement (chromosome)
         - Cost before optimization  (actual)
         - Cost after optimization   (actual)
         - Optimization improvement  (actual difference, not fabricated %)
         - Per-generation history
         - Per-book position mapping

The cost_before and cost_after are computed by the SAME FitnessFunction,
ensuring the improvement figure is honest and directly comparable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from optimization.fitness_function import (
    FitnessConfig,
    FitnessFunction,
    Chromosome,
    Position,
)
from optimization.genetic_algorithm import GAConfig, GeneticAlgorithm, GenerationStats


# ---------------------------------------------------------------------------
# BookInfo
# ---------------------------------------------------------------------------

@dataclass
class BookInfo:
    """
    Describes one book for the optimizer.

    Attributes
    ----------
    book_id      : Unique integer identifier.
    title        : Human-readable title.
    position     : Current (x, y) position in metres.
    usage_freq   : Historical interaction count.
    """
    book_id:    int
    title:      str
    position:   Position
    usage_freq: float = 1.0


# ---------------------------------------------------------------------------
# OptimizationResult
# ---------------------------------------------------------------------------

@dataclass
class OptimizationResult:
    """
    Complete result of one optimization run.

    Attributes
    ----------
    current_arrangement   : List of current (x,y) positions (input order).
    optimized_arrangement : List of optimal (x,y) positions (same order).
    cost_before           : Fitness cost of the CURRENT arrangement.
    cost_after            : Fitness cost of the OPTIMIZED arrangement.
    improvement           : cost_before - cost_after  (positive = better).
    improvement_pct       : 100 * improvement / cost_before
                            (None if cost_before == 0).
    book_titles           : Ordered list of book titles (matches arrays above).
    book_ids              : Ordered list of book IDs.
    history               : Per-generation statistics list.
    elapsed_s             : Wall-clock seconds for the GA run.
    cost_breakdown_before : Component costs for the current arrangement.
    cost_breakdown_after  : Component costs for the optimized arrangement.
    """
    current_arrangement:   List[Position]
    optimized_arrangement: List[Position]
    cost_before:           float
    cost_after:            float
    improvement:           float
    improvement_pct:       Optional[float]
    book_titles:           List[str]
    book_ids:              List[int]
    history:               List[GenerationStats] = field(default_factory=list)
    elapsed_s:             float = 0.0
    cost_breakdown_before: Dict[str, float] = field(default_factory=dict)
    cost_breakdown_after:  Dict[str, float] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable summary string."""
        lines = [
            "=" * 60,
            "  WORKSPACE PLACEMENT OPTIMIZATION — RESULT",
            "=" * 60,
            "",
            "  CURRENT ARRANGEMENT:",
        ]
        for i, (title, pos) in enumerate(
            zip(self.book_titles, self.current_arrangement)
        ):
            lines.append(f"    [{i}] {title:<30s}  pos=({pos[0]:.3f}, {pos[1]:.3f})")

        lines += [
            "",
            "  OPTIMIZED ARRANGEMENT:",
        ]
        for i, (title, pos) in enumerate(
            zip(self.book_titles, self.optimized_arrangement)
        ):
            lines.append(f"    [{i}] {title:<30s}  pos=({pos[0]:.3f}, {pos[1]:.3f})")

        lines += [
            "",
            f"  Cost Before Optimization : {self.cost_before:.6f}",
            f"  Cost After  Optimization : {self.cost_after:.6f}",
            f"  Improvement (absolute)   : {self.improvement:+.6f}",
        ]
        if self.improvement_pct is not None:
            lines.append(
                f"  Improvement (%)          : {self.improvement_pct:+.2f}%"
            )
        lines += [
            f"  GA Elapsed Time          : {self.elapsed_s:.2f}s",
            "",
            "  COST BREAKDOWN — CURRENT:",
        ]
        for k, v in self.cost_breakdown_before.items():
            lines.append(f"    {k:<25s}: {v:.6f}")
        lines += [
            "",
            "  COST BREAKDOWN — OPTIMIZED:",
        ]
        for k, v in self.cost_breakdown_after.items():
            lines.append(f"    {k:<25s}: {v:.6f}")
        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Return a serialisable dictionary."""
        return {
            "cost_before":       self.cost_before,
            "cost_after":        self.cost_after,
            "improvement":       self.improvement,
            "improvement_pct":   self.improvement_pct,
            "elapsed_s":         self.elapsed_s,
            "book_titles":       self.book_titles,
            "book_ids":          self.book_ids,
            "current_positions": [list(p) for p in self.current_arrangement],
            "optimal_positions": [list(p) for p in self.optimized_arrangement],
            "cost_breakdown_before": self.cost_breakdown_before,
            "cost_breakdown_after":  self.cost_breakdown_after,
            "generations_run":   len(self.history),
        }


# ---------------------------------------------------------------------------
# PlacementOptimizer
# ---------------------------------------------------------------------------

class PlacementOptimizer:
    """
    High-level facade that wires FitnessFunction + GeneticAlgorithm.

    Usage
    -----
    ::

        optimizer = PlacementOptimizer(
            books           = book_list,          # List[BookInfo]
            workflow_matrix = wf_matrix,          # np.ndarray [n x n]
            user_position   = (0.60, 0.0),        # robot base
            workspace       = (0.0, 0.0, 1.20, 0.80),
            obstacles       = [(0.5, 0.3, 0.7, 0.5)],
            fitness_config  = FitnessConfig(),
            ga_config       = GAConfig(),
        )
        result = optimizer.optimize(verbose=True)
        print(result.summary())

    Parameters
    ----------
    books           : Ordered list of BookInfo objects.
    workflow_matrix : [n x n] transition probability matrix.
    user_position   : (x, y) user / robot base position in metres.
    workspace       : (x_min, y_min, x_max, y_max) boundary.
    obstacles       : List of obstacle bounding boxes (x_min, y_min, x_max, y_max).
    fitness_config  : FitnessConfig (tunable weights).
    ga_config       : GAConfig (GA hyper-parameters).
    """

    def __init__(
        self,
        books:           List[BookInfo],
        workflow_matrix: np.ndarray,
        user_position:   Position,
        workspace:       Tuple[float, float, float, float],
        obstacles:       Optional[List[Tuple[float, float, float, float]]] = None,
        fitness_config:  Optional[FitnessConfig] = None,
        ga_config:       Optional[GAConfig] = None,
    ) -> None:
        if not books:
            raise ValueError("books list must not be empty.")

        self.books    = books
        self.n_books  = len(books)
        self.workspace = workspace

        usage_freq = [b.usage_freq for b in books]

        self.fitness_fn = FitnessFunction(
            usage_freq      = usage_freq,
            workflow_matrix = workflow_matrix,
            user_position   = user_position,
            workspace       = workspace,
            obstacles       = obstacles or [],
            config          = fitness_config,
        )

        self.ga_cfg = ga_config or GAConfig()

        self._last_result: Optional[OptimizationResult] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize(self, verbose: bool = False) -> OptimizationResult:
        """
        Run the optimization.

        1. Evaluate actual cost of the current arrangement.
        2. Run GA starting with the current arrangement as a seed.
        3. Evaluate actual cost of the best arrangement found.
        4. Build and return OptimizationResult.

        Parameters
        ----------
        verbose : Print per-generation progress to stdout.

        Returns
        -------
        OptimizationResult
        """
        # Current chromosome (actual book positions)
        current_chromosome: Chromosome = [b.position for b in self.books]

        # Actual cost BEFORE
        cost_before = self.fitness_fn.evaluate(current_chromosome)
        breakdown_before = self.fitness_fn.breakdown(current_chromosome)

        if verbose:
            print(f"\n[PlacementOptimizer] Cost BEFORE optimization: {cost_before:.6f}")
            print(f"[PlacementOptimizer] Running GA: "
                  f"pop={self.ga_cfg.population_size}, "
                  f"gen={self.ga_cfg.generations} ...")

        # Build GA
        ga = GeneticAlgorithm(
            fitness_fn = self.fitness_fn,
            workspace  = self.workspace,
            n_books    = self.n_books,
            config     = self.ga_cfg,
        )

        # Seed the GA with the current arrangement so it always has at
        # least one valid starting point.
        ga.initialise(seed_chromosomes=[current_chromosome])

        # Evolve
        t_start = time.perf_counter()
        best_chromosome = ga.evolve(verbose=verbose)
        elapsed = time.perf_counter() - t_start

        # Actual cost AFTER
        cost_after    = self.fitness_fn.evaluate(best_chromosome)
        breakdown_after = self.fitness_fn.breakdown(best_chromosome)

        improvement = cost_before - cost_after
        if cost_before > 0:
            improvement_pct = 100.0 * improvement / cost_before
        else:
            improvement_pct = None

        if verbose:
            print(f"[PlacementOptimizer] Cost AFTER  optimization: {cost_after:.6f}")
            if improvement_pct is not None:
                print(f"[PlacementOptimizer] Improvement: {improvement:+.6f} "
                      f"({improvement_pct:+.2f}%)")
            print(f"[PlacementOptimizer] GA elapsed: {elapsed:.2f}s")

        result = OptimizationResult(
            current_arrangement   = current_chromosome,
            optimized_arrangement = best_chromosome,
            cost_before           = cost_before,
            cost_after            = cost_after,
            improvement           = improvement,
            improvement_pct       = improvement_pct,
            book_titles           = [b.title for b in self.books],
            book_ids              = [b.book_id for b in self.books],
            history               = ga.history,
            elapsed_s             = elapsed,
            cost_breakdown_before = breakdown_before,
            cost_breakdown_after  = breakdown_after,
        )

        self._last_result = result
        return result

    @property
    def last_result(self) -> Optional[OptimizationResult]:
        """Return the result of the most recent optimization run."""
        return self._last_result

    # ------------------------------------------------------------------
    # Convenience: build from WorkflowAnalyzer output
    # ------------------------------------------------------------------

    @classmethod
    def from_workflow_analyzer(
        cls,
        book_positions: Dict[int, Position],
        workflow_analyzer,
        user_position:   Position,
        workspace:       Tuple[float, float, float, float],
        obstacles:       Optional[List[Tuple[float, float, float, float]]] = None,
        fitness_config:  Optional[FitnessConfig] = None,
        ga_config:       Optional[GAConfig] = None,
    ) -> "PlacementOptimizer":
        """
        Construct a PlacementOptimizer directly from a WorkflowAnalyzer.

        Parameters
        ----------
        book_positions   : {book_id: (x, y)} current positions.
        workflow_analyzer: WorkflowAnalyzer instance (Phase 7).
        user_position    : (x, y) robot / user base.
        workspace        : (x_min, y_min, x_max, y_max).
        obstacles        : Obstacle bounding boxes.
        fitness_config   : FitnessConfig.
        ga_config        : GAConfig.
        """
        # Pull metrics from the analyzer
        metrics = workflow_analyzer.compute_book_metrics()

        # Only consider books that have a known position
        books: List[BookInfo] = []
        book_id_order: List[int] = []
        for m in metrics:
            if m.book_id in book_positions:
                books.append(
                    BookInfo(
                        book_id    = m.book_id,
                        title      = m.book_title,
                        position   = book_positions[m.book_id],
                        usage_freq = float(m.usage_frequency),
                    )
                )
                book_id_order.append(m.book_id)

        if not books:
            raise ValueError(
                "No books with known positions found. "
                "Provide positions in book_positions dict."
            )

        # Build the workflow (transition probability) matrix
        n = len(books)
        wf_matrix = np.zeros((n, n), dtype=float)
        transitions = workflow_analyzer._transitions   # dict-of-dicts

        id_to_idx = {bid: i for i, bid in enumerate(book_id_order)}
        for from_id, to_counts in transitions.items():
            if from_id not in id_to_idx:
                continue
            fi = id_to_idx[from_id]
            total = max(sum(to_counts.values()), 1)
            for to_id, cnt in to_counts.items():
                if to_id in id_to_idx:
                    wf_matrix[fi][id_to_idx[to_id]] = cnt / total

        return cls(
            books           = books,
            workflow_matrix = wf_matrix,
            user_position   = user_position,
            workspace       = workspace,
            obstacles       = obstacles,
            fitness_config  = fitness_config,
            ga_config       = ga_config,
        )
