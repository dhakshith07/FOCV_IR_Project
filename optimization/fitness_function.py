"""
optimization/fitness_function.py
=================================
Phase 8 — Fitness function for Genetic Algorithm workspace placement.

A chromosome is a list of (x, y) positions, one per book, in the same
order as the book list provided to the optimizer.

Cost components (all minimised — lower = better)
-------------------------------------------------
1. retrieval_distance  : Euclidean distance from user to each book,
                         weighted by usage frequency.
2. workflow_distance   : Distance between books that are used together
                         (workflow relationship matrix), weighted by
                         transition probability.
3. movement_cost       : Total path-length to visit books in their most
                         likely sequential order.
4. constraint_penalty  : Penalty for positions outside workspace bounds.
5. overlap_penalty     : Penalty for books placed too close together
                         (collision proxy).
6. obstacle_penalty    : Penalty for positions that overlap any obstacle
                         region.

Total fitness (cost) = weighted sum of the six components.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Position = Tuple[float, float]       # (x, y) in meters
Chromosome = List[Position]          # one position per book


# ---------------------------------------------------------------------------
# FitnessConfig
# ---------------------------------------------------------------------------

class FitnessConfig:
    """
    Tunable parameters for the fitness function.

    Attributes
    ----------
    w_retrieval   : Weight for retrieval-distance cost.
    w_workflow    : Weight for workflow-distance cost.
    w_movement    : Weight for movement-cost.
    w_constraint  : Penalty multiplier for out-of-bounds positions.
    w_overlap     : Penalty multiplier for book-book overlaps.
    w_obstacle    : Penalty multiplier for obstacle intersections.
    book_radius_m : Approximate book footprint radius (metres) used for
                    overlap detection.
    """

    def __init__(
        self,
        w_retrieval:  float = 1.0,
        w_workflow:   float = 1.5,
        w_movement:   float = 0.8,
        w_constraint: float = 10.0,
        w_overlap:    float = 8.0,
        w_obstacle:   float = 10.0,
        book_radius_m: float = 0.10,
    ) -> None:
        self.w_retrieval   = w_retrieval
        self.w_workflow    = w_workflow
        self.w_movement    = w_movement
        self.w_constraint  = w_constraint
        self.w_overlap     = w_overlap
        self.w_obstacle    = w_obstacle
        self.book_radius_m = book_radius_m


# ---------------------------------------------------------------------------
# FitnessFunction
# ---------------------------------------------------------------------------

class FitnessFunction:
    """
    Stateless evaluator: maps a Chromosome to scalar cost (lower = better).

    Parameters
    ----------
    usage_freq        : List[float] -- interaction count per book (same order
                        as chromosome indices).
    workflow_matrix   : 2-D array [n_books x n_books] -- P(book_j used after
                        book_i).  Values in [0, 1].
    user_position     : (x, y) position of the user / robot base in metres.
    workspace         : (x_min, y_min, x_max, y_max) workspace boundary.
    obstacles         : List of (x_min, y_min, x_max, y_max) obstacle boxes.
    config            : FitnessConfig instance (uses defaults if None).
    """

    def __init__(
        self,
        usage_freq:      Sequence[float],
        workflow_matrix: np.ndarray,
        user_position:   Position,
        workspace:       Tuple[float, float, float, float],
        obstacles:       Optional[List[Tuple[float, float, float, float]]] = None,
        config:          Optional[FitnessConfig] = None,
    ) -> None:
        self.usage_freq      = list(usage_freq)
        self.workflow_matrix = np.asarray(workflow_matrix, dtype=float)
        self.user_position   = user_position
        self.workspace       = workspace          # (x_min, y_min, x_max, y_max)
        self.obstacles       = obstacles or []
        self.cfg             = config or FitnessConfig()

        n = len(self.usage_freq)
        if self.workflow_matrix.shape != (n, n):
            raise ValueError(
                f"workflow_matrix shape {self.workflow_matrix.shape} "
                f"does not match n_books={n}"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, chromosome: Chromosome) -> float:
        """
        Compute the total placement cost for one chromosome.

        Returns
        -------
        float -- total cost.  Lower = better arrangement.
        """
        n = len(chromosome)
        if n != len(self.usage_freq):
            raise ValueError(
                f"Chromosome length {n} != n_books {len(self.usage_freq)}"
            )

        cost = (
            self.cfg.w_retrieval  * self._retrieval_cost(chromosome)
            + self.cfg.w_workflow * self._workflow_cost(chromosome)
            + self.cfg.w_movement * self._movement_cost(chromosome)
            + self.cfg.w_constraint * self._constraint_penalty(chromosome)
            + self.cfg.w_overlap  * self._overlap_penalty(chromosome)
            + self.cfg.w_obstacle * self._obstacle_penalty(chromosome)
        )
        return float(cost)

    # ------------------------------------------------------------------
    # Component 1 -- retrieval distance
    # ------------------------------------------------------------------

    def _retrieval_cost(self, chromosome: Chromosome) -> float:
        """
        Weighted sum of distances from user to each book.

        C1 = sum_i  freq_i * dist(user, pos_i)

        Normalise freq to [0, 1] so cost is scale-independent.
        """
        freq = self.usage_freq
        max_freq = max(freq) if max(freq) > 0 else 1.0
        ux, uy = self.user_position

        total = 0.0
        for i, (bx, by) in enumerate(chromosome):
            w = freq[i] / max_freq
            total += w * math.hypot(bx - ux, by - uy)
        return total

    # ------------------------------------------------------------------
    # Component 2 -- workflow distance
    # ------------------------------------------------------------------

    def _workflow_cost(self, chromosome: Chromosome) -> float:
        """
        Penalise arrangements where frequently co-used books are far apart.

        C2 = sum_{i!=j}  W[i,j] * dist(pos_i, pos_j)

        W[i,j] = workflow_matrix[i,j]  (transition probability i->j).
        """
        n = len(chromosome)
        total = 0.0
        wf = self.workflow_matrix
        for i in range(n):
            xi, yi = chromosome[i]
            for j in range(n):
                if i == j:
                    continue
                xj, yj = chromosome[j]
                total += wf[i, j] * math.hypot(xi - xj, yi - yj)
        return total

    # ------------------------------------------------------------------
    # Component 3 -- movement cost
    # ------------------------------------------------------------------

    def _movement_cost(self, chromosome: Chromosome) -> float:
        """
        Approximate cost of visiting books in priority order.

        Sorts books by usage frequency (descending) and computes the total
        path length starting from the user position through each book in
        that ranked order.

        C3 = sum_{k=0}^{n-1}  dist(visit_k, visit_{k+1})
        """
        if not chromosome:
            return 0.0

        # Sort indices by frequency descending
        order = sorted(
            range(len(chromosome)),
            key=lambda i: -self.usage_freq[i],
        )

        ux, uy = self.user_position
        prev_x, prev_y = ux, uy
        total = 0.0
        for idx in order:
            bx, by = chromosome[idx]
            total += math.hypot(bx - prev_x, by - prev_y)
            prev_x, prev_y = bx, by
        return total

    # ------------------------------------------------------------------
    # Component 4 -- workspace constraint penalty
    # ------------------------------------------------------------------

    def _constraint_penalty(self, chromosome: Chromosome) -> float:
        """
        Penalise books placed outside the workspace boundary.

        Penalty = Euclidean distance beyond the boundary (clamped to 0).
        """
        x_min, y_min, x_max, y_max = self.workspace
        total = 0.0
        for bx, by in chromosome:
            dx = max(0.0, x_min - bx, bx - x_max)
            dy = max(0.0, y_min - by, by - y_max)
            total += math.hypot(dx, dy)
        return total

    # ------------------------------------------------------------------
    # Component 5 -- book overlap penalty
    # ------------------------------------------------------------------

    def _overlap_penalty(self, chromosome: Chromosome) -> float:
        """
        Penalise pairs of books whose centres are closer than 2 * book_radius.

        Penalty per pair = max(0, 2*r - dist(i, j)).
        """
        n = len(chromosome)
        min_dist = 2.0 * self.cfg.book_radius_m
        total = 0.0
        for i in range(n):
            xi, yi = chromosome[i]
            for j in range(i + 1, n):
                xj, yj = chromosome[j]
                d = math.hypot(xi - xj, yi - yj)
                overlap = max(0.0, min_dist - d)
                total += overlap
        return total

    # ------------------------------------------------------------------
    # Component 6 -- obstacle penalty
    # ------------------------------------------------------------------

    def _obstacle_penalty(self, chromosome: Chromosome) -> float:
        """
        Penalise books placed inside or overlapping obstacle bounding boxes.

        For each (book, obstacle) pair the penalty is the area of overlap
        between the book's circular footprint (approximated as a square of
        side 2*r) and the obstacle rectangle.
        """
        if not self.obstacles:
            return 0.0

        r = self.cfg.book_radius_m
        total = 0.0
        for bx, by in chromosome:
            # Book bounding box
            b_x1 = bx - r
            b_y1 = by - r
            b_x2 = bx + r
            b_y2 = by + r

            for (ox1, oy1, ox2, oy2) in self.obstacles:
                # Intersection
                ix1 = max(b_x1, ox1)
                iy1 = max(b_y1, oy1)
                ix2 = min(b_x2, ox2)
                iy2 = min(b_y2, oy2)
                w = max(0.0, ix2 - ix1)
                h = max(0.0, iy2 - iy1)
                total += w * h
        return total

    # ------------------------------------------------------------------
    # Cost breakdown (diagnostic)
    # ------------------------------------------------------------------

    def breakdown(self, chromosome: Chromosome) -> Dict[str, float]:
        """
        Return the un-weighted individual cost components for diagnosis.
        """
        return {
            "retrieval_distance": self._retrieval_cost(chromosome),
            "workflow_distance":  self._workflow_cost(chromosome),
            "movement_cost":      self._movement_cost(chromosome),
            "constraint_penalty": self._constraint_penalty(chromosome),
            "overlap_penalty":    self._overlap_penalty(chromosome),
            "obstacle_penalty":   self._obstacle_penalty(chromosome),
            "total_cost":         self.evaluate(chromosome),
        }
