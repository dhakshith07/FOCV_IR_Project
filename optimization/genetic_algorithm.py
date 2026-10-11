"""
optimization/genetic_algorithm.py
===================================
Phase 8 — Genetic Algorithm engine for workspace book placement.

Chromosome representation
--------------------------
A chromosome is a list of (x, y) float tuples, one per book:

    [BookPosition_0, BookPosition_1, ..., BookPosition_{n-1}]

Each gene is an (x, y) coordinate (metres) on the workspace surface.

Evolution pipeline
------------------
    Population
        |
    Fitness evaluation (FitnessFunction)
        |
    Selection (tournament)
        |
    Crossover (uniform / blend)
        |
    Mutation  (Gaussian perturbation)
        |
    Elitism   (preserve top-k individuals)
        |
    New Generation  (repeat until max_generations)
        |
    Best Solution

Configuration is read from a dict / config.yaml genetic_algorithm section.
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from optimization.fitness_function import FitnessFunction, Chromosome, Position

# ---------------------------------------------------------------------------
# GAConfig
# ---------------------------------------------------------------------------

@dataclass
class GAConfig:
    """Hyper-parameters for the Genetic Algorithm."""

    population_size: int   = 100
    generations:     int   = 150
    crossover_prob:  float = 0.80
    mutation_prob:   float = 0.15
    tournament_size: int   = 5
    elite_count:     int   = 5
    mutation_sigma:  float = 0.05   # Gaussian std-dev for gene mutation (metres)
    seed:            Optional[int] = None

    @classmethod
    def from_dict(cls, d: dict) -> "GAConfig":
        """Build from a config dict (e.g. from config.yaml)."""
        return cls(
            population_size = int(d.get("population_size", 100)),
            generations     = int(d.get("generations", 150)),
            crossover_prob  = float(d.get("crossover_prob", 0.80)),
            mutation_prob   = float(d.get("mutation_prob", 0.15)),
            tournament_size = int(d.get("tournament_size", 5)),
            elite_count     = int(d.get("elite_count", 5)),
            mutation_sigma  = float(d.get("mutation_sigma", 0.05)),
            seed            = d.get("seed", None),
        )


# ---------------------------------------------------------------------------
# GenerationStats
# ---------------------------------------------------------------------------

@dataclass
class GenerationStats:
    """Statistics captured at the end of each generation."""
    generation:    int
    best_cost:     float
    mean_cost:     float
    worst_cost:    float
    improvement:   float   # cost reduction vs. previous generation

    @property
    def best_fitness(self) -> float:
        return self.best_cost

    @property
    def avg_fitness(self) -> float:
        return self.mean_cost


# ---------------------------------------------------------------------------
# GeneticAlgorithm
# ---------------------------------------------------------------------------

class GeneticAlgorithm:
    """
    Genetic Algorithm for workspace book placement.

    Parameters
    ----------
    fitness_fn  : FitnessFunction instance.
    workspace   : (x_min, y_min, x_max, y_max) for random initialisation.
    n_books     : Number of books to place.
    config      : GAConfig instance.
    """

    def __init__(
        self,
        fitness_fn: FitnessFunction,
        workspace:  Tuple[float, float, float, float],
        n_books:    int,
        config:     Optional[GAConfig] = None,
    ) -> None:
        self.fitness_fn  = fitness_fn
        self.workspace   = workspace
        self.n_books     = n_books
        self.cfg         = config or GAConfig()

        if self.cfg.seed is not None:
            random.seed(self.cfg.seed)
            np.random.seed(self.cfg.seed)

        # State updated during evolution
        self._population: List[Chromosome] = []
        self._costs:      List[float]      = []
        self._history:    List[GenerationStats] = []
        self._best_chromosome: Optional[Chromosome] = None
        self._best_cost:       float = float("inf")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def initialise(
        self,
        seed_chromosomes: Optional[List[Chromosome]] = None,
    ) -> None:
        """
        Create the initial population.

        Parameters
        ----------
        seed_chromosomes : Optional list of hand-crafted chromosomes to
                           include (e.g. the current arrangement).
                           Remaining slots are filled randomly.
        """
        x_min, y_min, x_max, y_max = self.workspace
        population: List[Chromosome] = []

        # Include seed chromosomes
        if seed_chromosomes:
            for ch in seed_chromosomes:
                if len(ch) == self.n_books:
                    population.append(copy.deepcopy(ch))

        # Fill the rest randomly
        while len(population) < self.cfg.population_size:
            chromosome: Chromosome = []
            for _ in range(self.n_books):
                x = random.uniform(x_min, x_max)
                y = random.uniform(y_min, y_max)
                chromosome.append((x, y))
            population.append(chromosome)

        self._population = population
        self._costs = [self.fitness_fn.evaluate(ch) for ch in self._population]
        self._update_best()

    def evolve(self, verbose: bool = False) -> Chromosome:
        """
        Run the full GA evolution loop.

        Parameters
        ----------
        verbose : If True, prints per-generation statistics.

        Returns
        -------
        Best chromosome found.
        """
        if not self._population:
            raise RuntimeError("Call initialise() before evolve().")

        self._history.clear()

        for gen in range(self.cfg.generations):
            prev_best = self._best_cost
            new_pop = self._next_generation()
            self._population = new_pop
            self._costs = [self.fitness_fn.evaluate(ch) for ch in self._population]
            self._update_best()

            # Record stats
            improvement = prev_best - self._best_cost
            stats = GenerationStats(
                generation  = gen + 1,
                best_cost   = self._best_cost,
                mean_cost   = float(np.mean(self._costs)),
                worst_cost  = float(np.max(self._costs)),
                improvement = improvement,
            )
            self._history.append(stats)

            if verbose:
                print(
                    f"  Gen {gen + 1:4d} | "
                    f"Best={self._best_cost:.4f} | "
                    f"Mean={stats.mean_cost:.4f} | "
                    f"Imprv={improvement:+.4f}"
                )

        return copy.deepcopy(self._best_chromosome)

    @property
    def best_cost(self) -> float:
        return self._best_cost

    @property
    def best_chromosome(self) -> Optional[Chromosome]:
        return copy.deepcopy(self._best_chromosome)

    @property
    def history(self) -> List[GenerationStats]:
        return list(self._history)

    # ------------------------------------------------------------------
    # Evolution operators
    # ------------------------------------------------------------------

    def _next_generation(self) -> List[Chromosome]:
        """Produce the next generation via elitism + selection + crossover + mutation."""
        cfg = self.cfg

        # Sort population by cost (ascending)
        sorted_pairs = sorted(
            zip(self._costs, self._population),
            key=lambda x: x[0],
        )

        # Elitism: carry top-k individuals unchanged
        new_pop: List[Chromosome] = [
            copy.deepcopy(ch) for _, ch in sorted_pairs[: cfg.elite_count]
        ]

        # Fill remaining slots
        while len(new_pop) < cfg.population_size:
            # Tournament selection
            p1 = self._tournament_select()
            p2 = self._tournament_select()

            # Crossover
            if random.random() < cfg.crossover_prob:
                c1, c2 = self._crossover(p1, p2)
            else:
                c1, c2 = copy.deepcopy(p1), copy.deepcopy(p2)

            # Mutation
            c1 = self._mutate(c1)
            c2 = self._mutate(c2)

            new_pop.append(c1)
            if len(new_pop) < cfg.population_size:
                new_pop.append(c2)

        return new_pop

    # ------------------------------------------------------------------
    # Tournament selection
    # ------------------------------------------------------------------

    def _tournament_select(self) -> Chromosome:
        """Select one chromosome via tournament selection."""
        k = min(self.cfg.tournament_size, len(self._population))
        indices = random.sample(range(len(self._population)), k)
        winner = min(indices, key=lambda i: self._costs[i])
        return copy.deepcopy(self._population[winner])

    # ------------------------------------------------------------------
    # Crossover — uniform blend (BLX-alpha)
    # ------------------------------------------------------------------

    def _crossover(
        self,
        parent1: Chromosome,
        parent2: Chromosome,
    ) -> Tuple[Chromosome, Chromosome]:
        """
        Uniform crossover at gene (position) level.

        Each gene (x, y) is independently swapped with probability 0.5.
        """
        n = self.n_books
        child1: Chromosome = []
        child2: Chromosome = []

        for i in range(n):
            if random.random() < 0.5:
                child1.append(parent1[i])
                child2.append(parent2[i])
            else:
                child1.append(parent2[i])
                child2.append(parent1[i])

        return child1, child2

    # ------------------------------------------------------------------
    # Mutation — Gaussian perturbation
    # ------------------------------------------------------------------

    def _mutate(self, chromosome: Chromosome) -> Chromosome:
        """
        Apply Gaussian mutation to each gene with probability mutation_prob.

        The mutated position is clamped to the workspace boundary.
        """
        x_min, y_min, x_max, y_max = self.workspace
        sigma = self.cfg.mutation_sigma
        mutated: Chromosome = []

        for (x, y) in chromosome:
            if random.random() < self.cfg.mutation_prob:
                x = x + random.gauss(0, sigma)
                y = y + random.gauss(0, sigma)
                # Clamp to workspace
                x = max(x_min, min(x_max, x))
                y = max(y_min, min(y_max, y))
            mutated.append((x, y))

        return mutated

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_best(self) -> None:
        """Update best chromosome / cost from current population."""
        for i, cost in enumerate(self._costs):
            if cost < self._best_cost:
                self._best_cost = cost
                self._best_chromosome = copy.deepcopy(self._population[i])
