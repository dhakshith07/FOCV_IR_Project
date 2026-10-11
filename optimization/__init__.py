"""
optimization/ -- Layout optimization algorithms.

Phase 8 exports
---------------
GeneticAlgorithm    : Core GA engine (population, selection, crossover, mutation).
GAConfig            : GA hyper-parameter dataclass.
GenerationStats     : Per-generation statistics dataclass.
FitnessFunction     : Multi-component fitness evaluator.
FitnessConfig       : Fitness weight / parameter dataclass.
PlacementOptimizer  : High-level facade (end-to-end optimization).
BookInfo            : Per-book input descriptor.
OptimizationResult  : Full result (costs, positions, history).
"""

from optimization.fitness_function import (
    FitnessFunction,
    FitnessConfig,
    Chromosome,
    Position,
)
from optimization.genetic_algorithm import (
    GeneticAlgorithm,
    GAConfig,
    GenerationStats,
)
from optimization.placement_optimizer import (
    PlacementOptimizer,
    BookInfo,
    OptimizationResult,
)

__all__ = [
    "FitnessFunction",
    "FitnessConfig",
    "Chromosome",
    "Position",
    "GeneticAlgorithm",
    "GAConfig",
    "GenerationStats",
    "PlacementOptimizer",
    "BookInfo",
    "OptimizationResult",
]
