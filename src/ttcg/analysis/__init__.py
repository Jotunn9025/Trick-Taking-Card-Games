"""Statistical analysis of the paired-deal evaluation logs.

``statistics`` holds the dependency-light primitives; ``report`` includes the
plotting/CLI layer (matplotlib + seaborn) and is imported on demand.
"""

from . import statistics

__all__ = ["statistics"]