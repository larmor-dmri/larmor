"""The number of threads a compiled call runs on."""

import os


def n_threads(n_jobs: int) -> int:
    """Returns the number of threads for n_jobs, at most the number of cores.

    Args:
        n_jobs: The number of threads asked for; -1 uses every core.

    Returns:
        The number of threads.

    Raises:
        ValueError: If n_jobs is 0 or below -1.
    """
    if n_jobs == 0 or n_jobs < -1:
        raise ValueError(f"n_jobs must be -1 (all) or at least 1, got {n_jobs}")
    cores = os.cpu_count() or 1
    return cores if n_jobs == -1 else min(n_jobs, cores)
