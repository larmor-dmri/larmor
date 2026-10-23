# Plain numpy twin of the counter-based random numbers of tracking
# (cpp/src/tracking/rng.cpp): SplitMix64's finaliser keyed by the run seed and the
# seed's global index. The same numbers as the C++ ones, so that numpy references
# and checks can predict any draw.
import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

GAMMA = np.uint64(0x9E3779B97F4A7C15)
M1 = np.uint64(0xBF58476D1CE4E5B9)
M2 = np.uint64(0x94D049BB133111EB)
TO_UNIT = 2**-53
# The first-direction search draws from this counter on, the steps from 0.
SEARCH_COUNTER = 2**63


def mix(z: NDArray[np.uint64]) -> NDArray[np.uint64]:
    with np.errstate(over="ignore"):
        z = (z ^ (z >> np.uint64(30))) * M1
        z = (z ^ (z >> np.uint64(27))) * M2
        return z ^ (z >> np.uint64(31))


def uniform_array(seed: int, indices: ArrayLike, ks: ArrayLike) -> NDArray[np.float64]:
    """Draw ks of the streamlines with these indices, broadcast together."""
    u_seed = np.asarray(seed, dtype=np.uint64)
    idx = np.asarray(indices, dtype=np.uint64)
    k = np.asarray(ks, dtype=np.uint64)
    with np.errstate(over="ignore"):
        keys = mix(mix(u_seed) + idx)
        x = mix(keys + (k + np.uint64(1)) * GAMMA)
    out: NDArray[np.float64] = (x >> np.uint64(11)).astype(np.float64) * TO_UNIT
    return out


def stream(seed: int, index: int) -> int:
    """The key of the streamline with this index."""
    with np.errstate(over="ignore"):
        u = np.asarray(seed, dtype=np.uint64)
        return int(mix(mix(u) + np.asarray(index, dtype=np.uint64)))


def uniform(key: int, k: int) -> float:
    """Draw k of the streamline with this key."""
    with np.errstate(over="ignore"):
        u_key = np.asarray(key, dtype=np.uint64)
        u_k = np.asarray(k, dtype=np.uint64)
        x = mix(u_key + (u_k + np.uint64(1)) * GAMMA)
    return float(x >> np.uint64(11)) * TO_UNIT


def draw_vector(state: NDArray[np.uint64]) -> NDArray[np.float64]:
    """A random unit vector from [key, counter]; advances the counter by two."""
    key, k = int(state[0]), int(state[1])
    u = uniform(key, k)
    v = uniform(key, (k + 1) % 2**64)
    state[1] = (k + 2) % 2**64
    z = 2 * u - 1
    phi = 2 * math.pi * v
    r = math.sqrt(1 - z * z)
    return np.array([r * math.cos(phi), r * math.sin(phi), z])


def search_vector(run_seed: int, index: int) -> NDArray[np.float64]:
    """The direction a seed's first-direction search draws first."""
    state = np.array([stream(run_seed, index), SEARCH_COUNTER], dtype=np.uint64)
    return draw_vector(state)
