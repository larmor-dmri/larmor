"""Array helpers shared by the types."""

import numpy as np


def read_only(array: np.ndarray) -> np.ndarray:
    """Returns a read-only view of `array`, leaving the caller's array writable."""
    view = array.view()
    view.flags.writeable = False
    return view


def check_data(
    data: dict[str, np.ndarray], label: str, count: int
) -> dict[str, np.ndarray]:
    """Returns `data` with each value as an array, checking its first axis.

    Args:
        data: Named values, each with `count` entries along its first axis.
        label: The name of `data` in the error message.
        count: The number of entries expected.

    Raises:
        ValueError: If a value is a scalar or its first axis does not have `count`
            entries.
    """
    checked = {}
    for name, values in data.items():
        array = np.asarray(values)
        if array.ndim == 0 or array.shape[0] != count:
            raise ValueError(
                f"{label}[{name!r}] has shape {array.shape}, expected {count} entries "
                "along its first axis."
            )
        checked[name] = array
    return checked
