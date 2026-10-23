"""Lists the GPUs wgpu can draw with, and chooses the one the viewer uses."""

import pygfx as gfx
import wgpu

from larmor._log import LarmorLogger


def gpu_names() -> list[str]:
    """Returns the summary of each GPU wgpu can use; identical cards get a number."""
    adapters = _adapters()
    return _labels([a.summary for a in adapters]) if adapters is not None else []


def select_gpu(name: str) -> bool:
    """Makes wgpu draw with the GPU of this name; False if there is none.

    Takes effect only before the first view is drawn; "" leaves wgpu's choice.

    Returns:
        Whether a GPU of that name was found.
    """
    if not name:
        return True
    adapters = _adapters()
    if adapters is None:
        return False
    labels = _labels([a.summary for a in adapters])
    if name not in labels:
        LarmorLogger.warning("The GPU %s is not available; using the default.", name)
        return False
    try:
        gfx.renderers.wgpu.select_adapter(adapters[labels.index(name)])
    except Exception as error:  # the GPU is already in use
        LarmorLogger.warning("Cannot switch to the GPU %s: %s", name, error)
        return False
    return True


def _adapters() -> list[wgpu.GPUAdapter] | None:
    """Returns the GPUs wgpu can use, or None if they cannot be listed."""
    try:
        return list(wgpu.gpu.enumerate_adapters_sync())
    except Exception as error:  # no driver: nothing to choose from
        LarmorLogger.warning("Cannot list the GPUs: %s", error)
        return None


def _labels(summaries: list[str]) -> list[str]:
    """Returns the summaries, adding "(GPU n)" where several cards share one."""
    labels = []
    for i, summary in enumerate(summaries):
        if summaries.count(summary) > 1:
            summary += f" (GPU {summaries[: i + 1].count(summary)})"
        labels.append(summary)
    return labels
