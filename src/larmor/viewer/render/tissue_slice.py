"""Defines a pygfx slice material that leaves out a magnitude image's background."""

import numpy as np
import pygfx as gfx
from pygfx.renderers.wgpu import load_wgsl, register_wgpu_render_function
from pygfx.renderers.wgpu.shaders.volumeshader import VolumeSliceShader

# Share of an image's 1st to 99th percentile range, above the 1st percentile, below
# which a voxel counts as background; it catches the air around a head, not the CSF.
CUTOFF = 0.04

# Smallest cut-off of an image without negative values, so its zeros are background.
NO_ZERO = float(np.finfo(np.float32).tiny)

# At most about this many voxels are read to find the percentiles.
_SAMPLES = 1_000_000

# The line of pygfx's slice shader after which a voxel's value is known, and the
# test inserted after it, which drops a value below the cut-off. Colours are tested
# by the largest of red, green and blue.
_SAMPLE = "let value = sample_vol(varyings.texcoord.xyz, sizef);"
_DISCARD = """
    $$ if img_nchannels == 1
        let shown = value.r {{ climcorrection }};
    $$ else
        let shown = max(max(value.r, value.g), value.b) {{ climcorrection }};
    $$ endif
    if (shown < u_material.cutoff) {
        discard;
    }
"""


class TissueSliceMaterial(gfx.VolumeSliceMaterial):  # type: ignore[misc]
    """Draws a volume slice with nothing where its value is below the cut-off."""

    uniform_type = dict(gfx.VolumeSliceMaterial.uniform_type, cutoff="f4")

    def __init__(self, **kwargs: object) -> None:
        """Creates a slice that leaves out nothing until it is given a cut-off."""
        super().__init__(**kwargs)
        self.cutoff = float("-inf")

    @property
    def cutoff(self) -> float:
        """Returns the value, in texture units, below which a voxel is left out."""
        return float(self.uniform_buffer.data["cutoff"])

    @cutoff.setter
    def cutoff(self, value: float) -> None:
        self.uniform_buffer.data["cutoff"] = value
        self.uniform_buffer.update_full()


def background_cutoff(values: np.ndarray) -> float:
    """Returns the value below which voxels are background, or -inf if it is signed."""
    flat = values.reshape(-1)
    sample = flat
    if flat.size > _SAMPLES:
        # The same random voxels each time, as a stride could align with an axis.
        sample = flat[np.random.default_rng(0).integers(0, flat.size, _SAMPLES)]
    low, high = np.nanpercentile(sample, [1.0, 99.0])
    # A signed image, such as a z-map, has no background. A 1st percentile a little
    # below zero is the ringing that resampling leaves in a magnitude image's air.
    if not low >= -CUTOFF * (high - low):
        return float("-inf")
    # An image of one value keeps it, as the cut-off is then that value.
    return max(float(low + CUTOFF * (high - low)), NO_ZERO)


@register_wgpu_render_function(gfx.Volume, TissueSliceMaterial)
class TissueSliceShader(VolumeSliceShader):  # type: ignore[misc]
    """Draws pygfx's slices without the background's voxels."""

    def get_code(self) -> str:
        """Returns the WGSL of pygfx's slices, with the test after the sampling."""
        code: str = load_wgsl("volume_slice.wgsl")
        if _SAMPLE not in code:
            raise RuntimeError(
                "pygfx's volume_slice.wgsl no longer contains "
                f"{_SAMPLE!r}: TissueSliceShader cannot leave out the "
                "background; update it for this pygfx version"
            )
        return code.replace(_SAMPLE, _SAMPLE + _DISCARD)
