"""Defines a pygfx slice material drawn a little nearer the camera than other slices."""

import pygfx as gfx
from pygfx.renderers.wgpu import load_wgsl, register_wgpu_render_function
from pygfx.renderers.wgpu.shaders.volumeshader import VolumeSliceShader

# The line of pygfx's slice shader that sets the vertex's position, and its
# replacement, which moves it towards the camera by DEPTH_OFFSET in normalised depth.
DEPTH_OFFSET = 1e-4
_POSITION = "varyings.position = vec4<f32>(ndc_pos);"
_OFFSET_POSITION = (
    "varyings.position = vec4<f32>"
    f"(ndc_pos.xy, ndc_pos.z - {DEPTH_OFFSET} * ndc_pos.w, ndc_pos.w);"
)


class OverlaySliceMaterial(gfx.VolumeSliceMaterial):  # type: ignore[misc]
    """Draws a volume slice that wins the depth test over other slices in its plane."""


@register_wgpu_render_function(gfx.Volume, OverlaySliceMaterial)
class OverlaySliceShader(VolumeSliceShader):  # type: ignore[misc]
    """Draws pygfx's slices with the vertices moved towards the camera."""

    def get_code(self) -> str:
        """Returns the WGSL of pygfx's slices, with the position line replaced."""
        code: str = load_wgsl("volume_slice.wgsl")
        if _POSITION not in code:
            raise RuntimeError(
                "pygfx's volume_slice.wgsl no longer contains "
                f"{_POSITION!r}: OverlaySliceShader cannot offset the overlay's "
                "slices in depth; update it for this pygfx version"
            )
        return code.replace(_POSITION, _OFFSET_POSITION)
