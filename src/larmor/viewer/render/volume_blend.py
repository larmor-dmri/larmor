"""Defines a pygfx material that blends colour and opacity along rays in a volume."""

import math
from dataclasses import dataclass

import numpy as np
import pygfx as gfx
from pygfx.renderers.wgpu import (
    Binding,
    GfxSampler,
    GfxTextureView,
    load_wgsl,
    register_wgpu_render_function,
)
from pygfx.renderers.wgpu.shaders.volumeshader import BaseVolumeShader


@dataclass(frozen=True)
class Gradient:
    """Holds a volume's gradient as a texture, divided by `scale` to fit float16."""

    texture: gfx.Texture
    scale: float


class VolumeBlendMaterial(gfx.VolumeRayMaterial):  # type: ignore[misc]
    """Renders a volume, colour from the colormap and opacity rising across clim.

    A value at or below ``clim[0]`` is transparent; above it, the opacity per voxel
    of path rises linearly to ``density * opacity`` at ``clim[1]``, times the
    colormap's alpha. Samples are composited front to back.

    Args:
        density: Opacity per voxel of path at ``clim[1]``, before ``opacity``.
        step_size: Distance between samples along a ray, in voxels.
        slab_size: Length of the ray pieces drawn one after another, in voxels, so
            that opaque objects inside the volume are hidden by the tissue in front
            of them only; 0 draws each ray at once, which is faster but puts the
            whole ray in front of such objects.
        shading: Light the surfaces inside the volume from the camera, once
            ``gradient`` is set.
        **kwargs: Passed to ``gfx.VolumeRayMaterial``: ``clim``, ``map``, ``opacity``.
    """

    render_mode = "blend"
    uniform_type = dict(
        gfx.VolumeRayMaterial.uniform_type,
        density="f4",
        step_size="f4",
        slab_size="f4",
        gradient_scale="f4",
    )

    def __init__(
        self,
        density: float = 0.1,
        step_size: float = 0.5,
        slab_size: float = 16.0,
        shading: bool = True,
        **kwargs: object,
    ) -> None:
        """Sets the transfer function, the sampling and the shading."""
        kwargs.setdefault("alpha_mode", "blend")
        super().__init__(**kwargs)
        self.density = density
        self.step_size = step_size
        self.slab_size = slab_size
        self.shading = shading
        self.gradient: Gradient | None = None

    @property
    def density(self) -> float:
        """Returns the opacity per voxel of path at ``clim[1]``, before ``opacity``."""
        return float(self.uniform_buffer.data["density"])

    @density.setter
    def density(self, value: float) -> None:
        self.uniform_buffer.data["density"] = max(float(value), 0.0)
        self.uniform_buffer.update_full()

    @property
    def step_size(self) -> float:
        """Returns the distance between samples along a ray, in voxels."""
        return float(self.uniform_buffer.data["step_size"])

    @step_size.setter
    def step_size(self, value: float) -> None:
        if value <= 0:
            raise ValueError("step_size must be positive")
        self.uniform_buffer.data["step_size"] = float(value)
        self.uniform_buffer.update_full()

    @property
    def slab_size(self) -> float:
        """Returns the ray pieces' length, in voxels; 0 for whole rays."""
        return float(self._store.slab_size)

    @slab_size.setter
    def slab_size(self, value: float) -> None:
        value = max(float(value), 0.0)
        # In the store, so a change redraws with the new number of pieces.
        self._store.slab_size = value
        self.uniform_buffer.data["slab_size"] = value
        self.uniform_buffer.update_full()

    @property
    def shading(self) -> bool:
        """Returns whether surfaces inside the volume are lit from the camera."""
        return bool(self._store.shading)

    @shading.setter
    def shading(self, value: bool) -> None:
        self._store.shading = bool(value)

    @property
    def gradient(self) -> Gradient | None:
        """Returns the volume's gradient, from `volume_gradient`; unlit while None."""
        gradient: Gradient | None = self._store.gradient
        return gradient

    @gradient.setter
    def gradient(self, gradient: Gradient | None) -> None:
        self._store.gradient = gradient
        scale = 1.0 if gradient is None else gradient.scale
        self.uniform_buffer.data["gradient_scale"] = scale
        self.uniform_buffer.update_full()


def volume_gradient(values: np.ndarray) -> Gradient:
    """Returns the gradient of a volume texture's values (z, y, x), for the lighting.

    The central differences are stored as float16, divided by the median of their
    sizes so that typical gradients are about 1; a few outliers saturate. The
    lighting uses only the direction.

    Returns:
        Gradient: The gradient texture and its scale.
    """
    grad = np.zeros((*values.shape, 3), np.float32)
    for channel, axis in enumerate((2, 1, 0)):  # the texture's x is the last axis
        n = values.shape[axis]
        if n < 2:
            continue
        ahead = np.take(values, np.r_[1:n, n - 1], axis=axis)
        behind = np.take(values, np.r_[0, 0 : n - 1], axis=axis)
        grad[..., channel] = (ahead - behind) * 0.5
    sizes = np.abs(grad).ravel()
    sizes = sizes[sizes > 0]  # NaN and flat voxels say nothing about the scale
    sample = sizes[:: max(1, sizes.size // _SCALE_SAMPLE)]
    scale = float(np.median(sample)) if sample.size else 1.0
    out = np.zeros((*values.shape, 4), np.float16)
    out[..., :3] = np.clip(grad / scale, -_FLOAT16_MAX, _FLOAT16_MAX)
    return Gradient(gfx.Texture(out, dim=3), scale)


# Size of the sample of differences whose median is the gradient's scale.
_SCALE_SAMPLE = 1_000_000
_FLOAT16_MAX = float(np.finfo(np.float16).max)


def slab_count(shape: tuple[int, ...], slab_size: float) -> int:
    """Returns the number of slabs that cover the longest ray through the grid."""
    if slab_size <= 0:
        return 1
    return max(1, math.ceil(math.hypot(*shape[:3]) / slab_size))


@register_wgpu_render_function(gfx.Volume, VolumeBlendMaterial)
class VolumeBlendShader(BaseVolumeShader):  # type: ignore[misc]
    """Ray casts the volume in slabs, back to front, each composited front to back."""

    type = "render"

    def get_bindings(
        self, wobject: gfx.Volume, shared: object, scene: object
    ) -> object:
        """Adds the shader's template values to pygfx's volume bindings."""
        material = wobject.material
        gradient = material.gradient
        self["shading"] = material.shading and gradient is not None
        self["n_slabs"] = slab_count(wobject.geometry.grid.size, material.slab_size)
        bindings: dict[int, dict[int, Binding]] = super().get_bindings(
            wobject, shared, scene
        )
        if self["shading"]:
            group = bindings[0]
            extra = [
                Binding(
                    "t_grad",
                    "texture/auto",
                    GfxTextureView(gradient.texture),
                    "FRAGMENT",
                ),
                Binding(
                    "s_grad",
                    "sampler/filtering",
                    GfxSampler("linear", "clamp"),
                    "FRAGMENT",
                ),
            ]
            for binding in extra:
                index = len(group)
                group[index] = binding
                self.define_binding(0, index, binding)
        return bindings

    def get_pipeline_info(
        self, wobject: gfx.Volume, shared: object
    ) -> dict[str, object]:
        """Returns the box's triangles, front faces culled so rays start at the back."""
        return {
            "primitive_topology": "triangle-list",
            "cull_mode": "front",
        }

    def get_render_info(self, wobject: gfx.Volume, shared: object) -> dict[str, object]:
        """Returns the box's 36 vertices, once per slab."""
        n_slabs = slab_count(wobject.geometry.grid.size, wobject.material.slab_size)
        return {"indices": (36, n_slabs)}

    def get_code(self) -> str:
        """Returns the WGSL next to this module."""
        code: str = load_wgsl("volume_blend.wgsl", package_name=__package__)
        return code
