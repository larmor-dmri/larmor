"""Draws streamlines by ray tracing their voxel grid, as a pygfx object.

References:
    Kraaijeveld B, Jalba AC, Vilanova A, Chamberland M. Real-Time Rendering of
    Dynamic Line Sets using Voxel Ray Tracing. arXiv:2510.09081 (2025).
"""

from typing import cast

import numpy as np
import pygfx as gfx
import wgpu
from pygfx.renderers.wgpu import (
    BaseShader,
    Binding,
    load_wgsl,
    register_wgpu_render_function,
)
from pygfx.renderers.wgpu.engine.shared import Shared
from pygfx.renderers.wgpu.engine.update import ensure_wgpu_object

from larmor.viewer.render.line_voxels import LineVoxels, storage

_BUFFERS = (
    "segments",
    "colours",
    "offsets",
    "lists",
    "touched",
    "occupancy",
    "light",
    "shadow",
)


class RayTracedLinesMaterial(gfx.Material):  # type: ignore[misc]
    """Holds how the ray-traced lines look: radius, ambient light and translucency."""

    uniform_type = dict(  # noqa: RUF012 - pygfx's convention
        gfx.Material.uniform_type,
        level_start_a="4xu4",
        level_start_b="4xu4",
        size="u4",
        levels="u4",
        radius="f4",
        light_scale="f4",
        ambient_occlusion="f4",
    )

    def __init__(self, **kwargs: object) -> None:
        """Creates a material with no buffers; `RayTracedLines.set_grid` gives them."""
        super().__init__(**kwargs)
        self.transparent = False
        self.buffers = {}

    @property
    def transparent(self) -> bool:
        """Returns whether every hit is blended, or the first one hides the rest."""
        return bool(self._store.transparent)

    @transparent.setter
    def transparent(self, value: bool) -> None:
        self._store.transparent = bool(value)

    @property
    def buffers(self) -> dict[str, gfx.Buffer]:
        """Returns the GPU buffers the shader reads, by name."""
        return dict(self._store.buffers)

    @buffers.setter
    def buffers(self, value: dict[str, gfx.Buffer]) -> None:
        self._store.buffers = dict(value)

    @property
    def size(self) -> int:
        """Returns the voxels along a side of the grid."""
        return int(self.uniform_buffer.data["size"])

    def set_uniforms(self, **values: object) -> None:
        """Sets fields of the uniform buffer."""
        for name, value in values.items():
            self.uniform_buffer.data[name] = value
        self.uniform_buffer.update_full()


class RayTracedLines(gfx.WorldObject):  # type: ignore[misc]
    """Draws lines as capsules ray traced through their voxel grid.

    The hits write their depth, so the lines sit among the scene's other objects;
    translucent lines blend every hit along the ray.
    """

    def __init__(self, material: RayTracedLinesMaterial) -> None:
        """Creates an object with nothing to draw until `set_grid` and `set_colours`."""
        box = np.array(
            [[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], np.float32
        )
        super().__init__(gfx.Geometry(positions=box), material)
        self._colours = storage(1)
        self._shadow = storage(1)  # the lamp's light per voxel, made each frame

    def set_grid(
        self, voxels: LineVoxels, radius: float, light_scale: float | None
    ) -> None:
        """Draws the lines last built into `voxels`, capsules of `radius` mm.

        light_scale scales the voxels' ambient light so that the most exposed
        points get 1; None draws without ambient occlusion.
        """
        material = self.material
        assert isinstance(material, RayTracedLinesMaterial)
        grid = voxels.grid
        if self._shadow.nitems < voxels.size**3:
            self._shadow = storage(voxels.size**3)
        starts = np.zeros(8, np.uint32)
        starts[: len(voxels.level_starts)] = voxels.level_starts[:8]
        material.set_uniforms(
            level_start_a=starts[:4],
            level_start_b=starts[4:],
            size=voxels.size,
            levels=grid.levels,
            radius=radius / grid.voxel,
            light_scale=1.0 if light_scale is None else light_scale,
            ambient_occlusion=float(light_scale is not None),
        )
        buffers = {
            "segments": voxels.segments,
            "colours": self._colours,
            "offsets": voxels.offsets,
            "lists": voxels.lists,
            "touched": voxels.touched,
            "occupancy": voxels.occupancy,
            "light": voxels.light,
            "shadow": self._shadow,
        }
        if any(material.buffers.get(k) is not v for k, v in buffers.items()):
            material.buffers = buffers
        self.local.position = tuple(grid.origin)
        self.local.scale = (grid.voxel,) * 3
        self.visible = voxels.num_segments > 0

    def set_colours(self, colours: np.ndarray) -> None:
        """Gives the points their sRGB colours, (P, 3)."""
        if self._colours.nitems < max(3 * len(colours), 1):
            self._colours = storage(3 * len(colours))
            material = self.material
            assert isinstance(material, RayTracedLinesMaterial)
            material.buffers = {**material.buffers, "colours": self._colours}
        values = np.ascontiguousarray(colours, np.float32).ravel()
        if values.size:
            device = gfx.renderers.wgpu.get_shared().device
            device.queue.write_buffer(ensure_wgpu_object(self._colours), 0, values)


def _wgsl(*names: str) -> str:
    """Returns the WGSL files next to this module, one after the other."""
    return "\n".join(str(load_wgsl(name, package_name=__package__)) for name in names)


class _VoxelShader(BaseShader):  # type: ignore[misc]
    """Binds the uniforms and the grid's buffers that the WGSL reads."""

    stage = "FRAGMENT"  # where the buffers are read
    uniform_stages = wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT
    writes = ""  # the buffer the shader writes, if any

    def get_bindings(
        self, wobject: RayTracedLines, shared: object, scene: object
    ) -> dict[int, dict[int, Binding]]:
        """Returns the bindings of the uniforms and the voxel grid's buffers."""
        material = wobject.material
        assert isinstance(material, RayTracedLinesMaterial)
        self["transparent"] = material.transparent
        buffers = material.buffers
        everywhere = self.uniform_stages
        bindings = {
            0: Binding(
                "u_stdinfo",
                "buffer/uniform",
                cast("Shared", shared).uniform_buffer,
                everywhere,
            ),
            1: Binding(
                "u_wobject", "buffer/uniform", wobject.uniform_buffer, everywhere
            ),
            2: Binding(
                "u_material", "buffer/uniform", material.uniform_buffer, everywhere
            ),
        }
        for slot, name in enumerate(_BUFFERS, start=3):
            buffer = buffers.get(name) or storage(1)
            kind = "storage" if name == self.writes else "read_only_storage"
            bindings[slot] = Binding(f"s_{name}", f"buffer/{kind}", buffer, self.stage)
        self.define_bindings(0, bindings)
        return {0: bindings}


class VoxelShadowShader(_VoxelShader):
    """Computes the lamp's light at each occupied voxel, before each frame."""

    type = "compute"
    stage = "COMPUTE"
    uniform_stages = wgpu.ShaderStage.COMPUTE
    writes = "shadow"

    def get_pipeline_info(
        self, wobject: RayTracedLines, shared: object
    ) -> dict[str, object]:
        """Returns nothing, as a compute pass needs no pipeline info."""
        return {}

    def get_render_info(
        self, wobject: RayTracedLines, shared: object
    ) -> dict[str, object]:
        """Returns a thread per voxel, in workgroups of 64."""
        material = wobject.material
        assert isinstance(material, RayTracedLinesMaterial)
        groups = -(-(material.size**3) // 64)
        x = min(groups, 65535)
        return {"indices": (x, -(-groups // x), 1)}

    def get_code(self) -> str:
        """Returns the WGSL, from the viewer's resources."""
        return _wgsl("voxel_shadows.wgsl", "voxel_cones.wgsl")


class RayTracedLinesShader(_VoxelShader):
    """Draws `RayTracedLines`."""

    type = "render"

    def get_pipeline_info(
        self, wobject: RayTracedLines, shared: object
    ) -> dict[str, object]:
        """Returns triangles of the box, its back faces only."""
        return {
            "primitive_topology": wgpu.PrimitiveTopology.triangle_list,
            "cull_mode": wgpu.CullMode.front,
        }

    def get_render_info(
        self, wobject: RayTracedLines, shared: object
    ) -> dict[str, object]:
        """Returns the box's 12 triangles."""
        return {"indices": (36, 1)}

    def get_code(self) -> str:
        """Returns the WGSL, from the viewer's resources."""
        return _wgsl("ray_traced_lines.wgsl", "voxel_cones.wgsl")


def _shaders(wobject: RayTracedLines) -> list[BaseShader]:
    """Returns the shadow pass, then the ray tracer."""
    return [VoxelShadowShader(wobject), RayTracedLinesShader(wobject)]


register_wgpu_render_function(RayTracedLines, RayTracedLinesMaterial)(_shaders)
