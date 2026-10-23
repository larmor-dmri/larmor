"""Voxelizes streamlines on the GPU, for their ray tracing and ambient light.

References:
    Kraaijeveld B, Jalba AC, Vilanova A, Chamberland M. Real-Time Rendering of
    Dynamic Line Sets using Voxel Ray Tracing. arXiv:2510.09081 (2025).
"""

import time
from dataclasses import dataclass
from importlib.resources import files

import numpy as np
import pygfx as gfx
import wgpu
from pygfx.renderers.wgpu.engine.update import ensure_wgpu_object

from larmor.viewer.render.geometry import Lines

# Voxels along the cubic grid's side; each level of a pyramid halves it.
RESOLUTION = 128
# Radius in voxels below which a capsule is widened and its occupancy lowered to
# match, so thin lines do not alias.
MIN_RADIUS = 0.5
# Number of cones traced from each voxel, and how far they reach, in voxels.
CONES = 16
REACH = 16.0
# Percentile of the points' light that counts as fully lit.
LIT_PERCENTILE = 95
_GROUP = 64  # threads per workgroup of most passes
_SCAN = 256  # voxels per block of the scan
_MAX_GROUPS = 65535  # workgroups along one dimension of a dispatch
_STORAGE = (
    wgpu.BufferUsage.STORAGE
    | wgpu.BufferUsage.VERTEX
    | wgpu.BufferUsage.COPY_SRC
    | wgpu.BufferUsage.COPY_DST
)
_PASSES = (
    "voxelize",
    "resolve",
    "pyramid_level",
    "scan_blocks",
    "scan_sums",
    "add_sums",
    "fill",
    "ambient",
    "light_at_points",
)


@dataclass(frozen=True)
class Grid:
    """Holds a voxel grid round lines; voxel i spans origin + voxel * [i, i+1]."""

    origin: np.ndarray  # (3,) float64, mm
    voxel: float  # mm
    size: int

    @property
    def levels(self) -> int:
        """Returns the levels of a pyramid over the grid, down to one voxel."""
        return int(np.log2(self.size)) + 1

    def to_voxels(self, points: np.ndarray) -> np.ndarray:
        """Returns points in mm as voxel coordinates."""
        voxels: np.ndarray = (np.asarray(points, np.float64) - self.origin) / self.voxel
        return voxels


def grid_for(points: np.ndarray, size: int = RESOLUTION) -> Grid:
    """Returns a cubic grid round the points.

    A margin leaves room for the cones' reach and the radius.
    """
    p = np.asarray(points, np.float64)
    if len(p) == 0:
        return Grid(np.zeros(3), 1.0, size)
    low, high = p.min(axis=0), p.max(axis=0)
    voxel = max(float((high - low).max()), 1e-3) / (size - 2 * (REACH // 4) - 4)
    return Grid((low + high) / 2 - voxel * size / 2, voxel, size)


def cone_directions(count: int) -> np.ndarray:
    """Returns directions spread evenly over the sphere, on a Fibonacci spiral."""
    i = np.arange(count) + 0.5
    z = 1 - 2 * i / count
    angle = np.pi * (1 + 5**0.5) * i
    ring = np.sqrt(1 - z * z)
    return np.stack([ring * np.cos(angle), ring * np.sin(angle), z], axis=1)


def cone_tan_half(count: int) -> float:
    """Returns the tangent of half the aperture of cones sharing the sphere evenly."""
    return float(np.tan(np.arccos(1 - 2 / count)))


def segment_starts(offsets: np.ndarray, count: int) -> np.ndarray:
    """Returns the points that start a segment, all but each streamline's last."""
    joined = np.ones(max(count - 1, 0), bool)
    inner = offsets[1:-1]
    joined[inner[(inner > 0) & (inner < count)] - 1] = False
    return np.flatnonzero(joined)


def storage(nitems: int, fmt: str = "f4") -> gfx.Buffer:
    """Returns a GPU-only buffer of nitems values, for passes and shaders."""
    nitems = max(nitems, 1)
    width = int(fmt.split("x")[0]) if "x" in fmt else 1
    return gfx.Buffer(
        nbytes=4 * width * nitems, nitems=nitems, format=fmt, usage=_STORAGE
    )


class LineVoxels:
    """Holds the GPU buffers and passes for one set of lines at a time.

    After `build`, the buffers hold the occupancy and touched pyramids, each
    voxel's segments and its ambient light, for the ray tracer's shaders. Voxel
    (x, y, z) is at x + size * (y + size * z), and a pyramid's level k starts at
    `level_starts[k]`.
    """

    def __init__(self, size: int = RESOLUTION) -> None:
        """Makes the passes and the buffers of a grid of size^3 voxels."""
        self.device = gfx.renderers.wgpu.get_shared().device
        self.size = size
        voxels = size**3
        self.level_starts = np.concatenate(
            [[0], np.cumsum([(size >> k) ** 3 for k in range(int(np.log2(size)))])]
        ).astype(np.uint32)
        total = int(self.level_starts[-1]) + 1
        self.blocks = -(-voxels // _SCAN)
        self.filled = storage(voxels, "u4")
        self.counts = storage(voxels, "u4")
        self.occupancy = storage(total)
        self.touched = storage(total, "u4")
        self.offsets = storage(voxels + 1, "u4")
        self.sums = storage(self.blocks + 1, "u4")
        self.light = storage(voxels)
        self.points = storage(1)
        self.starts = storage(1, "u4")
        self.segments = storage(4, "4xf4")
        self.point_light = storage(1)
        self.lists = storage(1, "u4")
        self.grid = grid_for(np.zeros((0, 3)), size)
        self.num_points = 0
        self.num_segments = 0
        self.num_entries = 0
        limits = self.device.limits
        # The most bytes the segment lists may take, the size of one buffer the
        # shaders read on this GPU.
        self.max_list_bytes = min(
            int(limits["max-storage-buffer-binding-size"]),
            int(limits["max-buffer-size"]),
        )
        self.lists_fit = True  # whether the last build's lists were made
        self.timings: dict[str, float] = {}  # seconds per stage of the last build
        self._params = self.device.create_buffer(
            size=624, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        )
        self._levels = [
            self.device.create_buffer_with_data(
                data=np.array([k, 0, 0, 0], np.uint32),
                usage=wgpu.BufferUsage.UNIFORM,
            )
            for k in range(int(np.log2(size)) + 1)
        ]
        # Where the offsets' last value is copied to be read back alone.
        self._last = self.device.create_buffer(
            size=4, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC
        )
        self._layouts, self._pipelines = _pipelines(self.device)

    def build(self, lines: Lines, radius: float, lists: bool = True) -> None:
        """Voxelizes the lines, capsules of `radius` mm, and lights the voxels.

        With lists, also lists each voxel's segments, unless they would take more
        than max_list_bytes; lists_fit then says so.
        """
        start = time.perf_counter()
        self._upload_lines(lines, radius)
        self.timings = {"upload": time.perf_counter() - start}

        start = time.perf_counter()
        self._voxelize(lists)
        if not lists:
            self._wait()
            self.timings["voxels, pyramid and light"] = time.perf_counter() - start
            return
        self.num_entries = self._last_offset()
        self.timings["voxels, pyramid, light and scan"] = time.perf_counter() - start
        self.lists_fit = 4 * self.num_entries <= self.max_list_bytes
        if not self.lists_fit:
            return

        start = time.perf_counter()
        self._fill_lists()
        self.timings["segment lists"] = time.perf_counter() - start

    def _upload_lines(self, lines: Lines, radius: float) -> None:
        """Sets the grid round the lines and uploads their points and parameters."""
        self.grid = grid = grid_for(lines.points, self.size)
        starts = segment_starts(lines.offsets, len(lines.points)).astype(np.uint32)
        self.num_points, self.num_segments = len(lines.points), len(starts)
        voxels = grid.to_voxels(lines.points).astype(np.float32)
        self.points = self._upload(self.points, voxels.ravel(), "f4")
        self.starts = self._upload(self.starts, starts, "u4")
        if self.segments.nitems < 4 * max(self.num_segments, 1):
            self.segments = storage(4 * self.num_segments, "4xf4")
        if self.point_light.nitems < max(self.num_points, 1):
            self.point_light = storage(self.num_points)
        self._write_params(radius / grid.voxel)

    def _voxelize(self, lists: bool) -> None:
        """Submits the voxels, pyramid and light passes, and the scan with lists."""
        encoder = self.device.create_command_encoder()
        for buffer in (self.filled, self.counts):
            encoder.clear_buffer(ensure_wgpu_object(buffer))
        compute = encoder.begin_compute_pass()
        self._dispatch(compute, "voxelize", self.num_segments)
        self._dispatch(compute, "resolve", self.size**3)
        for k in range(1, self.grid.levels):
            self._dispatch(compute, "pyramid_level", (self.size >> k) ** 3, k)
        self._dispatch(compute, "ambient", self.size**3)
        if lists:
            self._dispatch_blocks(compute, "scan_blocks", self.blocks)
            self._dispatch_blocks(compute, "scan_sums", 1)
            self._dispatch(compute, "add_sums", self.size**3 + 1)
        compute.end()
        self.device.queue.submit([encoder.finish()])

    def _fill_lists(self) -> None:
        """Fills each voxel's list of segments, and waits for it."""
        if self.lists.nitems < max(self.num_entries, 1):
            self.lists = storage(self.num_entries, "u4")
        encoder = self.device.create_command_encoder()
        compute = encoder.begin_compute_pass()
        self._dispatch(compute, "fill", self.num_segments)
        compute.end()
        self.device.queue.submit([encoder.finish()])
        self._wait()

    def light_at_points(self) -> np.ndarray:
        """Returns the ambient light at the points last built, (P,) float32."""
        encoder = self.device.create_command_encoder()
        compute = encoder.begin_compute_pass()
        self._dispatch(compute, "light_at_points", self.num_points)
        compute.end()
        self.device.queue.submit([encoder.finish()])
        return self.read(self.point_light)[: self.num_points]

    def read(self, buffer: gfx.Buffer, fmt: str = "f4") -> np.ndarray:
        """Returns a buffer's values, copied back from the GPU."""
        data = self.device.queue.read_buffer(ensure_wgpu_object(buffer))
        return np.frombuffer(data, np.dtype("<" + fmt)).copy()

    def grid_values(self, buffer: gfx.Buffer, level: int = 0) -> np.ndarray:
        """Returns one level of a buffer over the grid, as an (x, y, z) array."""
        n = self.size >> level
        fmt = "u4" if buffer.format.startswith("u") else "f4"
        start = int(self.level_starts[level])
        values = self.read(buffer, fmt)[start : start + n**3]
        return values.reshape(n, n, n).transpose(2, 1, 0)

    def _upload(self, buffer: gfx.Buffer, values: np.ndarray, fmt: str) -> gfx.Buffer:
        """Writes values to buffer, or to a larger one it returns if they do not fit."""
        if buffer.nitems < max(len(values), 1):
            buffer = storage(len(values), fmt)
        if len(values):
            self.device.queue.write_buffer(ensure_wgpu_object(buffer), 0, values)
        return buffer

    def _write_params(self, radius: float) -> None:
        """Writes the passes' uniform parameters, with the radius in voxels."""
        head = np.zeros(12, np.uint32)
        head[:4] = (self.size, self.grid.levels, self.num_segments, self.num_points)
        head[4:8] = np.array(
            [radius, MIN_RADIUS, cone_tan_half(CONES), REACH], np.float32
        ).view(np.uint32)
        head[8:10] = (CONES, self.blocks)
        level_starts = np.zeros(16, np.uint32)
        level_starts[: len(self.level_starts)] = self.level_starts
        directions = np.zeros((32, 4), np.float32)
        directions[:CONES, :3] = cone_directions(CONES)
        data = np.concatenate([head, level_starts, directions.ravel().view(np.uint32)])
        self.device.queue.write_buffer(self._params, 0, data)

    def _bind_groups(self, level: int) -> tuple[wgpu.GPUBindGroup, wgpu.GPUBindGroup]:
        """Returns the bind groups of every buffer and of a pyramid level."""
        buffers = (
            self.points,
            self.starts,
            self.filled,
            self.counts,
            self.occupancy,
            self.touched,
            self.offsets,
            self.sums,
            self.lists,
            self.light,
            self.point_light,
            self.segments,
        )
        entries = [{"binding": 0, "resource": {"buffer": self._params}}]
        for slot, buffer in enumerate(buffers, start=1):
            entries.append(
                {"binding": slot, "resource": {"buffer": ensure_wgpu_object(buffer)}}
            )
        main = self.device.create_bind_group(layout=self._layouts[0], entries=entries)
        per_level = self.device.create_bind_group(
            layout=self._layouts[1],
            entries=[{"binding": 0, "resource": {"buffer": self._levels[level]}}],
        )
        return main, per_level

    def _dispatch(
        self,
        compute: wgpu.GPUComputePassEncoder,
        name: str,
        threads: int,
        level: int = 0,
    ) -> None:
        """Dispatches a pass over this many threads."""
        self._dispatch_blocks(compute, name, -(-threads // _GROUP), level)

    def _dispatch_blocks(
        self,
        compute: wgpu.GPUComputePassEncoder,
        name: str,
        groups: int,
        level: int = 0,
    ) -> None:
        """Dispatches a pass over this many workgroups, if there is any."""
        if groups <= 0:
            return
        main, per_level = self._bind_groups(level)
        compute.set_pipeline(self._pipelines[name])
        compute.set_bind_group(0, main)
        compute.set_bind_group(1, per_level)
        x = min(groups, _MAX_GROUPS)
        compute.dispatch_workgroups(x, -(-groups // x), 1)

    def _last_offset(self) -> int:
        """Returns the end of the last voxel's list, the entries in all lists."""
        encoder = self.device.create_command_encoder()
        encoder.copy_buffer_to_buffer(
            ensure_wgpu_object(self.offsets), 4 * self.size**3, self._last, 0, 4
        )
        self.device.queue.submit([encoder.finish()])
        return int(np.frombuffer(self.device.queue.read_buffer(self._last), "<u4")[0])

    def _wait(self) -> None:
        """Waits for the work submitted so far."""
        self.read(self.sums, "u4")  # a small read waits for the work before it


_CACHE: dict[
    int, tuple[list[wgpu.GPUBindGroupLayout], dict[str, wgpu.GPUComputePipeline]]
] = {}


def _pipelines(
    device: wgpu.GPUDevice,
) -> tuple[list[wgpu.GPUBindGroupLayout], dict[str, wgpu.GPUComputePipeline]]:
    """Returns the passes' layouts and pipelines, made once per device."""
    if id(device) in _CACHE:
        return _CACHE[id(device)]
    stage = wgpu.ShaderStage.COMPUTE
    entries = [{"binding": 0, "visibility": stage, "buffer": {"type": "uniform"}}]
    for slot in range(1, 13):
        kind = "read-only-storage" if slot in (1, 2) else "storage"
        entries.append({"binding": slot, "visibility": stage, "buffer": {"type": kind}})
    main = device.create_bind_group_layout(entries=entries)
    level = device.create_bind_group_layout(
        entries=[{"binding": 0, "visibility": stage, "buffer": {"type": "uniform"}}]
    )
    layout = device.create_pipeline_layout(bind_group_layouts=[main, level])
    code = files("larmor.viewer.render").joinpath("line_voxels.wgsl").read_text()
    module = device.create_shader_module(code=code)
    pipelines = {
        name: device.create_compute_pipeline(
            layout=layout, compute={"module": module, "entry_point": name}
        )
        for name in _PASSES
    }
    _CACHE[id(device)] = ([main, level], pipelines)
    return _CACHE[id(device)]


def ambient_light(
    lines: Lines, radius: float, voxels: LineVoxels | None = None
) -> np.ndarray:
    """Returns the ambient light at each point of the lines, 1 where nothing occludes.

    The lines are capsules of `radius` mm, built into `voxels` (new ones if None)
    without segment lists. Values are scaled so that the most exposed points (the
    LIT_PERCENTILE-th percentile) get 1.

    Returns:
        (P,) float32 from 0 to 1.
    """
    if len(lines.points) == 0:
        return np.ones(0, np.float32)
    voxels = voxels or LineVoxels()
    voxels.build(lines, radius, lists=False)
    return lit_share(voxels.light_at_points())


def lit_share(light: np.ndarray) -> np.ndarray:
    """Returns light scaled so its LIT_PERCENTILE-th percentile is 1.

    The result is clipped to 0..1.
    """
    top = float(np.percentile(light, LIT_PERCENTILE)) if len(light) else 1.0
    return np.clip(light / max(top, 1e-6), 0, 1).astype(np.float32)
