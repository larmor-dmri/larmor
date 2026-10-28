"""Computes what the Signal breakdown dock shows for one voxel of the phantom."""

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from larmor.types import GradientTable, Grid
from larmor.viewer.model.phantom import (
    PhantomState,
    ScanSettings,
    build_phantom,
    build_protocol,
)

if TYPE_CHECKING:
    from larmor.simulation import (
        Phantom,
        Protocol,
        SignalBreakdown,
        VoxelPartition,
    )
    from larmor.viewer.model.phantom_editor import PhantomEditor

# larmor.simulation is imported only in the functions that need it, so opening the
# viewer does not load it.

type Voxel = tuple[int, int, int]
type Vec3 = tuple[float, float, float]

# The problems shown instead of a breakdown.
NO_PHANTOM = "Start or open a phantom to see the signal of its voxels."
NO_VOXEL = "Point at a voxel of the phantom in a 2D view, or click one to pin it."
OUTSIDE = "The voxel is outside the phantom's grid."


@dataclass(frozen=True)
class BreakdownState:
    """What the Signal breakdown dock shows.

    Attributes:
        voxel: The voxel shown, (i, j, k) of the phantom's grid; None until the
            pointer is first over the phantom.
        pinned: Whether a click fixed the voxel; otherwise it follows the pointer.
        volume: The scan's volume, an index of its gradient table; None for the
            first diffusion-weighted volume.
        breakdown: The voxel's breakdown; None when there is none (see problem).
        problem: Why there is no breakdown, in one sentence; None when there is.
    """

    voxel: Voxel | None = None
    pinned: bool = False
    volume: int | None = None
    breakdown: SignalBreakdown | None = None
    problem: str | None = NO_PHANTOM


def voxel_at(grid: Grid, position: Vec3) -> Voxel | None:
    """Returns the grid's voxel nearest a world position in mm; None outside it."""
    inverse = np.linalg.inv(grid.affine)
    point = inverse[:3, :3] @ np.asarray(position, dtype=float) + inverse[:3, 3]
    nearest = np.floor(point + 0.5).astype(int)
    if np.any(nearest < 0) or np.any(nearest >= np.array(grid.shape)):
        return None
    return (int(nearest[0]), int(nearest[1]), int(nearest[2]))


def first_weighted(gradients: GradientTable) -> int:
    """Returns the first diffusion-weighted volume, else the first volume."""
    weighted = np.flatnonzero(gradients.bvals > gradients.b0_threshold)
    return int(weighted[0]) if len(weighted) else 0


def title(state: BreakdownState | None) -> str:
    """Returns the dock's title, which names the voxel shown."""
    if state is None or state.voxel is None:
        return "Signal breakdown"
    return "Signal breakdown - voxel ({}, {}, {})".format(*state.voxel)


@dataclass(frozen=True)
class CardRow:
    """A row of the table of what fills the voxel.

    Attributes:
        name: The tissue's name, or the bundle's under its tissue.
        share: The share of the voxel, such as "41%".
        angle: The angle between the bundle's fibres and the gradient, such as
            "43 deg"; empty for a tissue and at b = 0.
        bundle: Whether the row is a bundle, shown indented under its tissue.
    """

    name: str
    share: str
    angle: str = ""
    bundle: bool = False


def heading(state: BreakdownState) -> str:
    """Returns the voxel shown as text, such as "Voxel (7, 9, 7)"."""
    if state.voxel is None:
        return ""
    return "Voxel ({}, {}, {})".format(*state.voxel)


def card_rows(breakdown: SignalBreakdown) -> list[CardRow]:
    """Returns a row per tissue in the voxel, each followed by one per bundle of it.

    A tissue whose share rounds to 0% is left out.

    Returns:
        The rows, in the phantom's order of tissues and bundles.
    """
    rows = []
    for name, share in breakdown.tissue_fractions:
        if round(100 * share) == 0:
            continue
        rows.append(CardRow(name, f"{share:.0%}"))
        for b in breakdown.bundles:
            if b.tissue == name:
                angle = "" if b.angle_deg is None else f"{b.angle_deg:.0f} deg"
                rows.append(CardRow(b.name, f"{b.share:.0%}", angle, bundle=True))
    return rows


class BreakdownTool:
    """Carries out the Viewer's breakdown actions on its scene.

    The state lives in `scene.breakdown`; this keeps only caches of the phantom,
    its voxel partition and the scan's protocol.
    """

    def __init__(self, editor: PhantomEditor) -> None:
        """Follows the phantom of the editor's viewer."""
        self.editor = editor
        self.scene = editor.viewer.scene
        # (key, the data it names, partition)
        self._partition: tuple[object, object, VoxelPartition] | None = None
        # (scan, grid shape, grid affine, protocol)
        self._protocol: tuple[ScanSettings, object, bytes, Protocol] | None = None
        # (phantom state, data ids, data, phantom, bundle names)
        self._phantom: (
            tuple[PhantomState, tuple[int, ...], object, Phantom, list[str]] | None
        ) = None

    @property
    def state(self) -> BreakdownState | None:
        """What the dock shows; None while it is closed."""
        return self.scene.breakdown

    def show(self, shown: bool) -> None:
        """Starts following the phantom, or stops."""
        if not shown:
            # The caches stay, as they are checked against the phantom on use and
            # a partition can take seconds to make again when the panel reopens.
            if self.state is not None:
                self.scene.set_breakdown(None)
            return
        if self.state is None:
            self.scene.set_breakdown(self._computed(BreakdownState()))

    def forget(self) -> None:
        """Drops the cached partition, phantom and protocol when the phantom closes."""
        self._partition = self._protocol = self._phantom = None

    def hover(self, position: Vec3) -> None:
        """Follows the pointer to the voxel under it, unless a voxel is pinned."""
        state = self.state
        phantom = self.scene.phantom
        if state is None or state.pinned or phantom is None:
            return
        voxel = voxel_at(phantom.grid, position)
        if voxel is None or voxel == state.voxel:
            return
        self.scene.set_breakdown(self._computed(replace(state, voxel=voxel)))

    def pin(self, position: Vec3 | None) -> None:
        """Pins the voxel at a position, or unpins with None."""
        if position is None:
            state = self.state
            if state is not None and state.pinned:
                self.scene.set_breakdown(replace(state, pinned=False))
            return
        phantom = self.scene.phantom
        if phantom is None:
            raise ValueError("No phantom is open; start one with new_phantom.")
        voxel = voxel_at(phantom.grid, position)
        if voxel is None:
            raise ValueError("The point is outside the phantom's grid.")
        state = self.state if self.state is not None else BreakdownState()
        if state.pinned and state.voxel == voxel:
            return  # a drag within the voxel
        pinned = replace(state, voxel=voxel, pinned=True)
        self.scene.set_breakdown(self._computed(pinned))

    def move_to_grid(self, old: Grid, new: Grid) -> None:
        """Keeps the voxel shown at its place in mm when the phantom's grid changes.

        A voxel that falls outside the new grid is dropped, and so is its pin.
        """
        state = self.state
        if state is None or state.voxel is None:
            return
        centre = old.affine[:3, :3] @ np.asarray(state.voxel, float) + old.affine[:3, 3]
        voxel = voxel_at(new, (float(centre[0]), float(centre[1]), float(centre[2])))
        moved = replace(state, voxel=voxel, pinned=state.pinned and voxel is not None)
        self.scene.set_breakdown(self._computed(moved))

    def set_volume(self, volume: int) -> None:
        """Shows the voxel's signal in another volume of the scan."""
        state = self.state
        if state is None:
            raise ValueError("The signal breakdown is not shown.")
        gradients = self.gradients()
        if gradients is not None and not 0 <= volume < len(gradients.bvals):
            raise ValueError(
                f"volume must be in [0, {len(gradients.bvals)}), got {volume}."
            )
        self.scene.set_breakdown(self._computed(replace(state, volume=volume)))

    def refresh(self) -> None:
        """Computes the voxel again after an edit."""
        state = self.state
        if state is None:
            return
        new = self._computed(state)
        if new.breakdown == state.breakdown and new.problem == state.problem:
            return
        self.scene.set_breakdown(new)

    def gradients(self) -> GradientTable | None:
        """Returns the scan's gradient table; None without a phantom or a valid scan."""
        phantom = self.scene.phantom
        if phantom is None:
            return None
        try:
            return self._protocol_of(phantom.scan, phantom.grid).gradients
        except OSError, ValueError:
            return None

    def _computed(self, state: BreakdownState) -> BreakdownState:
        """Returns the state with the breakdown of its voxel, or the problem."""
        phantom_state = self.scene.phantom
        if phantom_state is None:
            return replace(state, breakdown=None, problem=NO_PHANTOM)
        if state.voxel is None:
            return replace(state, breakdown=None, problem=NO_VOXEL)
        if any(
            v >= n for v, n in zip(state.voxel, phantom_state.grid.shape, strict=True)
        ):
            return replace(state, breakdown=None, problem=OUTSIDE)
        import larmor.simulation as sim

        try:
            protocol = self._protocol_of(phantom_state.scan, phantom_state.grid)
            phantom, partition, names = self._phantom_of(phantom_state)
            volume = state.volume
            if volume is None or volume >= len(protocol.gradients.bvals):
                volume = first_weighted(protocol.gradients)
            # The same noise as a run.
            breakdown = sim.signal_breakdown(
                phantom,
                protocol,
                partition,
                state.voxel,
                volume,
                snr=phantom_state.scan.snr,
                noise_reference=phantom_state.scan.noise_reference,
                bundle_names=names,
            )
        except (OSError, TypeError, ValueError) as e:
            return replace(state, breakdown=None, problem=str(e))
        return replace(state, breakdown=breakdown, problem=None)

    def _protocol_of(self, scan: ScanSettings, grid: Grid) -> Protocol:
        """Returns the scan's protocol, built again when the scan or the grid change."""
        affine = grid.affine.tobytes()
        cached = self._protocol
        if cached is None or cached[:3] != (scan, grid.shape, affine):
            cached = (scan, grid.shape, affine, build_protocol(scan, grid))
            self._protocol = cached
        return cached[3]

    def _phantom_of(
        self, state: PhantomState
    ) -> tuple[Phantom, VoxelPartition, list[str]]:
        """Returns the phantom as the items show it, its partition and bundle names.

        The partition is made again only when what fills the voxels changes (the
        grid, the tissues' names, the bundles or the maps), not when the tissues'
        parameters or the scan change.

        Returns:
            The phantom, its partition and the name of each bundle.
        """
        import larmor.simulation as sim

        bundles, maps, names = self.editor.shown_parts(state)
        # Streamlines and map volumes are told apart by identity. The cache keeps
        # them, so that no other object takes the id of one while it is kept.
        data = (*(b[1] for b in bundles), *(m[1] for m in maps))
        ids = tuple(id(d) for d in data)
        if (
            self._phantom is None
            or self._phantom[0] is not state
            or self._phantom[1] != ids
        ):
            self._phantom = (
                state,
                ids,
                data,
                build_phantom(state, bundles, maps),
                names,
            )
        phantom = self._phantom[3]
        key = (
            state.grid.shape,
            state.grid.affine.tobytes(),
            tuple(t.name for t in state.tissues),
            tuple(
                (
                    b.tissue,
                    b.geometry,
                    b.streamline_area,
                    b.densest_fill,
                    None if b.geometry is not None else id(streamlines),
                )
                for b, streamlines, _ in bundles
            ),
            tuple((tissue, id(volume)) for tissue, volume, _ in maps),
        )
        if self._partition is None or self._partition[0] != key:
            self._partition = (key, data, sim.partition_voxels(phantom, n_jobs=1))
        return phantom, self._partition[2], self._phantom[4]
