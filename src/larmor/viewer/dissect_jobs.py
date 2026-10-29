"""Fits FODs and tracks for the Dissect workspace on the job runner."""

from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

from larmor.types import Volume
from larmor.viewer.jobs import JobRunner, Report
from larmor.viewer.model.dissect import (
    Dissection,
    DwiFiles,
    FodFit,
    FodSettings,
    TrackedRun,
    TrackSettings,
    fit_fods,
    fod_support,
    track,
)
from larmor.viewer.model.item import Category, Item, ItemId
from larmor.viewer.model.options import Basis, FodOptions, OptionName
from larmor.viewer.model.viewer import Viewer

if TYPE_CHECKING:
    from larmor.tractography import TractographyRun

# The names of the jobs, as the job bar and the step pages show them.
FIT_JOB = "Estimating FODs"
TRACK_JOB = "Tracking"


def start_fod_fit(
    viewer: Viewer,
    jobs: JobRunner,
    state: Dissection,
    dwi: DwiFiles,
    mask_id: ItemId,
    settings: FodSettings,
    *,
    on_done: Callable[[list[Item]], None] | None = None,
    on_failed: Callable[[str], None] | None = None,
) -> None:
    """Fits FODs to a DWI in the background, then adds them and the fractions.

    The first FOD becomes the one the Track step tracks; the items count as made
    by the dissection, so the Save step offers to save them.

    Args:
        viewer: The viewer to add the items to.
        jobs: The job runner.
        state: The dissection.
        dwi: The DWI and its gradient files.
        mask_id: The image item whose non-zero voxels are fitted.
        settings: The choices of the fit.
        on_done: Told the items added.
        on_failed: Told why the fit failed; the runner's error box if None.

    Raises:
        ValueError: If the mask is not a 3D image.
        JobBusyError: If a job runs already.
    """
    mask = viewer.scene.item(mask_id).data
    if not isinstance(mask, Volume) or mask.data.ndim != 3:
        raise ValueError("The mask must be a 3D image.")
    name = Path(dwi.path).name.split(".")[0] or "dwi"

    def done(fit: FodFit) -> None:
        """Adds the FODs and the fractions, and records them in the dissection."""
        added = []
        many = len(fit.fods) > 1
        for number, fod in enumerate(fit.fods, 1):
            label = f"{name}_fod{number}" if many else f"{name}_fod"
            item = viewer.add(fod, Category.FODS, name=label)
            viewer.set(item.id, OptionName.BASIS, Basis.TOURNIER)
            added.append(item)
        if fit.fractions is not None:
            label = f"{name}_tissue_fraction"
            added.append(viewer.add(fit.fractions, Category.VOLUMES, name=label))
        for item in added:
            state.adopt(item.id)
        state.fod, state.fit_mask = added[0].id, mask_id
        if on_done is not None:
            on_done(added)

    work = partial(fit_fods, dwi, mask, settings)
    jobs.start(FIT_JOB, work, done, on_failed)


def start_tracking(
    viewer: Viewer,
    jobs: JobRunner,
    state: Dissection,
    fod_id: ItemId,
    seed_mask_id: ItemId | None,
    mask_id: ItemId | None,
    settings: TrackSettings,
    *,
    on_done: Callable[[Item], None] | None = None,
    on_failed: Callable[[str], None] | None = None,
) -> None:
    """Tracks through an FOD item in the background, then adds the tractogram.

    The run's record stays with the tractogram in the dissection, for the Save
    step.

    Args:
        viewer: The viewer to add the tractogram to.
        jobs: The job runner.
        state: The dissection.
        fod_id: The FOD item.
        seed_mask_id: The image item streamlines start in; None for the voxels
            where the FOD's first coefficient is positive.
        mask_id: The tracking mask item; None for none.
        settings: The choices of the run.
        on_done: Told the tractogram item.
        on_failed: Told why the run failed; the runner's error box if None.

    Raises:
        ValueError: If an item is not an FOD or a 3D image.
        JobBusyError: If a job runs already.
    """
    fod_item = viewer.scene.item(fod_id)
    fod = fod_item.data
    options = fod_item.options
    if not isinstance(fod, Volume) or not isinstance(options, FodOptions):
        raise ValueError(f"{fod_item.name} is not an FOD.")
    support = None
    if seed_mask_id is None:
        seed_mask = support = fod_support(fod)
    else:
        seed_mask = _image(viewer, seed_mask_id)
    mask = None if mask_id is None else _image(viewer, mask_id)
    files: dict[ItemId, Path] = {}
    for used in (fod_id, seed_mask_id, mask_id):
        source = None if used is None else viewer.scene.item(used).source
        if used is not None and source is not None:
            files[used] = source

    def work(report: Report) -> TractographyRun:
        """Tracks the streamlines."""
        return track(fod, options.basis, seed_mask, mask, settings, report)

    def done(run: TractographyRun) -> None:
        """Adds the tractogram and keeps the run's record with it."""
        assert run.tractogram is not None
        item = viewer.add(run.tractogram, Category.TRACTOGRAMS, name="whole_brain")
        state.runs[item.id] = TrackedRun(
            run,
            fod_id,
            seed_mask_id,
            mask_id,
            settings.seeding,
            settings.seeds_per_voxel,
            support,
            files,
        )
        state.adopt(item.id)
        state.tractogram = item.id
        if on_done is not None:
            on_done(item)

    jobs.start(TRACK_JOB, work, done, on_failed)


def _image(viewer: Viewer, item_id: ItemId) -> Volume:
    """Returns a 3D image item's volume; raises ValueError for another item."""
    item = viewer.scene.item(item_id)
    if not isinstance(item.data, Volume) or item.data.data.ndim != 3:
        raise ValueError(f"{item.name} is not a 3D image.")
    return item.data
