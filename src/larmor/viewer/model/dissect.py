"""Holds the Dissect workspace's state, and fits FODs and tracks for its steps."""

import json
import secrets
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np

from larmor.types import Tractogram, Volume
from larmor.viewer.model.item import Category, ItemId
from larmor.viewer.model.options import Basis, TractogramOptions
from larmor.viewer.model.phantom import BvecFrame
from larmor.viewer.model.regions import applied_regions, image_of
from larmor.viewer.model.scene import Scene
from larmor.viewer.model.session import FORMAT, VERSION, region_entries, session_data
from larmor.viewer.model.viewer import Viewer

if TYPE_CHECKING:
    from larmor.sd import CSDModel, MSMTCSDModel, SDModel, SDSolver
    from larmor.tractography import Seeds, SeedStrategy, TractographyRun
    from larmor.types import GradientTable

# larmor.sd and larmor.tractography load scipy, so they are imported only in the
# functions that need them.

# Called by a long job with the steps done, their total and the stage; raising
# from it stops the job.
type Report = Callable[[int, int, str], None]


class DissectStep(StrEnum):
    """The steps of the Dissect workspace, in order."""

    START = "start"
    FODS = "fods"
    TRACK = "track"
    REGIONS = "regions"
    SAVE = "save"


DISSECT_STEPS: Final = tuple(DissectStep)


class StartFrom(StrEnum):
    """What a dissection starts from."""

    TRACTOGRAM = "tractogram"  # skips the FODs and Track steps
    FOD = "fod"  # the FODs step only loads it
    DWI = "dwi"  # the FODs step estimates the FODs from it


class FodSource(StrEnum):
    """Where the FODs step takes the FODs from."""

    LOAD = "load"  # a FOD file
    ESTIMATE = "estimate"  # a fit of the DWI


class FodMethod(StrEnum):
    """The spherical deconvolution methods, as `lar fod --method` names them."""

    GRL = "GRL"
    MFOD = "mFOD"
    CSD = "CSD"
    MSMT_CSD = "MSMT-CSD"


class FodSolver(StrEnum):
    """The solvers of GRL and mFOD, as `lar fod --solver` names them."""

    DRL = "dRL"
    L2NNLS = "L2NNLS"
    LASSO = "LASSO"


class GreyMatter(StrEnum):
    """How GRL and mFOD model grey matter."""

    AUTO = "auto"  # NODDI for mFOD, isotropic for GRL
    ISOTROPIC = "isotropic"  # an isotropic response of a given diffusivity
    NODDI = "noddi"  # an anisotropic NODDI response with its own FOD
    NONE = "none"  # no grey matter


class FodMerge(StrEnum):
    """How the two FODs of a fit with a NODDI grey matter are made one."""

    WS = "ws"  # each FOD times its tissue fraction, summed (mFOD-WS)
    MAJORITY = "majority"  # the FOD of the larger anisotropic fraction
    NONE = "none"  # both FODs, as two items


class TrackAlgorithm(StrEnum):
    """The tracking algorithms, as `lar track --algorithm` names them."""

    DET = "det"
    PROB = "prob"


class Seeding(StrEnum):
    """Where the Track step starts streamlines."""

    WHOLE_MASK = "whole_mask"  # random points in the seed mask, up to a number kept
    PER_VOXEL = "per_voxel"  # a number of random points in every voxel of it


# The grey matter diffusivity of GRL by default, mm^2/s.
GM_DIFFUSIVITY: Final = 0.7e-3
# The CSF diffusivity of GRL and mFOD by default, mm^2/s.
CSF_DIFFUSIVITY: Final = 3.0e-3
# The number of streamlines the Track step keeps by default.
STREAMLINES: Final = 100_000


@dataclass(frozen=True)
class DwiFiles:
    """The files of a DWI to estimate FODs from.

    Attributes:
        path: The DWI.
        bvals: The b-values file; None when the DWI's header holds the table.
        bvecs: The b-vectors file, in `frame`; None as for bvals.
        frame: The frame of the bvecs file.
    """

    path: Path
    bvals: Path | None = None
    bvecs: Path | None = None
    frame: BvecFrame = BvecFrame.FSL


# The endings of a DWI file, and those of the gradient files found beside it.
DWI_ENDINGS: Final = (".nii.gz", ".nii", ".mif.gz", ".mif")
GRADIENT_ENDINGS: Final = ((".bval", ".bvec"), (".bvals", ".bvecs"))


def gradient_files(dwi: Path) -> tuple[Path | None, Path | None]:
    """Returns the bvals and bvecs files beside a DWI with its name, if both exist.

    The Start step offers them on the FODs step; they are not read until then.
    """
    name = dwi.name
    for ending in DWI_ENDINGS:
        if name.endswith(ending):
            name = name[: -len(ending)]
            break
    for bvals, bvecs in GRADIENT_ENDINGS:
        pair = dwi.with_name(name + bvals), dwi.with_name(name + bvecs)
        if pair[0].is_file() and pair[1].is_file():
            return pair
    return None, None


def start_items(scene: Scene, start: StartFrom) -> list[ItemId]:
    """Returns the items of the scene a dissection can start from, in scene order.

    A DWI is a 4D volume read from a file, since the fit reads the file again
    with its gradient files.
    """
    if start == StartFrom.TRACTOGRAM:
        return [item.id for item in scene.in_category(Category.TRACTOGRAMS)]
    if start == StartFrom.FOD:
        return [item.id for item in scene.in_category(Category.FODS)]
    return [
        item.id
        for item in scene.in_category(Category.VOLUMES)
        if isinstance(item.data, Volume)
        and item.data.data.ndim == 4
        and item.source is not None
    ]


def item_from_file(scene: Scene, path: Path, category: Category) -> ItemId | None:
    """Returns the item of a category already read from a file; None if none is."""
    path = path.absolute()
    for item in scene.in_category(category):
        if item.source is not None and item.source.absolute() == path:
            return item.id
    return None


@dataclass(frozen=True)
class FodSettings:
    """The choices of a FOD fit, with the defaults of `lar fod`.

    Attributes:
        method: The deconvolution method.
        lmax: The highest SH degree of the FODs.
        solver: The solver of GRL and mFOD; None for the method's own.
        reg: The weight of the L2NNLS and LASSO penalty; None for the solver's own.
        inner_shell_weight: The weight of the shells below the highest in a
            multi-tissue GRL or mFOD fit; None for the method's own.
        grey_matter: How GRL and mFOD model grey matter.
        gm_diffusivity: The diffusivity of an isotropic grey matter, mm^2/s.
        csf_diffusivity: The diffusivity of the CSF, mm^2/s; None for no CSF.
        response: The white matter response file of CSD; None to estimate it.
        wm_response: The white matter response file of MSMT-CSD; None to estimate
            all three.
        gm_response: The grey matter response file of MSMT-CSD.
        csf_response: The CSF response file of MSMT-CSD.
        fod_merge: How the two FODs of a fit with a NODDI grey matter are made
            one; it applies to such fits only.
    """

    method: FodMethod = FodMethod.GRL
    lmax: int = 8
    solver: FodSolver | None = None
    reg: float | None = None
    inner_shell_weight: float | None = None
    grey_matter: GreyMatter = GreyMatter.AUTO
    gm_diffusivity: float = GM_DIFFUSIVITY
    csf_diffusivity: float | None = CSF_DIFFUSIVITY
    response: Path | None = None
    wm_response: Path | None = None
    gm_response: Path | None = None
    csf_response: Path | None = None
    fod_merge: FodMerge = FodMerge.WS


# The methods that fit their own responses with SDModel: GRL and mFOD.
SD_MODEL_METHODS: Final = (FodMethod.GRL, FodMethod.MFOD)
# Each choice of a fit that only some methods take: its name, then the methods.
FOD_OPTION_METHODS: Final[dict[str, tuple[str, tuple[FodMethod, ...]]]] = {
    "solver": ("The solver", SD_MODEL_METHODS),
    "reg": ("The penalty weight", SD_MODEL_METHODS),
    "inner_shell_weight": ("The inner-shell weight", SD_MODEL_METHODS),
    "grey_matter": ("The grey matter model", SD_MODEL_METHODS),
    "gm_diffusivity": ("The grey matter diffusivity", SD_MODEL_METHODS),
    "csf_diffusivity": ("The CSF diffusivity", SD_MODEL_METHODS),
    "response": ("The CSD response", (FodMethod.CSD,)),
    "wm_response": ("The white matter response", (FodMethod.MSMT_CSD,)),
    "gm_response": ("The grey matter response", (FodMethod.MSMT_CSD,)),
    "csf_response": ("The CSF response", (FodMethod.MSMT_CSD,)),
}


def check_fod_settings(settings: FodSettings) -> None:
    """Refuses the choices of a fit that its method does not take.

    A choice left at its default is always accepted.

    Raises:
        ValueError: If a choice other than its default does not apply to the
            method, such as a CSF diffusivity with CSD.
    """
    defaults = FodSettings()
    for name, (label, methods) in FOD_OPTION_METHODS.items():
        if settings.method in methods:
            continue
        if getattr(settings, name) != getattr(defaults, name):
            raise ValueError(f"{label} does not apply to {settings.method.value}.")
    if settings.fod_merge != defaults.fod_merge and not two_fods(settings):
        raise ValueError(
            "Merging FODs applies only to a fit with a NODDI grey matter, which "
            "has two."
        )


@dataclass(frozen=True)
class FodFit:
    """The result of a FOD fit, each map on the grid of the DWI.

    Attributes:
        fods: One FOD per anisotropic response, white matter first, in the
            tournier basis.
        fractions: The tissue fractions of a multi-tissue fit; None otherwise.
    """

    fods: tuple[Volume, ...]
    fractions: Volume | None


def grey_matter(settings: FodSettings) -> GreyMatter:
    """Returns how grey matter is modelled, AUTO resolved: NODDI for mFOD only."""
    if settings.grey_matter != GreyMatter.AUTO:
        return settings.grey_matter
    return (
        GreyMatter.NODDI if settings.method == FodMethod.MFOD else GreyMatter.ISOTROPIC
    )


def two_fods(settings: FodSettings) -> bool:
    """Tells whether a fit has two FODs: GRL or mFOD with a NODDI grey matter."""
    return settings.method in SD_MODEL_METHODS and grey_matter(settings) == (
        GreyMatter.NODDI
    )


def fod_solver(settings: FodSettings) -> SDSolver | None:
    """Returns the solver of a GRL or mFOD fit; None keeps the method's own.

    Raises:
        ValueError: If a penalty weight is given to dRL or to CSD and MSMT-CSD,
            which have their own solvers, or is out of range.
    """
    from larmor.sd import DRL, L2NNLS, Lasso, SDMethod

    solvers: dict[FodSolver, type[SDSolver]] = {
        FodSolver.DRL: DRL,
        FodSolver.L2NNLS: L2NNLS,
        FodSolver.LASSO: Lasso,
    }
    if settings.method in (FodMethod.CSD, FodMethod.MSMT_CSD):
        if settings.solver is not None or settings.reg is not None:
            raise ValueError(
                f"{settings.method} has its own solver; the solver and its "
                "weight do not apply."
            )
        return None
    name, reg = settings.solver, settings.reg
    if reg is None:
        return None if name is None else solvers[name]()
    solver = (
        SDMethod(settings.method.value).default_solver()
        if name is None
        else solvers[name]()
    )
    if isinstance(solver, L2NNLS):
        return L2NNLS(reg=reg)
    if isinstance(solver, Lasso):
        return Lasso(reg=reg)
    raise ValueError("The penalty weight applies to the L2NNLS and LASSO solvers only.")


def fod_model(
    settings: FodSettings, image: Volume, gtab: GradientTable, mask: Volume
) -> CSDModel | SDModel | MSMTCSDModel:
    """Builds the deconvolution model of a fit, its responses estimated if not given.

    The model writes its FODs in the tournier basis.

    Returns:
        The model, ready to fit.

    Raises:
        ValueError: If a choice does not apply to the method, or the choices do
            not go together, such as grey matter or CSF responses of MSMT-CSD
            without the white matter one.
    """
    from larmor.io import read_mrtrix_response
    from larmor.sd import (
        CSDModel,
        MSMTCSDModel,
        SDModel,
        build_csd_response,
        build_response_functions,
        build_shell_response,
    )

    check_fod_settings(settings)
    solver = fod_solver(settings)
    if settings.method == FodMethod.MSMT_CSD:
        given = [
            r
            for r in (settings.wm_response, settings.gm_response, settings.csf_response)
            if r is not None
        ]
        if given and settings.wm_response is None:
            raise ValueError(
                "The grey matter and CSF responses need the white matter response."
            )
        shells = [build_shell_response(read_mrtrix_response(p), gtab) for p in given]
        return MSMTCSDModel(shells or None, lmax=settings.lmax)
    if settings.method == FodMethod.CSD:
        rf = None
        if settings.response is not None:
            rf = build_csd_response(read_mrtrix_response(settings.response), gtab)
        return CSDModel(rf, lmax=settings.lmax)
    gm = grey_matter(settings)
    rfs = build_response_functions(
        image,
        gtab,
        mask,
        gm_diffusivity=settings.gm_diffusivity if gm == GreyMatter.ISOTROPIC else None,
        csf_diffusivity=settings.csf_diffusivity,
        gm_noddi=gm == GreyMatter.NODDI,
    )
    return SDModel(
        rfs=rfs,
        method=settings.method.value,
        solver=solver,
        inner_shell_weight=settings.inner_shell_weight,
        lmax=settings.lmax,
    )


def fit_fods(
    dwi: DwiFiles, mask: Volume, settings: FodSettings, report: Report
) -> FodFit:
    """Reads a DWI and fits its FODs, as `lar fod` does with the same choices.

    The two FODs of a fit with a NODDI grey matter are merged into one unless
    `fod_merge` is NONE.

    Args:
        dwi: The DWI and its gradient files.
        mask: The brain mask, on the DWI's grid.
        settings: The choices of the fit.
        report: Told the progress; raising from it stops the fit.

    Returns:
        The FODs, in the tournier basis, and the tissue fractions if any.

    Raises:
        ValueError: If the files cannot be read or the choices do not go together.
        FileNotFoundError: If a file is missing.
    """
    from larmor.io import load_dwi
    from larmor.sd import merge_fods
    from larmor.sd.progress import fit_progress

    check_fod_settings(settings)
    report(0, 0, "Reading the DWI")
    image, gtab = load_dwi(dwi.path, dwi.bvals, dwi.bvecs, frame=dwi.frame.value)
    report(0, 0, "Estimating the responses")
    model = fod_model(settings, image, gtab, mask)
    with fit_progress(report):
        fit = model.fit(image, gtab, mask=mask)
    fractions = fit.tissue_fractions
    data = fit.fods
    if two_fods(settings) and settings.fod_merge != FodMerge.NONE:
        assert fractions is not None  # a fit with NODDI grey matter has fractions
        majority = settings.fod_merge == FodMerge.MAJORITY
        data = [merge_fods(fit.fods, fractions, "majority" if majority else "ws")]
    fods = tuple(image.with_data(f) for f in data)
    return FodFit(fods, None if fractions is None else image.with_data(fractions))


@dataclass(frozen=True)
class TrackSettings:
    """The choices of a tracking run; None takes the library's default.

    Attributes:
        algorithm: The tracking algorithm.
        streamlines: The number of streamlines to keep, seeding the whole mask.
        step_size: The step size, mm.
        angle: The largest angle between two steps, degrees.
        cutoff: The FOD amplitude below which a streamline stops; None for auto.
        min_length: The shortest streamline kept, mm.
        max_length: The length at which a streamline stops, mm.
        seeding: Where streamlines start.
        seeds_per_voxel: The seeds in each voxel of the seed mask, per voxel.
        seed: The run's random seed; None draws one.
    """

    algorithm: TrackAlgorithm = TrackAlgorithm.DET
    streamlines: int = STREAMLINES
    step_size: float | None = None
    angle: float | None = None
    cutoff: float | None = None
    min_length: float | None = None
    max_length: float | None = None
    seeding: Seeding = Seeding.WHOLE_MASK
    seeds_per_voxel: int = 1
    seed: int | None = None


def fod_support(fod: Volume) -> Volume:
    """Returns the voxels where an FOD's first SH coefficient is positive, as a mask."""
    support = np.asarray(fod.data[..., 0] > 0, dtype=np.uint8)
    return Volume(support, fod.affine)


def track(
    fod: Volume,
    basis: Basis,
    seed_mask: Volume,
    mask: Volume | None,
    settings: TrackSettings,
    report: Report,
) -> TractographyRun:
    """Tracks streamlines through an FOD, as `lar track` does with the same choices.

    Args:
        fod: The FOD.
        basis: The FOD's SH basis.
        seed_mask: Where streamlines start.
        mask: The tracking mask; None for none.
        settings: The choices of the run; its seed is drawn if None.
        report: Told the progress; raising from it stops the run.

    Returns:
        The run's record, its seed and settings resolved, with its streamlines.

    Raises:
        ValueError: If a choice is out of range or no seed lies in the masks.
    """
    from larmor.tractography import MaskSeeds, Tracker, seeds_per_voxel
    from larmor.tractography.progress import track_progress

    seed = settings.seed if settings.seed is not None else secrets.randbits(64)
    tracker = Tracker(
        settings.algorithm.value,
        step_size=settings.step_size,
        max_angle=settings.angle,
        cutoff="auto" if settings.cutoff is None else settings.cutoff,
        min_length=settings.min_length,
        max_length=settings.max_length,
    )
    report(0, 0, "Preparing")
    seeds: Seeds | SeedStrategy
    if settings.seeding == Seeding.PER_VOXEL:
        seeds = seeds_per_voxel(seed_mask, settings.seeds_per_voxel, seed)
        target = None
    else:
        seeds, target = MaskSeeds(seed_mask), settings.streamlines
    with track_progress(report):
        return tracker.track(
            fod,
            None,
            seeds,
            num_streamlines=target,
            mask=mask,
            basis=basis.value,
            seed=seed,
        )


@dataclass(frozen=True)
class TrackedRun:
    """A tracking run of the Track step, with the items it tracked from.

    Attributes:
        run: The library's record of the run.
        fod: The FOD item.
        seed_mask: The seed mask item; None for the FOD's support.
        mask: The tracking mask item; None for none.
        seeding: Where streamlines started.
        seeds_per_voxel: The seeds in each voxel, seeding per voxel.
        support: The FOD's support, the seed mask when no item was.
        files: The file each input item showed when tracked, for those that had
            one; the run record names it once the item is removed.
    """

    run: TractographyRun
    fod: ItemId
    seed_mask: ItemId | None
    mask: ItemId | None
    seeding: Seeding
    seeds_per_voxel: int
    support: Volume | None = None
    files: Mapping[ItemId, Path] = field(default_factory=dict)

    @property
    def inputs(self) -> list[ItemId]:
        """Returns the input items: the FOD, then the seed and tracking masks."""
        return [i for i in (self.fod, self.seed_mask, self.mask) if i is not None]


def input_files(tracked: TrackedRun, scene: Scene) -> dict[ItemId, Path | None]:
    """Returns the file of each input item of a run; None for one without a file.

    An item still in the scene gives the file it shows now; a removed one gives
    the file it showed when tracked.
    """
    present = {item.id: item.source for item in scene.items}
    return {
        i: present[i] if i in present else tracked.files.get(i) for i in tracked.inputs
    }


def run_inputs(
    tracked: TrackedRun, paths: Mapping[ItemId, Path], support: Path | None
) -> dict[str, object]:
    """Returns the input files of a run as `lar track --json` records them.

    Args:
        tracked: The run.
        paths: The file of each item the run tracked from.
        support: The file of the FOD's support, when it was the seed mask.
    """
    seeds = support if tracked.seed_mask is None else paths[tracked.seed_mask]
    per_voxel = tracked.seeding == Seeding.PER_VOXEL
    return {
        "fod": str(paths[tracked.fod].resolve()),
        "mask": None if tracked.mask is None else str(paths[tracked.mask].resolve()),
        "seed_image": None if per_voxel or seeds is None else str(seeds.resolve()),
        "seed_random_per_voxel": [str(seeds.resolve()), tracked.seeds_per_voxel]
        if per_voxel and seeds is not None
        else None,
        "seed_file": None,
    }


@dataclass
class Dissection:
    """The state of the Dissect workspace's steps, by the items they made or chose.

    Attributes:
        start: What the dissection started from; None before the Start step.
        dwi: The DWI to estimate FODs from.
        fod_source: Where the FODs step takes the FODs from.
        fod: The FOD item the Track step tracks.
        fit_mask: The mask item of the FOD fit.
        tractogram: The tractogram item the regions dissect.
        runs: The run of each tractogram the Track step made.
        made: The items the steps made, which may need saving.
        saved: The bundle file of the last save; None before one.
    """

    start: StartFrom | None = None
    dwi: DwiFiles | None = None
    fod_source: FodSource = FodSource.ESTIMATE
    fod: ItemId | None = None
    fit_mask: ItemId | None = None
    tractogram: ItemId | None = None
    runs: dict[ItemId, TrackedRun] = field(default_factory=dict)
    made: list[ItemId] = field(default_factory=list)
    saved: Path | None = None

    def begin(self, start: StartFrom, dwi: DwiFiles | None = None) -> None:
        """Starts a new dissection, forgetting the items of the last one."""
        self.start, self.dwi = start, dwi
        self.fod_source = FodSource.ESTIMATE if dwi is not None else FodSource.LOAD
        self.fod = self.fit_mask = self.tractogram = None
        self.saved = None

    def adopt(self, item_id: ItemId) -> None:
        """Counts an item among those the dissection made, such as a new mask."""
        if item_id not in self.made:
            self.made.append(item_id)

    def skipped(self, step: DissectStep) -> bool:
        """Tells whether a step is skipped: FODs and Track, from a tractogram."""
        return self.start == StartFrom.TRACTOGRAM and step in (
            DissectStep.FODS,
            DissectStep.TRACK,
        )

    def forget_missing(self, scene: Scene) -> None:
        """Forgets the items no longer in the scene."""
        present = {item.id for item in scene.items}
        if self.fod not in present:
            self.fod = None
        if self.fit_mask not in present:
            self.fit_mask = None
        if self.tractogram not in present:
            self.tractogram = None
        self.runs = {i: r for i, r in self.runs.items() if i in present}
        self.made = [i for i in self.made if i in present]

    def unsaved(self, scene: Scene) -> list[ItemId]:
        """Returns the items the steps made or use that have no file, in scene order.

        These are the items made (FODs, tissue fractions, tractograms, masks),
        the masks of the fit and the runs, and the images of the mask and label
        regions.
        """
        wanted = set(self.made)
        wanted |= {
            i for i in (self.fit_mask, self.fod, self.tractogram) if i is not None
        }
        for tracked in self.runs.values():
            wanted |= {i for i in (tracked.seed_mask, tracked.mask) if i is not None}
        wanted |= set(region_images(scene))
        return [
            item.id for item in scene.items if item.id in wanted and item.source is None
        ]


@dataclass(frozen=True)
class DissectStatus:
    """Where a step of the Dissect workspace stands.

    Attributes:
        step: The step.
        ready: Whether the step is done.
        skipped: Whether the step is skipped.
        text: A short line on the step, such as "12,000 streamlines".
    """

    step: DissectStep
    ready: bool
    skipped: bool
    text: str


# How the Start step names each start.
START_TEXT: Final = {
    StartFrom.TRACTOGRAM: "From a tractogram",
    StartFrom.FOD: "From a FOD file",
    StartFrom.DWI: "From a DWI",
}


def dissect_status(state: Dissection, scene: Scene) -> list[DissectStatus]:
    """Returns where each step of the Dissect workspace stands, in order."""
    skipped = "Skipped: started from a tractogram"
    rows = [
        DissectStatus(
            DissectStep.START,
            state.start is not None,
            False,
            START_TEXT[state.start] if state.start else "Choose what you have",
        )
    ]
    present = {item.id for item in scene.items}
    fod = state.fod if state.fod in present else None
    tract = state.tractogram if state.tractogram in present else None
    if state.skipped(DissectStep.FODS):
        rows.append(DissectStatus(DissectStep.FODS, False, True, skipped))
        rows.append(DissectStatus(DissectStep.TRACK, False, True, skipped))
    else:
        fod_text = scene.item(fod).name if fod is not None else "Needs FODs"
        rows.append(DissectStatus(DissectStep.FODS, fod is not None, False, fod_text))
        tracked = tract is not None and tract in state.runs
        track_text = "Needs a tracking run"
        if tract is not None:
            data = scene.item(tract).data
            assert isinstance(data, Tractogram)
            track_text = f"{data.num_streamlines:,} streamlines"
        rows.append(
            DissectStatus(
                DissectStep.TRACK, tracked or tract is not None, False, track_text
            )
        )
    regions = [] if tract is None else applied_regions(scene, scene.item(tract))
    if tract is None:
        region_text = "Needs a tractogram"
    elif not regions:
        region_text = "No region yet"
    else:
        region_text = f"{len(regions)} region{'s' if len(regions) > 1 else ''}"
    rows.append(DissectStatus(DissectStep.REGIONS, bool(regions), False, region_text))
    saved = state.saved
    rows.append(
        DissectStatus(
            DissectStep.SAVE,
            saved is not None,
            False,
            f"Saved as {saved.name}" if saved is not None else "Not saved yet",
        )
    )
    return rows


def region_images(scene: Scene) -> list[ItemId]:
    """Returns the masks and parcellations the scene's regions name."""
    ids: list[ItemId] = []
    for region in scene.in_category(Category.REGIONS):
        image = image_of(region)
        if image is not None and image not in ids:
            ids.append(image)
    return ids


# The suffix of a regions file, a session file that holds only regions.
REGIONS_SUFFIX: Final = ".regions.json"


def regions_data(scene: Scene, tract_id: ItemId, folder: Path) -> dict[str, object]:
    """Returns the regions a tractogram applies, as a regions file holds them.

    The file is a session file with the regions and the images they name, and
    the tractogram's region filter, but no view state and no other item.

    Args:
        scene: The scene.
        tract_id: The tractogram.
        folder: The folder of the regions file, which relative paths start from.

    Raises:
        ValueError: If an image a region names has no file.
    """
    tract = scene.item(tract_id)
    applied = applied_regions(scene, tract)
    regions = {r.id for r in applied}
    entries = [e for e in region_entries(scene) if e["id"] in regions]
    images = {i for i in map(image_of, applied) if i is not None}
    for image in images:
        if scene.item(image).source is None:
            raise ValueError(f"{scene.item(image).name} has no file: save it first.")
    data, _ = session_data(scene, folder)
    items = data["items"]
    assert isinstance(items, list)
    o = tract.options
    assert isinstance(o, TractogramOptions)
    return {
        "format": FORMAT,
        "version": VERSION,
        "items": [e for e in items if e["id"] in images],
        "regions": entries,
        "view": {},
        "filter": {"region_filter": o.region_filter, "ends_only": o.ends_only},
    }


def write_regions(scene: Scene, tract_id: ItemId, path: Path) -> None:
    """Writes the regions a tractogram applies to a regions file.

    Raises:
        ValueError: If an image a region names has no file.
        OSError: If the file cannot be written.
    """
    path = Path(path).absolute()
    data = regions_data(scene, tract_id, path.parent)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", "utf-8")


@dataclass(frozen=True)
class SavePlan:
    """The files the Save step writes.

    Attributes:
        bundle: The kept streamlines.
        regions: The regions file.
        run: The run record; None when the tractogram was not tracked here.
        items: The file of each unsaved item to write, chosen by the user.
        support: The file of the FOD's support, when a run seeded from it.
        notes: Sentences on what the save leaves out, such as the run record
            when an input of the run was removed without a file.
    """

    bundle: Path
    regions: Path
    run: Path | None
    items: Mapping[ItemId, Path]
    support: Path | None = None
    notes: tuple[str, ...] = ()

    @property
    def paths(self) -> list[Path]:
        """Returns every file the plan writes."""
        found = [self.bundle, self.regions, *self.items.values()]
        return found + [p for p in (self.run, self.support) if p is not None]


# The extension each category of item is saved with.
ITEM_SUFFIX: Final = {Category.TRACTOGRAMS: ".tck"}


def item_file(scene: Scene, item_id: ItemId, folder: Path, name: str) -> Path:
    """Returns the file an unsaved item of a dissection is saved to."""
    item = scene.item(item_id)
    stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in item.name)
    return folder / f"{name}_{stem}{ITEM_SUFFIX.get(item.category, '.nii.gz')}"


def unique_path(path: Path, taken: set[Path]) -> Path:
    """Returns the path, or the first of its `_2`, `_3`, ... forms not in taken."""
    suffix = "".join(path.suffixes)
    stem = path.name[: len(path.name) - len(suffix)] if suffix else path.name
    number, found = 1, path
    while found in taken:
        number += 1
        found = path.with_name(f"{stem}_{number}{suffix}")
    return found


def plan_save(
    state: Dissection,
    scene: Scene,
    folder: Path,
    name: str,
    items: list[ItemId],
) -> SavePlan:
    """Names the files a save of the dissection writes into a folder.

    Args:
        state: The dissection; its tractogram must be set.
        scene: The scene.
        folder: The folder to write into.
        name: The start of every file's name, such as the bundle's name.
        items: The unsaved items to save beside them.

    Returns:
        The files to write, with the items the bundle needs added to `items`.

    Raises:
        ValueError: If there is no tractogram, or the name is empty.
    """
    if state.tractogram is None:
        raise ValueError("There is no tractogram to save a bundle of.")
    if not name.strip():
        raise ValueError("Give the bundle a name.")
    folder, name = Path(folder).absolute(), name.strip()
    tracked = state.runs.get(state.tractogram)
    present = {item.id for item in scene.items}
    needed, notes = [], []
    if tracked is not None:
        files = input_files(tracked, scene)
        lost = [i for i, f in files.items() if f is None and i not in present]
        if lost:
            notes.append(
                "The run record is not written, as an input of the tracking run "
                "was removed before it was saved."
            )
            tracked = None
        else:
            needed = [i for i, f in files.items() if f is None]
    chosen = list(dict.fromkeys([*items, *needed, *_unsaved_images(state, scene)]))
    support = None
    if tracked is not None and tracked.seed_mask is None:
        support = folder / f"{name}_seeds.nii.gz"
    bundle = folder / f"{name}.tck"
    regions = folder / f"{name}{REGIONS_SUFFIX}"
    run = folder / f"{name}_run.json" if tracked is not None else None
    taken = {p for p in (bundle, regions, run, support) if p is not None}
    paths = {}
    for i in chosen:
        paths[i] = unique_path(item_file(scene, i, folder, name), taken)
        taken.add(paths[i])
    return SavePlan(bundle, regions, run, paths, support, tuple(notes))


def _unsaved_images(state: Dissection, scene: Scene) -> list[ItemId]:
    """Returns the images the tractogram's regions name that have no file."""
    if state.tractogram is None:
        return []
    found = []
    for region in applied_regions(scene, scene.item(state.tractogram)):
        image = image_of(region)
        if image is not None and scene.item(image).source is None:
            found.append(image)
    return found


def tracked_run(tracked: TrackedRun, inputs: dict[str, object]) -> TractographyRun:
    """Returns a run's record with its input files, as the Save step writes it."""
    return replace(tracked.run, inputs=inputs)


def write_save(viewer: Viewer, state: Dissection, plan: SavePlan) -> None:
    """Writes the files of a save plan, its unsaved items first.

    The items then refer to their new files, so the regions file and the run
    record can name them.

    Raises:
        ValueError: If an item cannot be saved, or a region's image has no file.
        OSError: If a file cannot be written.

    Either error lists the files written before it.
    """
    assert state.tractogram is not None
    written: list[Path] = []
    try:
        _write_plan(viewer, state, state.tractogram, plan, written)
    except (OSError, ValueError) as e:
        if not written:
            raise
        names = ", ".join(p.name for p in written)
        kind = OSError if isinstance(e, OSError) else ValueError
        raise kind(f"{e} These files were written already: {names}.") from e
    state.saved = plan.bundle


def _write_plan(
    viewer: Viewer,
    state: Dissection,
    tract_id: ItemId,
    plan: SavePlan,
    written: list[Path],
) -> None:
    """Writes the files of a save plan in order, adding each one to written."""
    from larmor.io import save_volume

    for item_id, path in plan.items.items():
        viewer.save_item(item_id, path)
        written.append(path)
    tracked = state.runs.get(tract_id)
    if plan.support is not None and tracked is not None:
        assert tracked.support is not None
        save_volume(tracked.support, plan.support)
        written.append(plan.support)
    viewer.save_kept(tract_id, plan.bundle)
    written.append(plan.bundle)
    write_regions(viewer.scene, tract_id, plan.regions)
    written.append(plan.regions)
    if plan.run is not None and tracked is not None:
        paths: dict[ItemId, Path] = {}
        for used, source in input_files(tracked, viewer.scene).items():
            if source is None:
                raise ValueError("An input of the tracking run has no file.")
            paths[used] = source
        record = tracked_run(tracked, run_inputs(tracked, paths, plan.support))
        record.write_json(plan.run)
        written.append(plan.run)
