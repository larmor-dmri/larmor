"""Saving a simulation with its ground truth, and reading the ground truth back."""

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Final

import numpy as np

from larmor.io import (
    load_tractogram,
    load_volume,
    save_tractogram,
    save_volume,
    write_bvals_bvecs,
)
from larmor.simulation.bundles import BundleGeometry
from larmor.simulation.ground_truth import (
    PEAK_MERGE_ANGLE,
    PEAK_RELATIVE_THRESHOLD,
    GroundTruth,
)
from larmor.simulation.instant import SimulatedDWI, relaxation_weights
from larmor.simulation.phantom import (
    Bundle,
    Phantom,
    Progress,
)
from larmor.simulation.phantom_file import read_phantom, write_phantom
from larmor.simulation.protocol import Protocol
from larmor.simulation.tissues import CompartmentModel
from larmor.types import Grid, Tractogram, Volume

# The format name and version written to simulation.json.
FORMAT: Final = "larmor-simulation"
VERSION: Final = 1

# A tissue mask holds the voxels where the tissue fills at least this fraction.
MASK_FRACTION: Final = 0.5


@dataclass(frozen=True)
class SimulationFiles:
    """The files `save_simulation` writes, all inside one folder.

    Attributes:
        folder: The output folder.
        dwi: The DWI, one 4D NIfTI per echo time.
        bvals: The b-values, shared by the echoes (FSL format).
        bvecs: The b-vectors in the FSL frame, shared by the echoes.
        noise_free: The noise-free DWI per echo; empty if the simulation had no
            noise.
        tissue_fractions: 4D, the fraction of each tissue in each voxel, in the
            order of the phantom's tissues.
        background_fraction: 3D, the fraction of the voxel that no tissue fills.
        free_water_fraction: 3D, the volume fraction of each voxel that the
            FREE_WATER compartments fill, the background included.
        bundle_fractions: 4D, the fraction of each voxel that each bundle fills;
            None if the phantom has no bundle.
        peaks: The ground-truth fibre directions, in the layout `lar peaks`
            writes. Each peak takes 3 volumes, its direction scaled by its
            amplitude, the share of the voxel its fibres fill. The peaks are
            sorted largest first, with NaN where a voxel has fewer peaks. A
            bundle whose tissue has no fibre direction gives no peaks.
        compartment_fractions: 4D, the volume fraction of each voxel that each
            compartment fills, the background counted in its tissue. The volumes
            are in the order of the tissues and, within a tissue, of its
            compartments, as listed in simulation.json.
        signal_fractions: One 4D file per echo time, with each compartment's
            share of the voxel's signal at b = 0, in the order of
            `compartment_fractions`; 0 where the voxel has no signal. A model
            fitted to the images (free-water imaging, NODDI, the standard model)
            recovers these fractions, not the volume fractions.
        free_water_signal_fraction: One 3D file per echo time, with the share of
            the voxel's signal at b = 0 that comes from free water.
        masks: One 3D mask per tissue, by tissue name.
        bundle_mask: One 3D mask of the voxels the bundles fill together; None
            if the phantom has no bundle.
        bundles: One .tck per bundle, in the order of the phantom's bundles.
        tissue_maps: A copy of each tissue map of the phantom, by tissue name.
        phantom: The phantom file, with the scan, which `lar simulate` can run
            again.
        info: simulation.json, which records the protocol, the noise, the seed
            and what each file holds.
    """

    folder: Path
    dwi: tuple[Path, ...]
    bvals: Path
    bvecs: Path
    noise_free: tuple[Path, ...]
    tissue_fractions: Path
    background_fraction: Path
    free_water_fraction: Path
    bundle_fractions: Path | None
    compartment_fractions: Path
    signal_fractions: tuple[Path, ...]
    free_water_signal_fraction: tuple[Path, ...]
    peaks: Path
    masks: dict[str, Path]
    bundle_mask: Path | None
    bundles: tuple[Path, ...]
    tissue_maps: dict[str, Path]
    phantom: Path
    info: Path

    def all_paths(self) -> list[Path]:
        """Returns every file of the folder, in the order they are written."""
        found = [*self.dwi, self.bvals, self.bvecs, *self.noise_free]
        found += [
            self.tissue_fractions,
            self.background_fraction,
            self.free_water_fraction,
        ]
        if self.bundle_fractions is not None:
            found.append(self.bundle_fractions)
        found.append(self.compartment_fractions)
        found += [*self.signal_fractions, *self.free_water_signal_fraction]
        found += [self.peaks, *self.masks.values()]
        if self.bundle_mask is not None:
            found.append(self.bundle_mask)
        found += list(self.bundles)
        return [*found, *self.tissue_maps.values(), self.phantom, self.info]


def simulation_files(
    phantom: Phantom, protocol: Protocol, folder: str | Path, *, noise: bool
) -> SimulationFiles:
    """Returns the files a simulation of this phantom and protocol is saved as.

    Args:
        phantom: The phantom.
        protocol: The scan; its echo times name the DWI files.
        folder: The output folder.
        noise: Whether the simulation has noise, which adds the noise-free DWI.

    Returns:
        The paths; nothing is written.

    Raises:
        ValueError: If two tissues have names that give the same file name, or a
            tissue's mask would take the file of the bundles' mask.
    """
    folder = Path(folder)
    names = [_file_name(t.name) for t in phantom.tissues]
    # File systems on macOS and Windows take names that differ by case as one name.
    folded = [name.lower() for name in names]
    if len(set(folded)) != len(names):
        raise ValueError(
            f"Tissue names {[t.name for t in phantom.tissues]} give the same file "
            "names; rename a tissue."
        )
    if phantom.bundles and "bundles" in folded:
        raise ValueError(
            'A tissue named "bundles" would share its mask file with the bundles; '
            "rename the tissue."
        )
    echoes = [""] if protocol.num_echoes == 1 else _echo_names(protocol.te_ms)
    stems = [f"dwi{e}" for e in echoes]
    return SimulationFiles(
        folder=folder,
        dwi=tuple(folder / f"{stem}.nii.gz" for stem in stems),
        bvals=folder / "dwi.bval",
        bvecs=folder / "dwi.bvec",
        noise_free=(
            tuple(folder / f"{s}_noise_free.nii.gz" for s in stems) if noise else ()
        ),
        tissue_fractions=folder / "tissue_fractions.nii.gz",
        background_fraction=folder / "background_fraction.nii.gz",
        free_water_fraction=folder / "free_water_fraction.nii.gz",
        bundle_fractions=(
            folder / "bundle_fractions.nii.gz" if phantom.bundles else None
        ),
        compartment_fractions=folder / "compartment_fractions.nii.gz",
        signal_fractions=tuple(folder / f"signal_fractions{e}.nii.gz" for e in echoes),
        free_water_signal_fraction=tuple(
            folder / f"free_water_signal_fraction{e}.nii.gz" for e in echoes
        ),
        peaks=folder / "peaks.nii.gz",
        masks={
            t.name: folder / f"mask_{n}.nii.gz"
            for t, n in zip(phantom.tissues, names, strict=True)
        },
        bundle_mask=folder / "mask_bundles.nii.gz" if phantom.bundles else None,
        bundles=tuple(
            folder / f"bundle_{i}.tck" for i in range(1, len(phantom.bundles) + 1)
        ),
        tissue_maps={
            m.tissue: folder / f"tissue_map_{_file_name(m.tissue)}.nii.gz"
            for m in phantom.tissue_maps
        },
        phantom=folder / "phantom.json",
        info=folder / "simulation.json",
    )


def _echo_names(te_ms: Sequence[float]) -> list[str]:
    """Returns the part of each echo's file names, such as "_te80ms", one per TE.

    Echo times that would give the same name get more digits.
    """
    names: list[str] = []
    for digits in (6, 10, 17):
        names = [f"_te{te:.{digits}g}ms" for te in te_ms]
        if len(set(names)) == len(names):
            break
    return names


def save_simulation(
    simulated: SimulatedDWI,
    truth: GroundTruth,
    folder: str | Path,
    *,
    force: bool = False,
    progress: Progress | None = None,
) -> SimulationFiles:
    """Writes a simulation and its ground truth into one self-contained folder.

    The folder holds the DWI with its .bval and .bvec in the FSL frame, the
    ground truth, the bundles' streamlines, the phantom file with the scan and
    simulation.json, as `SimulationFiles` lists them. Running the phantom file
    again with the seed recorded in simulation.json gives the same images, and
    `read_ground_truth` reads the ground truth back.

    The folder is created if needed. If it holds an earlier simulation, the files
    that its simulation.json lists are removed first. A folder that holds other
    files is refused unless `force` is True, which then writes into it and keeps
    them.

    The ground-truth peaks come from the voxel's own fibres, not from a fit. A
    fibre within 20 degrees of a larger one of the same bundle joins its peak,
    a direction and its opposite counting as the same; different bundles never
    share a peak. Peaks below 10% of the voxel's largest are dropped, and each
    peak's direction is the weighted mean of its fibres. A tissue mask holds
    the voxels where the tissue, background included, fills at least half.

    The tissue, bundle, compartment and free-water fractions are volume
    fractions. A model fitted to the images recovers signal fractions, which
    differ because each compartment's signal is weighted by its proton density
    and relaxation: at TE 80 ms and TR 3000 ms, the NODDI preset's free-water
    volume fraction of 0.05 is about 0.11 of the b = 0 signal. Compare a fit
    with the signal fractions, which are written too.

    Args:
        simulated: The images `simulate` returns.
        truth: The ground truth `simulate` returns with them; its bundle names
            go in the phantom file's viewer block.
        folder: The output folder.
        force: Whether to write into a folder that is not empty and holds no
            simulation.json. Defaults to False.
        progress: Called with (done, total, "save") as the files are written.
            Defaults to None.

    Returns:
        The paths written.

    Raises:
        FileExistsError: If the folder is not empty, has no simulation.json and
            `force` is False.
        ValueError: If `simulated` does not carry its protocol (it was not made by
            `simulate`), or two tissue names give the same file name.
    """
    phantom, protocol = truth.phantom, simulated.protocol
    if protocol is None:
        raise ValueError("The simulation has no protocol; save the result of simulate.")
    files = simulation_files(
        phantom, protocol, folder, noise=simulated.sigma is not None
    )
    _clear_previous(files.folder, force)
    files.folder.mkdir(parents=True, exist_ok=True)
    steps = len(files.dwi) + len(files.noise_free) + 4
    done = 0

    def tick() -> None:
        """Tells progress that one more step of the saving is done."""
        nonlocal done
        done += 1
        if progress is not None:
            progress(done, steps, "save")

    for echo, path in zip(simulated.echoes, files.dwi, strict=True):
        save_volume(echo, path)
        tick()
    for echo, path in zip(simulated.noise_free, files.noise_free, strict=False):
        save_volume(echo, path)
        tick()
    write_bvals_bvecs(
        simulated.gradients, files.bvals, files.bvecs, ref_path=files.dwi[0]
    )

    _write_ground_truth(truth, protocol, files, tick)

    for streamlines, path in zip(truth.streamlines, files.bundles, strict=True):
        save_tractogram(Tractogram(streamlines.points, streamlines.offsets), path)
    for tissue_map in phantom.tissue_maps:
        save_volume(tissue_map.volume, files.tissue_maps[tissue_map.tissue])
    tick()

    _write_phantom_copy(phantom, protocol, files, truth.bundle_names)
    _write_info(simulated, truth, protocol, files)
    tick()
    return files


def _write_ground_truth(
    truth: GroundTruth,
    protocol: Protocol,
    files: SimulationFiles,
    tick: Callable[[], None],
) -> None:
    """Writes the fractions, peaks, masks and compartment fractions.

    `tick` is called once after the fractions and peaks, and once at the end.
    """
    affine = truth.grid.affine
    save_volume(_float_volume(truth.tissue_fractions, affine), files.tissue_fractions)
    save_volume(_float_volume(truth.background, affine), files.background_fraction)
    if files.bundle_fractions is not None:
        save_volume(
            _float_volume(truth.bundle_fractions, affine), files.bundle_fractions
        )
    save_volume(_float_volume(truth.peaks, affine), files.peaks)
    tick()

    fractions = truth.tissue_fractions.copy()
    if truth.phantom.background is not None:
        column = truth.tissue_names.index(truth.phantom.background)
        fractions[..., column] += truth.background
    _write_masks(truth, fractions, files)
    _write_compartment_fractions(truth.phantom, protocol, fractions, affine, files)
    tick()


def _float_volume(data: np.ndarray, affine: np.ndarray) -> Volume:
    """Returns the data as a float32 volume with a copy of the affine."""
    return Volume(data.astype(np.float32), affine.copy())


def _write_masks(
    truth: GroundTruth,
    fractions: np.ndarray,
    files: SimulationFiles,
) -> None:
    """Writes the mask of each tissue and the mask of the bundles.

    `fractions` holds each tissue's fraction with the background added to its
    tissue.
    """
    affine = truth.grid.affine
    for index, name in enumerate(truth.tissue_names):
        mask = fractions[..., index] >= MASK_FRACTION
        save_volume(Volume(mask.astype(np.uint8), affine.copy()), files.masks[name])
    if files.bundle_mask is not None:
        union = truth.bundle_fractions.sum(axis=-1) >= MASK_FRACTION
        save_volume(Volume(union.astype(np.uint8), affine.copy()), files.bundle_mask)


def _write_compartment_fractions(
    phantom: Phantom,
    protocol: Protocol,
    fractions: np.ndarray,
    affine: np.ndarray,
    files: SimulationFiles,
) -> None:
    """Writes the compartment and free-water fractions, by volume and by signal."""
    shares = [
        fractions[..., index] * c.fraction
        for index, tissue in enumerate(phantom.tissues)
        for c in tissue.compartments
    ]
    save_volume(
        _float_volume(np.stack(shares, axis=-1), affine), files.compartment_fractions
    )
    free_water = [
        share
        for share, c in zip(
            shares, (c for t in phantom.tissues for c in t.compartments), strict=True
        )
        if c.model == CompartmentModel.FREE_WATER
    ]
    save_volume(
        _float_volume(
            np.sum(free_water, axis=0) if free_water else 0 * fractions[..., 0], affine
        ),
        files.free_water_fraction,
    )
    is_free = np.array(
        [
            c.model == CompartmentModel.FREE_WATER
            for t in phantom.tissues
            for c in t.compartments
        ]
    )
    weights = np.concatenate([relaxation_weights(t, protocol) for t in phantom.tissues])
    volume_shares = np.stack(shares, axis=-1)
    for e, (path, water_path) in enumerate(
        zip(files.signal_fractions, files.free_water_signal_fraction, strict=True)
    ):
        weighted = volume_shares * weights[:, e]
        total = weighted.sum(axis=-1, keepdims=True)
        signal = np.divide(
            weighted, total, out=np.zeros_like(weighted), where=total > 0
        )
        save_volume(_float_volume(signal, affine), path)
        save_volume(
            _float_volume(signal[..., is_free].sum(axis=-1), affine), water_path
        )


def _larmor_version() -> str:
    """Returns the installed version of Larmor."""
    try:
        return version("larmor-dmri")
    except PackageNotFoundError:
        return "unknown"


def _clear_previous(folder: Path, force: bool) -> None:
    """Removes the files an earlier simulation in the folder lists.

    Raises:
        FileExistsError: If the folder is not empty, has no simulation.json and
            force is False.
    """
    if not folder.is_dir() or not any(folder.iterdir()):
        return
    try:
        listed = json.loads((folder / "simulation.json").read_text(encoding="utf-8"))
    except OSError, ValueError:
        listed = None
    if not isinstance(listed, dict) or listed.get("format") != FORMAT:
        if not force:
            raise FileExistsError(
                f"{folder} is not empty and holds no simulation.json; use an empty "
                "folder, or force to write into it."
            )
        return
    names: list[str] = []
    _collect_strings(listed.get("files"), names)
    for name in [*names, "simulation.json"]:
        target = (folder / name).resolve()
        if target.is_file() and folder.resolve() in target.parents:
            target.unlink()


def _collect_strings(value: object, names: list[str]) -> None:
    """Appends to `names` every string in `value` and in its nested lists and dicts."""
    if isinstance(value, str):
        names.append(value)
    elif isinstance(value, list):
        for item in value:
            _collect_strings(item, names)
    elif isinstance(value, dict):
        for item in value.values():
            _collect_strings(item, names)


def _file_name(name: str) -> str:
    """Returns a tissue name made safe for a file name."""
    return re.sub(r"[^A-Za-z0-9_-]", "_", name)


def _write_phantom_copy(
    phantom: Phantom,
    protocol: Protocol,
    files: SimulationFiles,
    bundle_names: Sequence[str] | None,
) -> None:
    """Writes the phantom file, with its tractograms and maps pointing into the folder.

    Bundle names go in the file's viewer block.
    """
    bundles: list[Bundle] = []
    for bundle, path in zip(phantom.bundles, files.bundles, strict=True):
        geometry = isinstance(bundle.source, BundleGeometry)
        bundles.append(bundle if geometry else replace(bundle, path=path))
    maps = [replace(m, path=files.tissue_maps[m.tissue]) for m in phantom.tissue_maps]
    written = replace(phantom, bundles=tuple(bundles), tissue_maps=tuple(maps))
    viewer = (
        {"bundles": [{"name": str(name)} for name in bundle_names]}
        if bundle_names is not None
        else None
    )
    write_phantom(written, files.phantom, protocol=protocol, viewer=viewer)


def _write_info(
    simulated: SimulatedDWI,
    truth: GroundTruth,
    protocol: Protocol,
    files: SimulationFiles,
) -> None:
    """Writes simulation.json."""
    phantom = truth.phantom

    def name(path: Path) -> str:
        return path.relative_to(files.folder).as_posix()

    info = {
        "format": FORMAT,
        "version": VERSION,
        "larmor_version": _larmor_version(),
        "engine": "instant",
        "protocol": protocol.to_dict(),
        "echo_times_ms": list(simulated.echo_times_ms),
        "snr": simulated.snr,
        "sigma": simulated.sigma,
        "noise_reference": simulated.noise_reference,
        "seed": simulated.seed,
        "tissues": list(truth.tissue_names),
        "compartments": [
            {"tissue": t.name, "model": c.model.value, "fraction": c.fraction}
            for t in phantom.tissues
            for c in t.compartments
        ],
        "background": phantom.background,
        "ground_truth_peaks": {
            "merge_angle_deg": PEAK_MERGE_ANGLE,
            "relative_threshold": PEAK_RELATIVE_THRESHOLD,
        },
        "mask_fraction": MASK_FRACTION,
        "files": {
            "dwi": [name(p) for p in files.dwi],
            "bvals": name(files.bvals),
            "bvecs": name(files.bvecs),
            "noise_free": [name(p) for p in files.noise_free],
            "tissue_fractions": name(files.tissue_fractions),
            "background_fraction": name(files.background_fraction),
            "free_water_fraction": name(files.free_water_fraction),
            "bundle_fractions": (
                None if files.bundle_fractions is None else name(files.bundle_fractions)
            ),
            "compartment_fractions": name(files.compartment_fractions),
            "signal_fractions": [name(p) for p in files.signal_fractions],
            "free_water_signal_fraction": [
                name(p) for p in files.free_water_signal_fraction
            ],
            "peaks": name(files.peaks),
            "masks": {k: name(p) for k, p in files.masks.items()},
            "bundle_mask": None
            if files.bundle_mask is None
            else name(files.bundle_mask),
            "bundles": [name(p) for p in files.bundles],
            "tissue_maps": {k: name(p) for k, p in files.tissue_maps.items()},
            "phantom": name(files.phantom),
        },
    }
    files.info.write_text(
        json.dumps(info, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def read_ground_truth(folder: str | Path) -> GroundTruth:
    """Reads the ground truth of a simulation from the folder `save_simulation` wrote.

    The files are found through the folder's simulation.json. The fractions and
    peaks are float32, as they are stored, and the partition is None, since the
    folder does not keep each voxel's fibres.

    Args:
        folder: The simulation folder.

    Returns:
        The ground truth.

    Raises:
        ValueError: If the folder holds no simulation.json, or it is not one that
            `save_simulation` wrote.
        OSError: If a file cannot be read.
    """
    folder = Path(folder)
    info_path = folder / "simulation.json"
    if not info_path.is_file():
        raise ValueError(f"{folder} holds no simulation.json; it is not a simulation.")
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except ValueError as e:
        raise ValueError(f"{info_path} cannot be read: {e}") from e
    if not isinstance(info, dict) or info.get("format") != FORMAT:
        raise ValueError(f"{info_path} is not a Larmor simulation file.")
    try:
        files = info["files"]
        tissues = load_volume(folder / files["tissue_fractions"])
        grid = Grid(tissues.data.shape[:3], tissues.affine.copy())
        loaded = read_phantom(folder / files["phantom"])
        bundles = files.get("bundles", [])
        bundle_fractions = (
            load_volume(folder / files["bundle_fractions"]).data
            if files.get("bundle_fractions") is not None
            else np.zeros((*grid.shape, 0), dtype=np.float32)
        )
        return GroundTruth(
            phantom=loaded.phantom,
            grid=grid,
            tissue_names=tuple(info["tissues"]),
            tissue_fractions=tissues.data.reshape(*grid.shape, -1),
            background=load_volume(folder / files["background_fraction"]).data,
            bundle_fractions=bundle_fractions.reshape(*grid.shape, len(bundles)),
            streamlines=tuple(load_tractogram(folder / p) for p in bundles),
            peaks=load_volume(folder / files["peaks"]).data,
            bundle_names=(
                None if loaded.bundle_names is None else tuple(loaded.bundle_names)
            ),
        )
    except (KeyError, TypeError) as e:
        raise ValueError(f"{info_path} does not list the ground truth: {e!r}.") from e
