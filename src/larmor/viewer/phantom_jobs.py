"""Opens and runs phantoms on the job runner, off the main thread."""

from collections.abc import Callable
from functools import partial
from pathlib import Path

from larmor.viewer.jobs import JobRunner, Report
from larmor.viewer.model.item import Category
from larmor.viewer.model.phantom import (
    LoadedPhantom,
    RunOutputs,
    SimulationOutputs,
    read_phantom_file,
    run_simulation,
    simulation_outputs,
)
from larmor.viewer.model.viewer import Viewer

# Opens files as items of a category, such as the loader's `open`.
type OpenFiles = Callable[[list[Path], Category], None]


def open_outputs(files: SimulationOutputs, open_files: OpenFiles) -> None:
    """Opens a simulation's DWI as volumes, masks as overlays and peaks."""
    open_files(list(files.dwi), Category.VOLUMES)
    open_files(list(files.masks), Category.OVERLAYS)
    if files.peaks is not None:
        open_files([files.peaks], Category.PEAKS)


def start_run(
    viewer: Viewer,
    jobs: JobRunner,
    folder: str | Path,
    open_files: OpenFiles,
    *,
    overwrite: bool = False,
    on_done: Callable[[RunOutputs], None] | None = None,
    on_failed: Callable[[str], None] | None = None,
) -> None:
    """Simulates the phantom in the background, then opens what it wrote.

    The phantom is checked and taken as it is now; later edits do not change
    the run. Once it ends, the outputs open through open_files. A cancelled
    run opens nothing.

    Args:
        viewer: The viewer whose phantom runs.
        jobs: The job runner.
        folder: The output folder.
        open_files: Opens files as items of a category.
        overwrite: Replace outputs already in the folder.
        on_done: Told the files written, after they are opened.
        on_failed: Told why the run failed; the runner's error box if None.

    Raises:
        ValueError: If no phantom is open or it cannot run.
        FileExistsError: If outputs exist and overwrite is False.
        JobBusyError: If a job runs already.
    """
    inputs = viewer.prepare_run(folder, overwrite=overwrite)
    path = Path(folder).absolute()

    def done(outputs: RunOutputs) -> None:
        """Opens the run's outputs and records the run in the viewer."""
        open_outputs(outputs.files, open_files)
        viewer.finish_run(outputs)
        if on_done is not None:
            on_done(outputs)

    jobs.start("Simulation", partial(run_simulation, inputs, path), done, on_failed)


def start_open(
    viewer: Viewer,
    jobs: JobRunner,
    path: str | Path,
    *,
    on_done: Callable[[], object] | None = None,
    on_failed: Callable[[str], None] | None = None,
) -> None:
    """Reads a phantom file and builds its bundles in the background, then shows it.

    Args:
        viewer: The viewer to show the phantom in.
        jobs: The job runner.
        path: The phantom file.
        on_done: Called once the phantom shows.
        on_failed: Told why it could not be read; the runner's error box if None.

    Raises:
        JobBusyError: If a job runs already.
    """

    def done(loaded: LoadedPhantom) -> None:
        """Shows the phantom read."""
        viewer.apply_phantom(loaded)
        if on_done is not None:
            on_done()

    work = partial(read_phantom_file, Path(path).absolute())
    jobs.start("Opening phantom", work, done, on_failed)


def start_open_simulation(
    viewer: Viewer,
    jobs: JobRunner,
    folder: str | Path,
    open_files: OpenFiles,
    *,
    on_done: Callable[[SimulationOutputs], object] | None = None,
    on_failed: Callable[[str], None] | None = None,
) -> None:
    """Opens a simulation folder's phantom in the editor and its files as items.

    The phantom file is read and its bundles built in the background; the DWI,
    masks and peaks open through open_files.

    Args:
        viewer: The viewer.
        jobs: The job runner.
        folder: The simulation folder, holding simulation.json.
        open_files: Opens files as items of a category.
        on_done: Told the folder's files, once they are queued.
        on_failed: Told why it could not be read; the runner's error box if None.

    Raises:
        ValueError: If it is not a simulation folder.
        OSError: If simulation.json cannot be read.
        JobBusyError: If a job runs already.
    """
    files = simulation_outputs(Path(folder))

    def read(report: Report) -> LoadedPhantom:
        """Reads the folder's phantom file."""
        return read_phantom_file(files.phantom, report)

    def done(loaded: LoadedPhantom) -> None:
        """Shows the phantom, links it to the folder and opens the folder's files."""
        viewer.apply_phantom(loaded)
        viewer.link_simulation(files)
        open_outputs(files, open_files)
        if on_done is not None:
            on_done(files)

    jobs.start("Opening simulation", read, done, on_failed)
