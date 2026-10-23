import logging
from pathlib import Path

import numpy as np
import pytest

from larmor.io import save_surface, save_tractogram, save_volume
from larmor.io.formats import write_freesurfer_surface, write_freesurfer_values
from larmor.types import Surface, Tractogram, Volume
from larmor.viewer import Category, Colormap, OptionError, OptionName, Viewer
from larmor.viewer.model.options import Colour, SurfaceOptions, VolumeOptions
from larmor.viewer.model.registry import REGISTRY, item_type
from larmor.viewer.model.surface import SurfaceData, read_surface

# Small FreeSurfer files committed for the tests: see tests/data/README.md.
SUBJECT = Path(__file__).parents[2] / "data" / "surfaces"
CURV = np.array([-0.3, 0.1, 0.4, -0.2], np.float32)
CORTEX = np.array([True, True, False, True])


def _mesh(**values: np.ndarray) -> Surface:
    vertices = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0], [0, 0, 10]], np.float32)
    faces = np.array([[0, 1, 2], [0, 1, 3]])
    return Surface(vertices, faces, vertex_data=dict(values))


def _write_label(path: Path, indices: list[int]) -> None:
    lines = ["#!ascii label", str(len(indices))]
    lines += [f"{i} 0.0 0.0 0.0 0.0" for i in indices]
    path.write_text("\n".join(lines) + "\n")


def _subject(folder: Path, *, curv: bool = True, label: bool = True) -> Path:
    # A FreeSurfer subject's layout: surf/lh.white, surf/lh.curv, label/lh.cortex.label
    (folder / "surf").mkdir(parents=True)
    (folder / "label").mkdir()
    white = folder / "surf" / "lh.white"
    write_freesurfer_surface(_mesh(), white)
    if curv:
        write_freesurfer_values(CURV, folder / "surf" / "lh.curv")
    if label:
        _write_label(folder / "label" / "lh.cortex.label", [0, 1, 3])
    return white


def _surface_item(viewer: Viewer, **values: np.ndarray) -> int:
    return viewer.add(_mesh(**values), Category.SURFACES, name="lh.inflated").id


def _options(viewer: Viewer, item: int) -> SurfaceOptions:
    options = viewer.scene.item(item).options
    assert isinstance(options, SurfaceOptions)
    return options


# Which category reads a file


def _write(path: Path) -> Path:
    if path.name == "lh.curv":
        write_freesurfer_values(CURV, path)
    elif path.name.endswith(".nii.gz"):
        save_volume(Volume(np.zeros((2, 2, 2), np.float32), np.eye(4)), path)
    elif path.name.endswith(".tck"):
        points = np.zeros((2, 3), np.float32)
        save_tractogram(Tractogram(points=points, offsets=np.array([0, 2])), path)
    else:
        save_surface(_mesh(), path)
    return path


@pytest.mark.parametrize(
    ("name", "reads"),
    [
        ("lh.white.gii", True),
        ("lh.white", True),  # FreeSurfer: named by hemisphere, no extension
        ("RH.PIAL", True),
        ("brain.white", False),
        ("t1.nii.gz", False),
    ],
)
def test_the_surfaces_category_reads_gifti_and_hemisphere_names(
    name: str, reads: bool
) -> None:
    assert item_type(Category.SURFACES).reads(Path(name)) is reads


def test_a_file_is_never_judged_by_its_content(tmp_path: Path) -> None:
    surface = _write(tmp_path / "white_matter")  # a FreeSurfer surface, any name
    assert not any(kind.reads(surface) for kind in REGISTRY.values())


# Reading a surface


def test_a_freesurfer_surface_reads_its_curvature_and_cortex_label(
    tmp_path: Path,
) -> None:
    data = read_surface(_subject(tmp_path))
    assert list(data.surface.vertex_data) == ["curv"]
    np.testing.assert_array_equal(data.surface.vertex_data["curv"], CURV)
    assert data.cortex is not None
    np.testing.assert_array_equal(data.cortex, CORTEX)


def test_a_gifti_surface_reads_no_curvature_or_label_beside_it(
    tmp_path: Path,
) -> None:
    white = _subject(tmp_path)
    gifti = white.with_name("lh.white.gii")
    save_surface(_mesh(), gifti)
    data = read_surface(gifti)
    assert data.surface.vertex_data == {}
    assert data.cortex is None


def test_a_surface_without_curvature_or_label_loads_alone(tmp_path: Path) -> None:
    data = read_surface(_subject(tmp_path, curv=False, label=False))
    assert data.surface.vertex_data == {}
    assert data.cortex is None


@pytest.mark.parametrize(
    ("broken", "content"),
    [
        ("surf/lh.curv", None),  # 3 values for 4 vertices
        ("label/lh.cortex.label", "#!ascii label\nnot a count\n"),
        ("label/lh.cortex.label", "#!ascii label\n1\n9 0 0 0 0\n"),  # no vertex 9
    ],
)
def test_an_unreadable_curvature_or_label_is_logged_and_the_surface_loads(
    broken: str, content: str | None, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    white = _subject(tmp_path)
    path = tmp_path / broken
    if content is None:
        write_freesurfer_values(CURV[:3], path)
    else:
        path.write_text(content)
    with caplog.at_level(logging.WARNING, logger="larmor"):
        data = read_surface(white)
    assert data.surface.num_vertices == 4
    [record] = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert str(path) in record.getMessage()
    if broken.endswith("curv"):
        assert data.surface.vertex_data == {} and data.cortex is not None
    else:
        assert "curv" in data.surface.vertex_data and data.cortex is None


# A new surface's options


@pytest.mark.parametrize(
    ("name", "colour"),
    [
        ("lh.white", "#ffff00"),
        ("rh.pial", "#ff0000"),
        ("lh.inflated", "#d9d9d9"),
        (None, "#d9d9d9"),
    ],
)
def test_a_new_surface_is_coloured_by_its_name(name: str | None, colour: str) -> None:
    viewer = Viewer()
    item = viewer.add(_mesh(), Category.SURFACES, name=name)
    assert _options(viewer, item.id).colour == Colour(colour)


@pytest.mark.parametrize("values", [{"curv": CURV}, {"thickness": CURV + 2}])
def test_a_new_surface_is_coloured_by_its_curvature_only_when_it_has_one(
    values: dict[str, np.ndarray],
) -> None:
    viewer = Viewer()
    options = _options(viewer, _surface_item(viewer, **values))
    if "curv" in values:
        assert options.colour_by == "curv"
        assert options.colormap == Colormap.GRAY_R
        assert (options.minimum, options.maximum) == (-0.3, 0.3)
    else:
        assert options.colour_by is None


# Colour by


def test_colour_by_offers_the_numeric_values_per_vertex() -> None:
    viewer = Viewer()
    item = _surface_item(
        viewer,
        curv=CURV,
        labels=np.arange(4),
        normals=np.zeros((4, 3)),
        names=np.array(["a", "b", "c", "d"]),
    )
    assert viewer.data_choices(item, OptionName.COLOUR_BY) == ["curv", "labels"]


def test_colour_by_refuses_values_the_surface_does_not_have() -> None:
    viewer = Viewer()
    item = _surface_item(viewer, curv=CURV)
    with pytest.raises(OptionError, match="thickness"):
        viewer.set(item, OptionName.COLOUR_BY, "thickness")
    assert _options(viewer, item).colour_by == "curv"


@pytest.mark.parametrize(
    ("name", "colormap", "value_range"),
    [
        ("curv", Colormap.GRAY_R, (-0.3, 0.3)),
        ("sulc", Colormap.GRAY_R, (-0.3, 0.3)),
        ("thickness", VolumeOptions().colormap, (1.0, 4.0)),  # NaN left out
    ],
)
def test_new_values_to_colour_by_set_their_colormap_and_range(
    name: str, colormap: Colormap, value_range: tuple[float, float]
) -> None:
    viewer = Viewer()
    sulc = np.array([2.0, -3.0, 1.0, 0.0])
    thickness = np.array([1.0, np.nan, 4.0, 2.5])
    item = _surface_item(viewer, curv=CURV, sulc=sulc, thickness=thickness)
    viewer.set(item, OptionName.COLOUR_BY, None)
    viewer.set(item, OptionName.COLORMAP, Colormap.HOT)
    viewer.set(item, OptionName.MINIMUM, 0.1)
    viewer.set(item, OptionName.COLOUR_BY, name)
    options = _options(viewer, item)
    assert options.colour_by == name
    assert options.colormap == colormap
    assert (options.minimum, options.maximum) == value_range


def test_colouring_by_nothing_keeps_the_colormap_and_range() -> None:
    viewer = Viewer()
    item = _surface_item(viewer, curv=CURV)
    viewer.set(item, OptionName.COLORMAP, Colormap.HOT)
    viewer.set(item, OptionName.MINIMUM, 0.1)
    viewer.set(item, OptionName.COLOUR_BY, None)
    options = _options(viewer, item)
    assert options.colormap == Colormap.HOT
    assert (options.minimum, options.maximum) == (0.1, 0.3)


# Sessions


@pytest.mark.parametrize(
    ("kind", "colour_by", "minimum"),
    [("freesurfer", "curv", -0.2), ("gifti", "thickness", 1.5)],
)
def test_a_session_restores_a_surfaces_values_colouring_and_range(
    kind: str, colour_by: str, minimum: float, tmp_path: Path
) -> None:
    # The GIfTI's restored colour_by differs from the default curv, so a range
    # restored before it would be reset.
    if kind == "freesurfer":
        path = _subject(tmp_path / "subject")
    else:
        path = tmp_path / "lh.white.gii"
        save_surface(_mesh(curv=CURV, thickness=np.array([1.0, 2, 3, 4])), path)
    saved = Viewer()
    item = saved.load(path, Category.SURFACES).id
    saved.set(item, OptionName.COLOUR_BY, colour_by)
    saved.set(item, OptionName.MINIMUM, minimum)
    assert saved.save_session(tmp_path / "s.lar.json") == []

    viewer = Viewer()
    report = viewer.load_session(tmp_path / "s.lar.json")

    assert report.problems == ()
    [restored] = viewer.scene.items
    assert restored.source == path
    assert restored.options == saved.scene.item(item).options
    assert isinstance(restored.data, SurfaceData)
    assert (restored.data.cortex is not None) == (kind == "freesurfer")


# Real data


def test_freesurfer_white_loads_with_its_curvature_and_cortex() -> None:
    path = SUBJECT / "surf" / "lh.white"
    viewer = Viewer()
    item = viewer.load(path, Category.SURFACES)
    assert isinstance(item.data, SurfaceData)
    assert item.data.surface.num_vertices == 642
    assert item.data.cortex is not None
    assert item.data.cortex.sum() == 578
    options = _options(viewer, item.id)
    assert options.colour == Colour("#ffff00")
    assert options.colour_by == "curv"


# Hiding the medial wall


def test_hiding_the_medial_wall_of_a_surface_without_a_cortex_label_is_refused() -> (
    None
):
    viewer = Viewer()
    item = _surface_item(viewer)
    with pytest.raises(OptionError, match="no cortex label"):
        viewer.set(item, OptionName.HIDE_MEDIAL_WALL, True)
    assert not _options(viewer, item).hide_medial_wall
    viewer.set(item, OptionName.HIDE_MEDIAL_WALL, False)


def test_the_surfaces_category_hides_the_medial_wall_of_those_with_a_label() -> None:
    viewer = Viewer()
    bare = _surface_item(viewer)
    labelled = viewer.add(
        SurfaceData(_mesh(), np.array([True, True, False, True])),
        Category.SURFACES,
        name="lh.white",
    ).id
    viewer.set(Category.SURFACES, OptionName.HIDE_MEDIAL_WALL, True)
    assert _options(viewer, labelled).hide_medial_wall
    assert _options(viewer, bare).hide_medial_wall  # set, and ignored without a mask
