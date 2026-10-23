from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from viewer_helpers import tract_options

from larmor.io import save_tractogram, save_volume, write_tsf
from larmor.types import Tractogram, Volume
from larmor.viewer import (
    Category,
    Colormap,
    Colouring,
    Item,
    OptionError,
    OptionName,
    Viewer,
)

# Two streamlines of 3 and 2 points; the values are exact in float32.
OFFSETS = np.array([0, 3, 5])
VALUES = np.array([0.5, -1.0, 2.0, 4.0, 1.5], np.float32)


def _tractogram() -> Tractogram:
    points = np.arange(15, dtype=np.float32).reshape(-1, 3)
    return Tractogram(points, OFFSETS)


def _tsf(
    path: Path, values: np.ndarray = VALUES, offsets: np.ndarray = OFFSETS
) -> Path:
    write_tsf(values, offsets, path)
    return path


def _volume(volumes: int = 1) -> Volume:
    data = np.linspace(2.0, 7.0, 8 * volumes, dtype=np.float32)
    return Volume(data.reshape(2, 2, 2, volumes), np.eye(4))


def _options(item: Item) -> dict[str, object]:
    o = item.options
    return {n: getattr(o, n) for n in ("values_file", "values_volume", "colouring")}


def test_a_tsf_file_colours_by_value_over_its_range(tmp_path: Path) -> None:
    viewer = Viewer()
    tracts = viewer.add(_tractogram(), Category.TRACTOGRAMS)
    path = _tsf(tmp_path / "v.tsf")
    viewer.set(tracts.id, OptionName.VALUES_FILE, str(path))
    o = tract_options(tracts)
    assert _options(tracts) == {
        "values_file": str(path),
        "values_volume": None,
        "colouring": Colouring.VALUES,
    }
    assert (o.minimum, o.maximum) == (-1.0, 4.0)
    values = viewer._tsf_point_values(tracts.id)
    assert values is not None and np.array_equal(values, VALUES)


@pytest.mark.parametrize(
    ("values", "offsets"),
    [(VALUES[:4], np.array([0, 3, 4])), (VALUES, np.array([0, 2, 5]))],
    ids=["fewer points", "other streamline lengths"],
)
def test_a_tsf_file_that_does_not_match_the_tractogram_is_refused(
    values: np.ndarray, offsets: np.ndarray, tmp_path: Path
) -> None:
    viewer = Viewer()
    tracts = viewer.add(_tractogram(), Category.TRACTOGRAMS)
    before = tracts.options
    path = _tsf(tmp_path / "v.tsf", values, offsets)
    with pytest.raises(OptionError, match="TSF file"):
        viewer.set(tracts.id, OptionName.VALUES_FILE, str(path))
    assert tracts.options == before
    assert viewer._tsf_point_values(tracts.id) is None


def test_a_sampled_volume_colours_by_value_over_its_range(tmp_path: Path) -> None:
    viewer = Viewer()
    tracts = viewer.add(_tractogram(), Category.TRACTOGRAMS)
    fa = viewer.add(_volume(), Category.OVERLAYS)
    viewer.set(tracts.id, OptionName.VALUES_VOLUME, fa.id)
    o = tract_options(tracts)
    assert _options(tracts) == {
        "values_file": "",
        "values_volume": fa.id,
        "colouring": Colouring.VALUES,
    }
    assert (o.minimum, o.maximum) == (2.0, 7.0)


@pytest.mark.parametrize("last", [OptionName.VALUES_FILE, OptionName.VALUES_VOLUME])
def test_choosing_one_source_of_values_clears_the_other(
    last: OptionName, tmp_path: Path
) -> None:
    viewer = Viewer()
    tracts = viewer.add(_tractogram(), Category.TRACTOGRAMS)
    fa = viewer.add(_volume(), Category.VOLUMES)
    path = str(_tsf(tmp_path / "v.tsf"))
    sources: dict[OptionName, object] = {
        OptionName.VALUES_FILE: path,
        OptionName.VALUES_VOLUME: fa.id,
    }
    first = next(n for n in sources if n != last)
    viewer.set(tracts.id, first, sources[first])
    viewer.set(tracts.id, last, sources[last])
    o = tract_options(tracts)
    assert getattr(o, last) == sources[last]
    assert getattr(o, first) == ("" if first == OptionName.VALUES_FILE else None)
    assert (viewer._tsf_point_values(tracts.id) is not None) == (
        last == OptionName.VALUES_FILE
    )
    expected = (-1.0, 4.0) if last == OptionName.VALUES_FILE else (2.0, 7.0)
    assert (o.minimum, o.maximum) == expected


def test_an_image_of_several_volumes_cannot_be_sampled() -> None:
    viewer = Viewer()
    tracts = viewer.add(_tractogram(), Category.TRACTOGRAMS)
    dwi = viewer.add(_volume(3), Category.VOLUMES)
    single = viewer.add(_volume(1), Category.VOLUMES)
    choices = viewer.item_choices(tracts.id, OptionName.VALUES_VOLUME)
    assert [i.id for i in choices] == [single.id]
    with pytest.raises(OptionError):
        viewer.set(tracts.id, OptionName.VALUES_VOLUME, dwi.id)
    assert tract_options(tracts).values_volume is None


@pytest.mark.parametrize("colouring", [Colouring.VALUES, Colouring.SINGLE])
@pytest.mark.parametrize("source", ["file", "volume"])
def test_colouring_by_value_reopens_from_a_session_with_its_range(
    source: str, colouring: Colouring, tmp_path: Path
) -> None:
    save_tractogram(_tractogram(), tmp_path / "cst.tck")
    save_volume(_volume(), tmp_path / "fa.nii.gz")
    path = _tsf(tmp_path / "v.tsf")
    saved = Viewer()
    # A removed item, so saved ids differ from those a fresh viewer gives.
    saved.remove(saved.add(_volume(), Category.VOLUMES).id)
    fa = saved.load(tmp_path / "fa.nii.gz", Category.OVERLAYS)
    tracts = saved.load(tmp_path / "cst.tck", Category.TRACTOGRAMS)
    if source == "file":
        saved.set(tracts.id, OptionName.VALUES_FILE, str(path))
    else:
        saved.set(tracts.id, OptionName.VALUES_VOLUME, fa.id)
    # A range and colormap of the user's own, not the values' range set above.
    saved.set(tracts.id, OptionName.MINIMUM, 0.25)
    saved.set(tracts.id, OptionName.MAXIMUM, 3.0)
    saved.set(tracts.id, OptionName.COLORMAP, Colormap.HOT)
    saved.set(tracts.id, OptionName.COLOUR_BAR, True)
    # Switched away from the values after choosing them, which the session keeps.
    saved.set(tracts.id, OptionName.COLOURING, colouring)
    assert saved.save_session(tmp_path / "s.lar.json") == []

    viewer = Viewer()
    report = viewer.load_session(tmp_path / "s.lar.json")

    assert report.problems == ()
    new = {i.name: i for i in viewer.scene.items}
    reopened = new["cst.tck"]
    expected = tract_options(tracts)
    if source == "volume":
        expected = replace(expected, values_volume=new["fa.nii.gz"].id)
    assert reopened.options == expected
    values = viewer._tsf_point_values(reopened.id)
    assert (values is not None) == (source == "file")
