import math
from pathlib import Path

import numpy as np
import pytest

from larmor.types import Tractogram, Volume
from larmor.viewer import Category, OptionName, TractStyle, Viewer
from larmor.viewer.model.camera import Camera3D
from larmor.viewer.model.export import (
    DEFAULT_PRESETS,
    ExportPreset,
    find_preset,
    load_presets,
    presets_folder,
    read_preset,
    save_preset,
    write_preset,
)
from larmor.viewer.model.options import Colour


def _preset() -> ExportPreset:
    return ExportPreset(
        "Poster",
        size=(640, 480),
        scale=3.0,
        background=Colour("#102030"),
        overlay=True,
        transparent=True,
        overrides={
            Category.TRACTOGRAMS: {
                OptionName.STYLE: TractStyle.TUBES,
                OptionName.TUBE_RADIUS: 0.5,
            },
            Category.VOLUMES: {OptionName.INTERPOLATE: False},
        },
        camera=Camera3D((0.0, -1.0, 0.0), (0.0, 0.0, 1.0), 2.0),
    )


@pytest.mark.parametrize("preset", [_preset(), *DEFAULT_PRESETS])
def test_a_preset_is_read_back_from_its_file(
    preset: ExportPreset, tmp_path: Path
) -> None:
    path = tmp_path / "p.toml"
    write_preset(preset, path)
    read = read_preset(path)
    assert read.overrides == preset.overrides
    assert read == preset


def test_the_shipped_presets_are_listed_and_a_file_replaces_one(tmp_path: Path) -> None:
    assert list(load_presets(tmp_path)) == ["Screen", "Figure", "Talk", "Website"]
    save_preset(ExportPreset("Website", scale=4.0), tmp_path)
    save_preset(_preset(), tmp_path)
    presets = load_presets(tmp_path)
    assert presets["Website"].scale == 4.0
    assert presets["Poster"] == _preset()


def test_an_unreadable_preset_file_is_skipped(tmp_path: Path) -> None:
    (tmp_path / "bad.toml").write_text('name = "Bad"\nscale = -1\n')
    (tmp_path / "unknown.toml").write_text(
        'name = "Unknown"\n[overrides.volumes]\nno_such_option = 1\n'
    )
    assert set(load_presets(tmp_path)) == {"Screen", "Figure", "Talk", "Website"}


def test_an_unknown_preset_name_is_refused_with_the_names() -> None:
    with pytest.raises(ValueError, match="Screen, Figure, Talk, Website"):
        find_preset("Poster")


def test_the_presets_folder_is_beside_the_settings_file() -> None:
    from larmor.viewer.model import settings

    assert presets_folder().parent == settings.settings_path().parent


def test_the_website_camera_looks_down_at_12_degrees_from_the_front_right() -> None:
    camera = find_preset("Website").camera
    assert camera is not None
    forward = np.asarray(camera.forward)
    assert math.degrees(math.asin(-forward[2])) == pytest.approx(12.0)
    assert forward[0] < 0 and forward[1] < 0  # from the subject's right and front
    assert np.dot(forward, camera.up) == pytest.approx(0.0, abs=1e-12)


def test_preset_options_apply_for_the_block_and_are_restored() -> None:
    viewer = Viewer()
    volume = viewer.add(
        Volume(np.zeros((4, 4, 4), np.float32), np.eye(4)), Category.VOLUMES
    )
    points = np.array([[0.0, 0, 0], [1, 1, 1]])
    tracts = viewer.add(Tractogram(points, np.array([0, 2])), Category.TRACTOGRAMS)
    camera = viewer.scene.view.cameras.three_d
    preset = _preset()
    with viewer.preset_options(preset):
        assert getattr(viewer.scene.item(tracts.id).options, OptionName.STYLE) == (
            TractStyle.TUBES
        )
        assert not getattr(viewer.scene.item(volume.id).options, OptionName.INTERPOLATE)
        assert viewer.scene.view.cameras.three_d.zoom == 2.0
    assert getattr(viewer.scene.item(tracts.id).options, OptionName.STYLE) == (
        TractStyle.LINES
    )
    assert getattr(viewer.scene.item(volume.id).options, OptionName.INTERPOLATE)
    assert viewer.scene.view.cameras.three_d == camera
