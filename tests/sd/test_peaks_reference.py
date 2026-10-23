import math
import pathlib

import numpy as np

from larmor.io import load_volume
from larmor.sd import find_peaks
from larmor.sd.sh import build_basis_matrix
from larmor.sd.sphere import get_theta_phi

DATA = pathlib.Path(__file__).parents[1] / "data" / "peaks"

# Both climb to the same maximum with Newton steps, so directions agree far
# below this; the reference is float32, so amplitudes agree to ~1e-7 relative.
COS_TOL = math.cos(math.radians(0.01))
AMP_RTOL = 1e-5


def test_largest_peak_matches_mrtrix_sh2peaks() -> None:
    fod = load_volume(DATA / "fod.mif")
    ref = load_volume(DATA / "peaks_ref.mif")
    # sh2peaks writes its own strides; compare both in the same voxel order.
    fod.to_closest_canonical()
    ref.to_closest_canonical()
    assert np.allclose(fod.affine, ref.affine)

    found = find_peaks(fod.as_float(dtype=np.float64))
    ref_first = ref.as_float(dtype=np.float64)[..., :3]

    larmor_has_peak = np.isfinite(found.amplitudes[..., 0])
    ref_has_peak = np.isfinite(ref_first).all(axis=-1)
    np.testing.assert_array_equal(larmor_has_peak, ref_has_peak)
    assert ref_has_peak.any()

    both = larmor_has_peak & ref_has_peak
    ref_amp = np.linalg.norm(ref_first[both], axis=-1)
    ref_dir = ref_first[both] / ref_amp[:, None]
    cos = np.abs(np.sum(found.directions[both][:, 0] * ref_dir, axis=-1))
    assert np.all(cos >= COS_TOL)
    np.testing.assert_allclose(found.amplitudes[both][:, 0], ref_amp, rtol=AMP_RTOL)


def test_every_peak_matches_mrtrix_sh2peaks_above_the_threshold() -> None:
    # sh2peaks keeps maxima of any sign and has no relative threshold, so its
    # peaks are first reduced to the ones find_peaks keeps: positive amplitude
    # (the FOD at the peak, since the stored length is |amplitude|) and at least
    # 0.1 of the voxel's largest. Then every slot is compared, not just the first.
    fod = load_volume(DATA / "fod.mif")
    ref = load_volume(DATA / "peaks_ref.mif")
    fod.to_closest_canonical()
    ref.to_closest_canonical()
    coeffs = fod.as_float(dtype=np.float64).reshape(-1, fod.data.shape[-1])
    vectors = ref.as_float(dtype=np.float64).reshape(-1, 3, 3)
    lmax = 8

    found = find_peaks(coeffs)

    angles = []
    missed = 0
    for i in range(len(coeffs)):
        stored = vectors[i][np.isfinite(vectors[i]).all(axis=1)]
        dirs = stored / np.linalg.norm(stored, axis=1, keepdims=True)
        theta, phi = get_theta_phi(dirs)
        signed = build_basis_matrix(theta, phi, lmax) @ coeffs[i]
        dirs, signed = dirs[signed > 0], signed[signed > 0]
        kept = signed >= 0.1 * signed.max() if len(signed) else signed > 0
        dirs, signed = dirs[kept], signed[kept]

        mine = found.directions[i][np.isfinite(found.amplitudes[i])]
        for d in mine:
            best = np.abs(dirs @ d).max() if len(dirs) else 0.0
            angles.append(math.degrees(math.acos(min(best, 1.0))))
        for d, a in zip(dirs, signed, strict=True):
            # A margin above the threshold: a peak right at 0.1 may fall either side.
            near = np.abs(mine @ d).max() if len(mine) else 0.0
            if a >= 0.101 * signed.max() and near < math.cos(math.radians(1.0)):
                missed += 1

    # The ticket's criteria: median below 0.01 deg, 95th percentile below 0.1 deg.
    # A few lar peaks have no sh2peaks match, lobes its 60 fixed starts miss.
    assert len(angles) > 500
    assert np.median(angles) < 0.01
    assert np.percentile(angles, 95) < 0.1
    assert missed == 0
