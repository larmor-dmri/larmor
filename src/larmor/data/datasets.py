"""The example datasets: their files, licences and citations, and their fetchers."""

from pathlib import Path

from larmor.data.fetch import Dataset, fetch_dataset
from larmor.data.files import CFIN_FILES, ISMRM2015_FILES, STANFORD_FILES

CFIN_MULTISHELL = Dataset(
    name="cfin_multishell",
    title=(
        "CFIN multi-shell diffusion MRI of a normal human brain (Hansen and Jespersen, "
        "2016)"
    ),
    page="https://doi.org/10.5061/dryad.9bc43",
    licence="CC0 1.0 (https://creativecommons.org/publicdomain/zero/1.0/)",
    citation=(
        "Hansen B, Jespersen SN. Data for evaluation of fast kurtosis strategies, "
        "b-value optimization and exploration of diffusion MRI contrast. Scientific "
        "Data 3 (2016) 160072. https://doi.org/10.1038/sdata.2016.72"
    ),
    files=CFIN_FILES,
)

STANFORD_HARDI = Dataset(
    name="stanford_hardi",
    title="Stanford high angular resolution diffusion MRI (Rokem et al., 2013)",
    page="https://purl.stanford.edu/yx282xq2090",
    licence=(
        "Open Data Commons Public Domain Dedication and License (PDDL) 1.0 "
        "(https://opendatacommons.org/licenses/pddl/1-0/)"
    ),
    citation=(
        "Rokem A, Yeatman J, Pestilli F, Wandell B. High angular resolution "
        "diffusion MRI. Stanford Digital Repository (2013). "
        "http://purl.stanford.edu/yx282xq2090"
    ),
    files=STANFORD_FILES,
)

ISMRM2015 = Dataset(
    name="ismrm2015",
    title=(
        "ISMRM 2015 Tractography Challenge data: simulated diffusion MRI and a "
        "T1-like image (Maier-Hein et al., 2017)"
    ),
    page="https://doi.org/10.5281/zenodo.572345",
    licence="CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)",
    citation=(
        "Maier-Hein KH, Neher PF, Houde J-C, et al. The challenge of mapping the "
        "human connectome based on diffusion tractography. Nature Communications 8 "
        "(2017) 1349. https://doi.org/10.1038/s41467-017-01285-x"
    ),
    files=ISMRM2015_FILES,
    changes=(
        "dwi.bvecs differs from Diffusion.bvecs in the Zenodo archive: the sign of "
        "its first row (the x components) is changed, the correction the "
        "challenge's data page gives for this file. The other files are as "
        "published, renamed."
    ),
)


def fetch_cfin_multishell(folder: str | Path | None = None) -> dict[str, Path]:
    """Downloads the CFIN multi-shell dataset, if needed, and returns its files.

    One healthy volunteer scanned at 3 T (Siemens Trio) by Hansen and Jespersen:
    diffusion-weighted images at 2.5 mm isotropic resolution (96 x 96 x 19 voxels)
    with one b=0 image and 33 directions at each of 15 b-values from 200 to 3000
    s/mm^2, as acquired (no preprocessing), and a defaced 1 mm T1-weighted MPRAGE
    of the same session. Enough shells and directions for DKI and multi-shell
    deconvolution. About 197 MB.

    The data is published at https://doi.org/10.5061/dryad.9bc43.

    Please cite: Hansen B, Jespersen SN. Data for evaluation of fast kurtosis
    strategies, b-value optimization and exploration of diffusion MRI contrast.
    Scientific Data 3 (2016) 160072. https://doi.org/10.1038/sdata.2016.72

    Args:
        folder: The cache folder; None for `data_folder()`.

    Returns:
        The paths of the files, by key: "dwi" (NIfTI), "bvals" and "bvecs" (FSL
        format), and "t1" (NIfTI, on its own grid).

    Raises:
        OSError: If a download fails; run it again to start that file anew.
        ValueError: If a downloaded file does not match its checksum.
    """
    return fetch_dataset(CFIN_MULTISHELL, folder)


def fetch_stanford_hardi(folder: str | Path | None = None) -> dict[str, Path]:
    """Downloads the Stanford HARDI dataset, if needed, and returns its files.

    One healthy volunteer scanned at 3 T at Stanford: diffusion-weighted images at
    2 mm isotropic resolution (81 x 106 x 76 voxels), 10 b=0 images and 150
    directions at b=2000 s/mm^2, motion corrected; a T1-weighted image and a
    reduced FreeSurfer parcellation (aparc), both resampled to the diffusion grid.
    A single shell with many directions, for DTI, FODs and tractography. About
    93 MB.

    The b-vectors are in world coordinates: load them with `frame="world"`.

    The data is published at https://purl.stanford.edu/yx282xq2090. The
    repository asks that the data never be used to identify the volunteer.

    Please cite: Rokem A, Yeatman J, Pestilli F, Wandell B. High angular resolution
    diffusion MRI. Stanford Digital Repository (2013).
    http://purl.stanford.edu/yx282xq2090

    Args:
        folder: The cache folder; None for `data_folder()`.

    Returns:
        The paths of the files, by key: "dwi" (NIfTI), "bvals", "bvecs" (in world
        coordinates), "t1" and "labels" (NIfTI, on the diffusion grid).

    Raises:
        OSError: If a download fails; run it again to start that file anew.
        ValueError: If a downloaded file does not match its checksum.
    """
    return fetch_dataset(STANFORD_HARDI, folder)


def fetch_ismrm2015(folder: str | Path | None = None) -> dict[str, Path]:
    """Downloads the ISMRM 2015 Tractography Challenge data, if needed.

    The data given to the challenge's contestants, simulated from bundles traced
    in a real brain: diffusion-weighted images at 2 mm isotropic resolution (90 x
    108 x 90 voxels), one b=0 image and 32 directions at b=1000 s/mm^2, with the
    noise and artefacts of a scan, and a simulated T1-like image at 1 mm on the
    same field of view. One shell with few directions: enough for DTI, FODs up to
    lmax 6 and tractography. About 41 MB to download (a zip archive, deleted once
    its files are extracted), 36 MB in the cache.

    The x components of the published b-vectors have the wrong sign. They are
    flipped when the files are extracted, so the saved `dwi.bvecs` differs from
    the Zenodo original and is correct with the default frame of `load_dwi`.

    Published on Zenodo (https://doi.org/10.5281/zenodo.572345) under CC BY 4.0.

    Please cite: Maier-Hein KH, Neher PF, Houde J-C, et al. The challenge of
    mapping the human connectome based on diffusion tractography. Nature
    Communications 8 (2017) 1349. https://doi.org/10.1038/s41467-017-01285-x

    Args:
        folder: The cache folder; None for `data_folder()`.

    Returns:
        The paths of the files, by key: "dwi" (NIfTI), "bvals" and "bvecs" (FSL
        format, corrected) and "t1" (NIfTI, at 1 mm).

    Raises:
        OSError: If a download fails; run it again to start that file anew.
        ValueError: If a downloaded file does not match its checksum.
    """
    return fetch_dataset(ISMRM2015, folder)
