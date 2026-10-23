"""The URL, SHA-256 and size in bytes of the files of each example dataset."""

from larmor.data.fetch import Member, RemoteFile

_CFIN = "https://digital.lib.washington.edu/researchworks/bitstream/handle/1773/38488/"
_CFIN_DWI = _CFIN + "__DTI_AX_ep2d_2_5_iso_33d_20141015095334_4"
_STANFORD = "https://stacks.stanford.edu/file/druid:yx282xq2090/"
_ISMRM = "https://zenodo.org/records/572345/files/"
_ISMRM_DATA = "ISMRM_2015_Tracto_challenge_data/"

CFIN_FILES = (
    RemoteFile(
        "dwi",
        "dwi.nii",
        _CFIN_DWI + ".nii",
        "1081b5b586a5429a6754078653ffe4ab430d348eb712bc7231c8f99919ac3eff",
        173703520,
    ),
    RemoteFile(
        "bvals",
        "dwi.bval",
        _CFIN_DWI + ".bval",
        "508d18044a4e220e28bdfcbe192238f0abb20869e36b0e2c2c4ead85fc75e614",
        2345,
    ),
    RemoteFile(
        "bvecs",
        "dwi.bvec",
        _CFIN_DWI + ".bvec",
        "4774696dfb4b20223fd1bbb5e05b90a2c01a8b2b8e6f13ebc807b152cb8b5fdd",
        14069,
    ),
    RemoteFile(
        "t1",
        "t1.nii",
        _CFIN + "T1.nii",
        "8f1f3561a745ab4a92b3437a794518c7c750823add2c622da3263ec9fa825158",
        23069024,
    ),
)

STANFORD_FILES = (
    RemoteFile(
        "dwi",
        "dwi.nii.gz",
        _STANFORD + "dwi.nii.gz",
        "2c22516114eb4acbedf0052bc4fda6d0549a8c34758da69164a219d39bafd656",
        91378947,
    ),
    RemoteFile(
        "bvals",
        "dwi.bvals",
        _STANFORD + "dwi.bvals",
        "aa9399094599ddd516603ad81272892bdda85c2d9bee5314086efce8db231e4d",
        770,
    ),
    RemoteFile(
        "bvecs",
        "dwi.bvecs",
        _STANFORD + "dwi.bvecs",
        "42dd0ed0b39824bac888be7f3d68e9dd4c6a85f1c8bfa26a050653c618111723",
        3889,
    ),
    RemoteFile(
        "t1",
        "t1.nii.gz",
        _STANFORD + "t1.nii.gz",
        "df5c5bf8a645509802fcb591af0672ed00251b08998f4c8fb56c9c06aa127754",
        1153166,
    ),
    RemoteFile(
        "labels",
        "aparc-reduced.nii.gz",
        _STANFORD + "aparc-reduced.nii.gz",
        "6ab1ac058eebc6a3ea065b45dc329c3da6827c69f069f5cbb9d66e5a315202a6",
        59648,
    ),
)


def _negate_x(bvecs: bytes) -> bytes:
    """Returns an FSL bvecs file with the sign of its x components changed.

    Args:
        bvecs: The content of the file.
    """
    rows = bvecs.decode("ascii").split("\n")
    rows[0] = " ".join(_negated(value) for value in rows[0].split())
    return "\n".join(rows).encode("ascii")


def _negated(value: str) -> str:
    """Returns the number written in `value` with its sign changed, its digits kept."""
    return value[1:] if value.startswith("-") else "-" + value


# One zip archive of the challenge's data, without its field maps. The x components
# of the b-vectors are flipped when the files are extracted.
ISMRM2015_FILES = (
    RemoteFile(
        "archive",
        "ISMRM_2015_Tracto_challenge_data_v1_1.zip",
        _ISMRM + "ISMRM_2015_Tracto_challenge_data_v1_1.zip?download=1",
        "d2a3d08cb0ef2efbc386041c2e0a8b3305fd25f68e4e721486c0ac30658f8548",
        40657189,
        members=(
            Member("dwi", "dwi.nii.gz", _ISMRM_DATA + "Diffusion.nii.gz", 28601266),
            Member("bvals", "dwi.bvals", _ISMRM_DATA + "Diffusion.bvals", 198),
            Member(
                "bvecs",
                "dwi.bvecs",
                _ISMRM_DATA + "Diffusion.bvecs",
                893,
                _negate_x,
                "184194dd89940f48607063f4bd7e912d67e90c717caf3c6047d4ab467f6bc83c",
            ),
            Member("t1", "t1.nii.gz", _ISMRM_DATA + "T1.nii.gz", 7313549),
        ),
    ),
)
