"""Reading and writing volumes, tractograms, surfaces and gradient tables."""

from larmor.io.bvals_bvecs import (
    read_b_matrix,
    read_bvals_bvecs,
    read_mrtrix_gradients,
    read_mrtrix_response,
    write_b_matrix,
    write_bvals_bvecs,
    write_mrtrix_gradients,
)
from larmor.io.dwi import load_dwi
from larmor.io.formats.lookup_table import read_lookup_table
from larmor.io.formats.matlab import MatlabEntry, MatlabFile, read_matlab
from larmor.io.formats.mif import has_mif_gradients
from larmor.io.formats.streamline_weights import (
    read_streamline_weights,
    write_streamline_weights,
)
from larmor.io.formats.tsf import read_track_timestamp, read_tsf, write_tsf
from larmor.io.load_save import (
    SURFACE_EXTENSIONS,
    TRACTOGRAM_EXTENSIONS,
    VOLUME_EXTENSIONS,
    file_kind,
    iter_tractogram,
    load_grid,
    load_header,
    load_surface,
    load_tractogram,
    load_tractogram_subset,
    load_volume,
    open_tractogram_writer,
    output_files,
    save_surface,
    save_tractogram,
    save_volume,
)
from larmor.io.mritoolkit import (
    MRIToolkitFit,
    read_mritoolkit_fit,
    read_mritoolkit_frame,
    read_mritoolkit_grid,
    read_mritoolkit_kind,
    read_mritoolkit_map,
    read_mritoolkit_peaks,
    read_mritoolkit_tracts,
    read_mritoolkit_volume,
)
from larmor.io.reader import Progress, ReadCancelledError
from larmor.io.writer import BatchWriter

__all__ = [
    "SURFACE_EXTENSIONS",
    "TRACTOGRAM_EXTENSIONS",
    "VOLUME_EXTENSIONS",
    "BatchWriter",
    "MRIToolkitFit",
    "MatlabEntry",
    "MatlabFile",
    "Progress",
    "ReadCancelledError",
    "file_kind",
    "has_mif_gradients",
    "iter_tractogram",
    "load_dwi",
    "load_grid",
    "load_header",
    "load_surface",
    "load_tractogram",
    "load_tractogram_subset",
    "load_volume",
    "open_tractogram_writer",
    "output_files",
    "read_b_matrix",
    "read_bvals_bvecs",
    "read_lookup_table",
    "read_matlab",
    "read_mritoolkit_fit",
    "read_mritoolkit_frame",
    "read_mritoolkit_grid",
    "read_mritoolkit_kind",
    "read_mritoolkit_map",
    "read_mritoolkit_peaks",
    "read_mritoolkit_tracts",
    "read_mritoolkit_volume",
    "read_mrtrix_gradients",
    "read_mrtrix_response",
    "read_streamline_weights",
    "read_track_timestamp",
    "read_tsf",
    "save_surface",
    "save_tractogram",
    "save_volume",
    "write_b_matrix",
    "write_bvals_bvecs",
    "write_mrtrix_gradients",
    "write_streamline_weights",
    "write_tsf",
]
