"""Readers and writers of each file format, with the options only one format has."""

from larmor.io.formats.freesurfer import (
    read_freesurfer_label,
    read_freesurfer_surface,
    read_freesurfer_values,
    write_freesurfer_surface,
    write_freesurfer_values,
)
from larmor.io.formats.gifti import (
    read_gifti_surface,
    read_gifti_values,
    write_gifti_surface,
    write_gifti_values,
)
from larmor.io.formats.mif import (
    read_mif,
    read_mif_gradients,
    read_mif_header,
    write_mif,
)
from larmor.io.formats.nifti import read_nifti, read_nifti_header, write_nifti
from larmor.io.formats.tck import read_tck, write_tck
from larmor.io.formats.trk import read_trk, write_trk
from larmor.io.formats.vtk import read_vtk, write_vtk

__all__ = [
    "read_freesurfer_label",
    "read_freesurfer_surface",
    "read_freesurfer_values",
    "read_gifti_surface",
    "read_gifti_values",
    "read_mif",
    "read_mif_gradients",
    "read_mif_header",
    "read_nifti",
    "read_nifti_header",
    "read_tck",
    "read_trk",
    "read_vtk",
    "write_freesurfer_surface",
    "write_freesurfer_values",
    "write_gifti_surface",
    "write_gifti_values",
    "write_mif",
    "write_nifti",
    "write_tck",
    "write_trk",
    "write_vtk",
]
