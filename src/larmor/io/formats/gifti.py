"""Reads and writes GIfTI surfaces and values per vertex."""

import base64
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path
from typing import Literal

import numpy as np

from larmor.io.formats.freesurfer import (
    grid_from_volume_info,
    tkregister_to_scanner,
)
from larmor.types import Grid, Surface

# The intents of the vertex and face arrays of a mesh.
_POINTSET = "NIFTI_INTENT_POINTSET"
_TRIANGLE = "NIFTI_INTENT_TRIANGLE"

# The GIfTI data types Larmor reads and writes, and their numpy types.
_DTYPES: dict[str, type[np.generic]] = {
    "NIFTI_TYPE_UINT8": np.uint8,
    "NIFTI_TYPE_INT32": np.int32,
    "NIFTI_TYPE_FLOAT32": np.float32,
}


def read_gifti_surface(path: str | Path) -> Surface:
    """Reads a GIfTI surface (.surf.gii) and any values per vertex stored with it.

    When the pointset has FreeSurfer's VolGeom metadata, its vertices are taken as
    tkregister RAS and converted to scanner RAS, and that volume becomes the
    reference, unless the pointset's DataSpace already says scanner space. Other
    vertices are taken as stored, in scanner RAS; the file's coordinate transform is
    not applied.

    Args:
        path: The file.

    Returns:
        Surface: The mesh, with the file's other arrays in `vertex_data`, named
        after their "Name" metadata or "data<index>". The reference is the VolGeom
        volume, or None without one.

    Raises:
        ValueError: If the file is not GIfTI XML, does not hold exactly one pointset
            and one triangle array, an array cannot be decoded, or its VolGeom
            metadata is malformed.
    """
    try:
        return _read_surface(path)
    except (ValueError, ET.ParseError) as error:
        raise ValueError(f"{path}: {error}") from error


def _read_surface(path: str | Path) -> Surface:
    """Reads the GIfTI surface at `path`, with errors that do not name the file."""
    arrays = _data_arrays(path)
    points = [a for a in arrays if a.get("Intent") == _POINTSET]
    triangles = [a for a in arrays if a.get("Intent") == _TRIANGLE]
    if len(points) != 1 or len(triangles) != 1:
        raise ValueError(
            "Expected one pointset and one triangle array, got "
            f"{len(points)} and {len(triangles)}; read a file of values only with "
            "read_gifti_values."
        )
    vertex_data = _named_values(
        [a for a in arrays if a is not points[0] and a is not triangles[0]], arrays
    )
    vertices = _decode(points[0])
    reference = _volume_geometry(_metadata(points[0]))
    space = points[0].findtext("./CoordinateSystemTransformMatrix/DataSpace")
    if reference is not None and (space or "").strip() != "NIFTI_XFORM_SCANNER_ANAT":
        vertices = tkregister_to_scanner(vertices, reference)
    return Surface(
        vertices=vertices,
        faces=_decode(triangles[0]),
        vertex_data=vertex_data,
        reference=reference,
    )


def read_gifti_values(
    path: str | Path, *, unnamed: str | None = None
) -> dict[str, np.ndarray]:
    """Reads the values per vertex of a GIfTI file (.shape.gii, .func.gii, .label.gii).

    Args:
        path: The file.
        unnamed: The name of an array without "Name" metadata; with several such
            arrays, the array's index is added. Defaults to None, "data<index>".

    Returns:
        dict[str, np.ndarray]: Each array that is not a mesh, by its "Name"
        metadata or as `unnamed` says. Label files give the integer keys; their
        label table is not read.

    Raises:
        ValueError: If the file is not GIfTI XML or an array cannot be decoded.
    """
    try:
        arrays = _data_arrays(path)
        values = [a for a in arrays if a.get("Intent") not in (_POINTSET, _TRIANGLE)]
        return _named_values(values, arrays, unnamed)
    except (ValueError, ET.ParseError) as error:
        raise ValueError(f"{path}: {error}") from error


def write_gifti_surface(surface: Surface, path: str | Path) -> None:
    """Writes a surface and its values per vertex to a GIfTI file (.surf.gii).

    Vertices are written as float32 in scanner space, faces as int32, and each
    `vertex_data` array as its own data array named by its key. The reference is
    not written, since GIfTI has no place for it.

    Args:
        surface: The surface to write.
        path: The file.

    Raises:
        ValueError: If a `vertex_data` array holds integers beyond int32.
    """
    root = _root(2 + len(surface.vertex_data))
    _encode(root, _POINTSET, surface.vertices.astype(np.float32), scanner_space=True)
    _encode(root, _TRIANGLE, surface.faces.astype(np.int32))
    for name, values in surface.vertex_data.items():
        _encode(root, *_values_intent(values), name=name)
    _write(root, path)


def write_gifti_values(values: dict[str, np.ndarray], path: str | Path) -> None:
    """Writes named values per vertex to a GIfTI file (.shape.gii or .func.gii).

    Args:
        values: Arrays whose first axis is the vertices.
        path: The file.

    Raises:
        ValueError: If an array is a scalar or holds integers beyond int32.
    """
    root = _root(len(values))
    for name, array in values.items():
        if np.ndim(array) == 0:
            raise ValueError(f"Values {name!r} must be an array, got a scalar.")
        _encode(root, *_values_intent(np.asarray(array)), name=name)
    _write(root, path)


def _data_arrays(path: str | Path) -> list[ET.Element]:
    """Returns the file's DataArray elements, in order."""
    root = ET.parse(path).getroot()
    if root.tag != "GIFTI":
        raise ValueError(f"Expected a GIfTI file, got root element <{root.tag}>.")
    return root.findall("DataArray")


def _named_values(
    chosen: list[ET.Element], arrays: list[ET.Element], unnamed: str | None = None
) -> dict[str, np.ndarray]:
    """Returns the DataArrays `chosen` from `arrays`, decoded, by name.

    An array without a name is "data<index>", or `unnamed` when given and it is the
    only one without a name (`unnamed<index>` otherwise); a repeated name gets its
    index in `arrays`.
    """
    nameless = sum(_name(array) is None for array in chosen)
    values: dict[str, np.ndarray] = {}
    for array in chosen:
        index = arrays.index(array)
        name = _name(array)
        if name is None:
            if unnamed is None:
                name = f"data{index}"
            else:
                name = unnamed if nameless == 1 else f"{unnamed}{index}"
        if name in values:
            name = f"{name}{index}"
        values[name] = _decode(array)
    return values


def _decode(array: ET.Element) -> np.ndarray:
    """Returns the values of a DataArray, with its shape, in native byte order."""
    data_type = array.get("DataType", "")
    if data_type not in _DTYPES:
        raise ValueError(f"Unsupported GIfTI data type {data_type!r}.")
    dtype = np.dtype(_DTYPES[data_type])
    shape = tuple(
        int(array.get(f"Dim{i}", "0"))
        for i in range(int(array.get("Dimensionality", "1")))
    )
    text: str = (array.findtext("Data") or "").strip()

    match array.get("Encoding"):
        case "ASCII":
            flat = np.array(text.split(), dtype=dtype)
        case "Base64Binary" | "GZipBase64Binary" as encoding:
            raw: bytes = base64.b64decode(text)
            if encoding == "GZipBase64Binary":
                # Reads zlib, as the standard says, and gzip, which some writers use.
                raw = zlib.decompress(raw, wbits=zlib.MAX_WBITS | 32)
            byte_order: Literal["<", ">"] = (
                ">" if array.get("Endian") == "BigEndian" else "<"
            )
            flat = np.frombuffer(raw, dtype=dtype.newbyteorder(byte_order))
        case other:
            raise ValueError(
                "Expected ASCII, Base64Binary or GZipBase64Binary encoding, "
                f"got {other}."
            )

    if flat.size != int(np.prod(shape)):
        raise ValueError(f"A GIfTI array of shape {shape} holds {flat.size} values.")
    order: Literal["C", "F"] = (
        "F" if array.get("ArrayIndexingOrder") == "ColumnMajorOrder" else "C"
    )
    decoded: np.ndarray = flat.reshape(shape, order=order).astype(dtype)
    return decoded


def _metadata(array: ET.Element) -> dict[str, str]:
    """Returns the array's metadata entries as names and values."""
    return {
        (entry.findtext("Name") or "").strip(): (entry.findtext("Value") or "").strip()
        for entry in array.findall("./MetaData/MD")
    }


def _name(array: ET.Element) -> str | None:
    """Returns the value of the array's "Name" metadata, if it has one."""
    return _metadata(array).get("Name") or None


def _volume_geometry(metadata: dict[str, str]) -> Grid | None:
    """Returns the grid in FreeSurfer's VolGeom metadata, or None if any is missing.

    Raises:
        ValueError: If a VolGeom value is not a number.
    """
    names = ["Width", "Height", "Depth", "Xsize", "Ysize", "Zsize"]
    names += [f"{axis}_{ras}" for axis in "XYZC" for ras in "RAS"]
    if not all(f"VolGeom{name}" in metadata for name in names):
        return None
    try:
        value = {name: float(metadata[f"VolGeom{name}"]) for name in names}
    except ValueError as error:
        raise ValueError(f"Malformed VolGeom metadata: {error}.") from error
    shape = (int(value["Width"]), int(value["Height"]), int(value["Depth"]))
    sizes = np.array([value["Xsize"], value["Ysize"], value["Zsize"]])
    directions = np.array([[value[f"{axis}_{ras}"] for axis in "XYZ"] for ras in "RAS"])
    centre = np.array([value[f"C_{ras}"] for ras in "RAS"])
    return grid_from_volume_info(shape, sizes, directions, centre)


def _values_intent(values: np.ndarray) -> tuple[str, np.ndarray]:
    """Returns the intent of values per vertex and the values in the type to write."""
    if np.issubdtype(values.dtype, np.floating):
        return "NIFTI_INTENT_SHAPE", values.astype(np.float32)
    if values.dtype in (np.bool_, np.uint8):
        return "NIFTI_INTENT_NONE", values.astype(np.uint8)
    if values.size > 0 and (
        values.min() < np.iinfo(np.int32).min or values.max() > np.iinfo(np.int32).max
    ):
        raise ValueError("GIfTI stores integers as int32; these values do not fit.")
    return "NIFTI_INTENT_NONE", values.astype(np.int32)


def _root(count: int) -> ET.Element:
    """Returns an empty GIFTI element for `count` data arrays."""
    root = ET.Element("GIFTI", Version="1.0", NumberOfDataArrays=str(count))
    ET.SubElement(root, "MetaData")
    ET.SubElement(root, "LabelTable")
    return root


def _encode(
    root: ET.Element,
    intent: str,
    values: np.ndarray,
    name: str | None = None,
    scanner_space: bool = False,
) -> None:
    """Appends `values` to `root` as a compressed DataArray of the given intent.

    `name` goes into the array's "Name" metadata. With `scanner_space`, the array
    states that it is in scanner space, as a pointset should.
    """
    data_type = next(k for k, v in _DTYPES.items() if np.dtype(v) == values.dtype)
    attributes = {
        "Intent": intent,
        "DataType": data_type,
        "ArrayIndexingOrder": "RowMajorOrder",
        "Dimensionality": str(values.ndim),
        **{f"Dim{i}": str(n) for i, n in enumerate(values.shape)},
        "Encoding": "GZipBase64Binary",
        "Endian": "LittleEndian",
        "ExternalFileName": "",
        "ExternalFileOffset": "0",
    }
    array = ET.SubElement(root, "DataArray", attributes)
    metadata = ET.SubElement(array, "MetaData")
    if name is not None:
        entry = ET.SubElement(metadata, "MD")
        ET.SubElement(entry, "Name").text = "Name"
        ET.SubElement(entry, "Value").text = name
    if scanner_space:
        space = ET.SubElement(array, "CoordinateSystemTransformMatrix")
        ET.SubElement(space, "DataSpace").text = "NIFTI_XFORM_SCANNER_ANAT"
        ET.SubElement(space, "TransformedSpace").text = "NIFTI_XFORM_SCANNER_ANAT"
        ET.SubElement(space, "MatrixData").text = "\n".join(
            " ".join(str(x) for x in row) for row in np.eye(4)
        )
    little = np.ascontiguousarray(values, dtype=values.dtype.newbyteorder("<"))
    ET.SubElement(array, "Data").text = base64.b64encode(
        zlib.compress(little.tobytes())
    ).decode("ascii")


def _write(root: ET.Element, path: str | Path) -> None:
    """Writes the XML tree to `path` with an XML declaration."""
    ET.ElementTree(root).write(path, encoding="UTF-8", xml_declaration=True)
