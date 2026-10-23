"""Lists the variables of MATLAB `.mat` files and reads them as Python values."""

import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import h5py
    from scipy import sparse

# The largest cell or struct array whose elements are listed one by one.
MAX_LISTED_ELEMENTS = 100

# A value read from a `.mat` file.
type MatlabValue = (
    np.ndarray
    | sparse.csc_array
    | str
    | list[str]
    | dict[str, MatlabValue]
    | list[MatlabValue]
)
# An HDF5 object of a version 7.3 file, and a value as scipy reads a version 4-7 one.
type _H5Object = h5py.Group | h5py.Dataset
type _Raw = np.ndarray | sparse.csc_array

# The numpy type of each numeric MATLAB class.
_NUMERIC_CLASSES = {
    "double": np.float64,
    "single": np.float32,
    "int8": np.int8,
    "uint8": np.uint8,
    "int16": np.int16,
    "uint16": np.uint16,
    "int32": np.int32,
    "uint32": np.uint32,
    "int64": np.int64,
    "uint64": np.uint64,
    "logical": np.bool_,
}
# The MATLAB class of each numpy type.
_CLASS_OF_DTYPE: dict[np.dtype, str] = {
    np.dtype(t): c for c, t in _NUMERIC_CLASSES.items()
}
_CLASS_OF_DTYPE[np.dtype(np.complex128)] = "double"
_CLASS_OF_DTYPE[np.dtype(np.complex64)] = "single"

# One step of a path: `.field`, `{i}` or `(i)`.
_STEP = re.compile(r"\.([A-Za-z]\w*)|\{(\d+)\}|\((\d+)\)")
# A MATLAB variable name.
_NAME = re.compile(r"[A-Za-z]\w*")


@dataclass(frozen=True)
class MatlabEntry:
    """One variable of a `.mat` file, or a struct field or cell inside one.

    Attributes:
        path: Where it is, as a variable name followed by `.field` for a struct
            field, `{i}` for a cell element and `(i)` for a struct array element,
            with MATLAB's 1-based linear indices (for example `Tracts{3}`).
        shape: Its MATLAB size; empty for an object whose size is not stored.
        matlab_class: Its MATLAB class (`double`, `logical`, `char`, `struct`,
            `cell`, or the class of an object such as `string`).
        sparse: Whether it is a sparse matrix.
    """

    path: str
    shape: tuple[int, ...]
    matlab_class: str
    sparse: bool = False


def read_matlab(path: str | Path) -> MatlabFile:
    """Opens a MATLAB `.mat` file (version 4, 5 to 7, or 7.3) for listing and reading.

    Args:
        path: The `.mat` file.

    Returns:
        The file, whose variables are read on request.

    Raises:
        ValueError: If the file is not a MATLAB file.
    """
    return MatlabFile(path)


class MatlabFile:
    """A `.mat` file whose variables can be listed and read one at a time.

    Values come back the same for every version of the file. Numeric and logical
    arrays are numpy arrays with MATLAB's shape, so `a[i, j, k]` is MATLAB's
    `a(i+1, j+1, k+1)`. Sparse matrices are `scipy.sparse.csc_array`, a char row is
    a `str` and a char matrix a list of its rows, a scalar struct is a dict of its
    fields, and struct arrays and cell arrays are lists in MATLAB's linear
    (column-major) order, their shape given by `entries`. An empty char is `""` and
    an empty cell or struct is `[]`. MATLAB objects (strings, tables, function
    handles) cannot be read; reading one raises an error naming its class.

    A variable is read from the file only when it is first read or listed.

    Args:
        path: The `.mat` file.
    """

    def __init__(self, path: str | Path) -> None:
        """Opens the file and lists its variables."""
        self.path = Path(path)
        self.version = _version(self.path)
        self._v5: dict[str, _Raw] = {}  # the variables read so far
        if self.version == "7.3":
            with self._h5() as f:
                self._names = [n for n in f if not n.startswith("#")]
        else:
            self._names = self._list_v5()

    @property
    def names(self) -> list[str]:
        """The names of the file's variables, in the file's order."""
        return list(self._names)

    def __contains__(self, name: object) -> bool:
        """Returns whether the file has a variable of this name."""
        return name in self._names

    def __iter__(self) -> Iterator[str]:
        """Iterates over the names of the file's variables."""
        return iter(self.names)

    def __getitem__(self, path: str) -> MatlabValue:
        """Reads the value at a path, as `read`."""
        return self.read(path)

    def __repr__(self) -> str:
        """Returns the file, its version and its variable names as text."""
        return f"MatlabFile({str(self.path)!r}, version {self.version}, {self._names})"

    def entries(self, recursive: bool = True) -> list[MatlabEntry]:
        """Lists the variables with their shape and class, without reading their data.

        Args:
            recursive: If True, also list the fields of structs and the elements of
                cells and struct arrays with at most 100 elements, after the entry
                that holds them. Defaults to True.

        Returns:
            The entries, in the file's order.
        """
        if self.version == "7.3":
            with self._h5() as f:
                return [
                    e
                    for n in self._names
                    for e in _h5_entries(f, _H5Node(f[n]), n, recursive)
                ]
        return [
            e for n in self._names for e in _v5_entries(self._load_v5(n), n, recursive)
        ]

    def entry(self, path: str) -> MatlabEntry:
        """Returns the shape and class of the value at a path, without its data.

        Args:
            path: A variable name, optionally followed by `.field`, `{i}` and `(i)`
                steps as in `MatlabFile.read`.

        Returns:
            MatlabEntry: Its shape and class.
        """
        steps = _parse(path)
        if self.version == "7.3":
            with self._h5() as f:
                return _h5_describe(f, self._h5_node(f, path, steps), path)
        return _v5_describe(self._v5_node(path, steps), path)

    def read(self, path: str) -> MatlabValue:
        """Reads a variable, or a struct field or cell element inside one.

        Args:
            path: A variable name, optionally followed by `.field`, `{i}` and `(i)`
                steps as in `MatlabEntry.path`.

        Returns:
            The value, converted as described in the class.

        Raises:
            KeyError: If there is nothing at the path.
            ValueError: If the value is a MATLAB object, or holds one.
        """
        steps = _parse(path)
        if self.version == "7.3":
            with self._h5() as f:
                return _h5_value(f, self._h5_node(f, path, steps), path)
        return _v5_value(self._v5_node(path, steps), path)

    def read_array(self, path: str) -> np.ndarray:
        """Reads a numeric or logical array.

        Args:
            path: As in `MatlabFile.read`.

        Returns:
            The array, with MATLAB's shape.

        Raises:
            ValueError: If the value is not a numeric or logical array.
        """
        value = self.read(path)
        if not isinstance(value, np.ndarray):
            raise ValueError(
                f"{self.path}: {path} is a {self.entry(path).matlab_class} value, "
                "not a numeric array."
            )
        return value

    def _h5(self) -> h5py.File:
        """Opens the file with h5py."""
        import h5py

        return h5py.File(self.path, "r")

    def _list_v5(self) -> list[str]:
        """Returns the names of a version 4 to 7 file's variables.

        Only the names are read, unless scipy cannot list the file, which is then
        read whole.
        """
        from scipy.io import whosmat

        try:
            return [name for name, _, _ in whosmat(self.path)]
        except TypeError, ValueError:
            # scipy cannot list some MATLAB objects, but can read them whole.
            self._v5 = self._read_v5(None)
            return list(self._v5)

    def _load_v5(self, name: str) -> _Raw:
        """Returns one variable as scipy reads it, reading it the first time only."""
        if name not in self._v5:
            self._v5 |= self._read_v5([name])
        return self._v5[name]

    def _read_v5(self, names: list[str] | None) -> dict[str, _Raw]:
        """Returns the variables `names`, or all if None, as scipy reads them.

        Complex values are kept.
        """
        import warnings

        from scipy.io import loadmat

        options: dict[str, Any] = {
            "chars_as_strings": False,
            "spmatrix": False,
            "variable_names": names,
        }
        if self.version == "5":
            # MATLAB stores characters as UTF-16, in the file's byte order ("IM" at
            # bytes 126-127 is little-endian).
            with open(self.path, "rb") as f:
                f.seek(126)
                little = f.read(2) == b"IM"
            options["uint16_codec"] = "utf-16-le" if little else "utf-16-be"
        with warnings.catch_warnings():
            # Silences scipy's warnings about its sparse return type and about the
            # cast of complex values, which the second read restores.
            warnings.simplefilter("ignore")
            typed = loadmat(self.path, mat_dtype=True, **options)
            stored = loadmat(self.path, mat_dtype=False, **options)
        return {
            k: _restore_complex(v, stored[k])
            for k, v in typed.items()
            if not k.startswith("__")
        }

    def _h5_node(
        self, f: h5py.File, path: str, steps: list[tuple[str, str]]
    ) -> _H5Node:
        """Returns the HDF5 node at `path`, parsed into `steps`."""
        name = steps[0][1]
        if name not in self._names:
            raise KeyError(f"{self.path}: no variable {name!r}.")
        node = _H5Node(f[name])
        for kind, key in steps[1:]:
            node = _h5_step(f, node, kind, key, path, self.path)
        return node

    def _v5_node(self, path: str, steps: list[tuple[str, str]]) -> _Raw:
        """Returns the scipy value at `path`, parsed into `steps`."""
        name = steps[0][1]
        if name not in self._names:
            raise KeyError(f"{self.path}: no variable {name!r}.")
        node = self._load_v5(name)
        for kind, key in steps[1:]:
            node = _v5_step(node, kind, key, path, self.path)
        return node


def _version(path: Path) -> str:
    """Returns the file's MATLAB format version: "4", "5" (MATLAB 5 to 7) or "7.3"."""
    with open(path, "rb") as f:
        head = f.read(520)
    if head.startswith(b"MATLAB 7.3") or head[512:520] == b"\x89HDF\r\n\x1a\n":
        return "7.3"
    if head.startswith(b"MATLAB 5.0"):
        return "5"
    if _is_v4(head):
        return "4"
    raise ValueError(f"{path}: not a MATLAB .mat file.")


def _is_v4(head: bytes) -> bool:
    """Returns whether the bytes `head` start with a version 4 matrix header.

    The header is five int32 (type, rows, columns, imaginary flag, name length) in
    the file's byte order, then the name, ending with a zero byte.
    """
    if len(head) < 21:
        return False
    for order, machine in (("<", 0), (">", 1)):
        kind, rows, columns, imaginary, length = np.frombuffer(
            head[:20], dtype=f"{order}i4"
        )
        if not (
            kind // 1000 == machine
            and kind % 1000 // 100 == 0
            and kind % 100 // 10 <= 5
            and kind % 10 <= 2
            and min(rows, columns) >= 0
            and imaginary in (0, 1)
            and 1 <= length <= 64
            and len(head) >= 20 + length
        ):
            continue
        name = head[20 : 20 + length]
        if name[-1] == 0 and _NAME.fullmatch(name[:-1].decode("latin-1")):
            return True
    return False


def _parse(path: str) -> list[tuple[str, str]]:
    """Splits a path into (kind, key) steps; the first is ("name", variable)."""
    head = _NAME.match(path)
    if head is None:
        raise KeyError(f"not a MATLAB path: {path!r}")
    steps = [("name", head.group())]
    pos = head.end()
    while pos < len(path):
        m = _STEP.match(path, pos)
        if m is None:
            raise KeyError(f"not a MATLAB path: {path!r}")
        field, cell, element = m.groups()
        if field is not None:
            steps.append(("field", field))
        elif cell is not None:
            steps.append(("cell", cell))
        else:
            steps.append(("element", element))
        pos = m.end()
    return steps


def _object_error(path: str, matlab_class: str) -> ValueError:
    """Returns the error for a MATLAB object of class `matlab_class` at `path`."""
    return ValueError(
        f"{path} is a MATLAB object of class {matlab_class!r}, which cannot be read "
        "outside MATLAB; convert it to a plain value (numeric, char, struct or cell) "
        "in MATLAB and save the file again."
    )


def _index(key: str, count: int, path: str) -> int:
    """Returns the 0-based index of a 1-based MATLAB index, checked against `count`."""
    i = int(key) - 1
    if not 0 <= i < count:
        raise KeyError(f"{path}: index {key} is outside 1..{count}.")
    return i


def _decode_chars(codes: np.ndarray) -> str | list[str]:
    """Decodes MATLAB char data, UTF-16 in MATLAB's shape, as a string or its rows."""
    if codes.size == 0:
        return ""
    if codes.ndim != 2:
        raise ValueError("char arrays with more than two dimensions are not read.")
    rows = [
        codes[r].astype("<u2").tobytes().decode("utf-16-le", "surrogatepass")
        for r in range(codes.shape[0])
    ]
    return rows[0] if len(rows) == 1 else rows


# Version 7.3 (HDF5). MATLAB writes each array with its dimensions reversed, so the
# transpose of what h5py reads has MATLAB's shape and order.


@dataclass(frozen=True)
class _H5Node:
    """An HDF5 object, or for a struct array element the group and the element."""

    obj: _H5Object
    element: int | None = None


def _attr(obj: _H5Object, name: str) -> str | None:
    """Returns the string attribute `name` of an HDF5 object, or None."""
    value = obj.attrs.get(name)
    if value is None:
        return None
    return value.decode() if isinstance(value, bytes) else str(value)


def _is_struct_array(group: _H5Object) -> bool:
    """Returns whether a struct group holds a struct array rather than one struct."""
    import h5py

    fields = [group[k] for k in group]
    return bool(fields) and all(
        isinstance(d, h5py.Dataset)
        and d.dtype == h5py.ref_dtype
        and "MATLAB_class" not in d.attrs
        for d in fields
    )


def _h5_fields(group: _H5Object) -> list[str]:
    """Returns the field names of a struct group, in MATLAB's order when stored."""
    stored = group.attrs.get("MATLAB_fields")
    if stored is None:
        return list(group)
    return [b"".join(name).decode() for name in stored]


def _h5_refs(dataset: _H5Object) -> np.ndarray:
    """Returns the references of a cell or struct array field, in linear order."""
    return np.asarray(dataset[()]).ravel()


def _h5_step(
    f: h5py.File, node: _H5Node, kind: str, key: str, path: str, file: Path
) -> _H5Node:
    """Returns the node one path step (`kind`, `key`) below `node`."""
    import h5py

    obj = node.obj
    cls = _attr(obj, "MATLAB_class")
    if kind == "field":
        if not isinstance(obj, h5py.Group) or cls != "struct":
            raise KeyError(f"{file}: {path}: .{key} of a value that is not a struct.")
        if node.element is None and _is_struct_array(obj):
            raise KeyError(f"{file}: {path}: .{key} of a struct array; index it first.")
        if key not in obj:
            raise KeyError(f"{file}: {path}: no field {key!r}.")
        if node.element is None:
            return _H5Node(obj[key])
        return _H5Node(f[_h5_refs(obj[key])[node.element]])
    if kind == "cell":
        if cls != "cell" or not isinstance(obj, h5py.Dataset) or _is_empty(obj):
            raise KeyError(f"{file}: {path}: {{{key}}} of a value that is not a cell.")
        refs = _h5_refs(obj)
        return _H5Node(f[refs[_index(key, len(refs), path)]])
    if not isinstance(obj, h5py.Group) or cls != "struct" or node.element is not None:
        raise KeyError(f"{file}: {path}: ({key}) of a value that is not a struct.")
    count = _h5_struct_count(obj)
    return _H5Node(obj, _index(key, count, path))


def _is_empty(obj: _H5Object) -> bool:
    """Returns whether an HDF5 dataset holds an empty MATLAB value."""
    return bool(obj.attrs.get("MATLAB_empty", 0))


def _h5_struct_count(group: _H5Object) -> int:
    """Returns the number of elements of a struct group, 1 for a scalar struct."""
    if not _is_struct_array(group):
        return 1
    return int(group[next(iter(group))].size)


def _h5_shape(obj: _H5Object) -> tuple[int, ...]:
    """Returns the MATLAB shape of an HDF5 object."""
    import h5py

    if isinstance(obj, h5py.Group):
        rows = obj.attrs.get("MATLAB_sparse")
        if rows is not None:
            return (int(rows), len(obj["jc"]) - 1)
        if _attr(obj, "MATLAB_class") == "struct" and _is_struct_array(obj):
            return tuple(int(n) for n in obj[next(iter(obj))].shape[::-1])
        return (1, 1)
    if _is_empty(obj):
        return tuple(int(n) for n in obj[()])
    return tuple(int(n) for n in obj.shape[::-1])


def _h5_describe(f: h5py.File, node: _H5Node, path: str) -> MatlabEntry:
    """Returns the entry of an HDF5 node."""
    obj = node.obj
    cls = _attr(obj, "MATLAB_class") or "unknown"
    if cls == "canonical empty":
        return MatlabEntry(path, (0, 0), "double")
    if node.element is not None:
        return MatlabEntry(path, (1, 1), "struct")
    if "MATLAB_object_decode" in obj.attrs:
        return MatlabEntry(path, (), cls)
    sparse = "MATLAB_sparse" in obj.attrs
    return MatlabEntry(path, _h5_shape(obj), cls, sparse)


def _h5_children(f: h5py.File, node: _H5Node, path: str) -> list[tuple[_H5Node, str]]:
    """Returns the struct fields, cells or struct array elements below a node.

    A cell or struct array with more than MAX_LISTED_ELEMENTS elements gives none.
    """
    import h5py

    obj = node.obj
    cls = _attr(obj, "MATLAB_class")
    if "MATLAB_object_decode" in obj.attrs:
        return []
    if cls == "struct" and isinstance(obj, h5py.Group):
        if node.element is None and _is_struct_array(obj):
            count = _h5_struct_count(obj)
            if count > MAX_LISTED_ELEMENTS:
                return []
            return [(_H5Node(obj, i), f"{path}({i + 1})") for i in range(count)]
        return [
            (_h5_step(f, node, "field", k, path, Path(f.filename)), f"{path}.{k}")
            for k in _h5_fields(obj)
        ]
    if cls == "cell" and isinstance(obj, h5py.Dataset) and not _is_empty(obj):
        refs = _h5_refs(obj)
        if len(refs) > MAX_LISTED_ELEMENTS:
            return []
        return [(_H5Node(f[r]), f"{path}{{{i + 1}}}") for i, r in enumerate(refs)]
    return []


def _h5_entries(
    f: h5py.File, node: _H5Node, path: str, recursive: bool
) -> list[MatlabEntry]:
    """Returns the entry of a node, then those below it if `recursive`."""
    found = [_h5_describe(f, node, path)]
    if recursive:
        for child, child_path in _h5_children(f, node, path):
            found += _h5_entries(f, child, child_path, recursive)
    return found


def _h5_value(f: h5py.File, node: _H5Node, path: str) -> MatlabValue:
    """Converts an HDF5 node to a Python value."""
    import h5py

    obj = node.obj
    cls = _attr(obj, "MATLAB_class")
    if cls == "canonical empty":
        return np.zeros((0, 0))
    if cls is None:
        raise ValueError(f"{path}: an HDF5 object without a MATLAB class.")
    if "MATLAB_object_decode" in obj.attrs:
        raise _object_error(path, cls)
    if isinstance(obj, h5py.Group):
        return _h5_group_value(f, node, path, cls)
    if _is_empty(obj):
        return _empty(cls, _h5_shape(obj), path)
    if cls == "cell":
        return [
            _h5_value(f, _H5Node(f[r]), f"{path}{{{i + 1}}}")
            for i, r in enumerate(_h5_refs(obj))
        ]
    if cls == "char":
        return _decode_chars(np.asarray(obj[()]).T)
    if cls not in _NUMERIC_CLASSES:
        raise _object_error(path, cls)
    return _h5_numeric(obj, cls)


def _h5_group_value(f: h5py.File, node: _H5Node, path: str, cls: str) -> MatlabValue:
    """Converts an HDF5 group (a sparse matrix or a struct) to a Python value."""
    obj = node.obj
    if "MATLAB_sparse" in obj.attrs:
        return _h5_sparse(obj, cls)
    if cls != "struct":
        raise _object_error(path, cls)
    if node.element is None and _is_struct_array(obj):
        return [
            _h5_value(f, _H5Node(obj, i), f"{path}({i + 1})")
            for i in range(_h5_struct_count(obj))
        ]
    return {
        k: _h5_value(
            f, _h5_step(f, node, "field", k, path, Path(f.filename)), f"{path}.{k}"
        )
        for k in _h5_fields(obj)
    }


def _h5_numeric(obj: _H5Object, cls: str) -> np.ndarray:
    """Reads a numeric or logical dataset as an array in MATLAB's axis order."""
    data = np.asarray(obj[()])
    if data.dtype.names is not None:
        data = data["real"] + 1j * data["imag"]
    if cls == "logical":
        data = data.astype(bool)
    return data.T


def _h5_sparse(group: _H5Object, cls: str) -> sparse.csc_array:
    """Reads a sparse matrix group as a CSC array."""
    from scipy import sparse

    rows = int(group.attrs["MATLAB_sparse"])
    jc = np.asarray(group["jc"][()], dtype=np.int64)
    nnz = int(jc[-1])
    if "data" in group:
        data = np.asarray(group["data"][()])
        ir = np.asarray(group["ir"][()], dtype=np.int64)
    else:
        data, ir = np.zeros(nnz), np.zeros(nnz, dtype=np.int64)
    if data.dtype.names is not None:
        data = data["real"] + 1j * data["imag"]
    if cls == "logical":
        data = data.astype(bool)
    return sparse.csc_array((data, ir, jc), shape=(rows, len(jc) - 1))


def _empty(cls: str, shape: tuple[int, ...], path: str) -> MatlabValue:
    """Returns an empty MATLAB value of class `cls`."""
    if cls == "char":
        return ""
    if cls in ("cell", "struct"):
        return []
    if cls not in _NUMERIC_CLASSES:
        raise _object_error(path, cls)
    return np.zeros(shape, dtype=_NUMERIC_CLASSES[cls], order="F")


# Versions 4 to 7, read with scipy.io.loadmat.


def _v5_object_class(raw: _Raw) -> str | None:
    """Returns the class of a MATLAB object as scipy reads it, or None for a value."""
    from scipy.io.matlab import MatlabFunction, MatlabObject, MatlabOpaque

    if isinstance(raw, MatlabOpaque):
        return str(np.asarray(raw["_Class"]).ravel()[0])
    if isinstance(raw, MatlabFunction):
        return "function_handle"
    if isinstance(raw, MatlabObject):
        return str(raw.classname)
    return None


def _flat(raw: np.ndarray) -> list[_Raw]:
    """Returns the elements of a scipy array in MATLAB's linear order."""
    return list(raw.reshape(-1, order="F"))


def _restore_complex(typed: _Raw, stored: np.ndarray) -> _Raw:
    """Returns `typed` with the complex values that a read in MATLAB's classes lost.

    `typed` is the value read in MATLAB's classes, which drops imaginary parts,
    and `stored` the same value read in its stored types, which keeps them.
    """
    if not isinstance(typed, np.ndarray) or _v5_object_class(typed) is not None:
        return typed
    if stored.dtype.kind == "c":
        single = typed.dtype == np.float32
        return stored.astype(np.complex64 if single else np.complex128)
    if typed.dtype.names is not None:
        for k in typed.dtype.names:
            field = typed[k]
            for idx in np.ndindex(typed.shape):
                field[idx] = _restore_complex(field[idx], stored[k][idx])
    elif typed.dtype == object:
        for idx in np.ndindex(typed.shape):
            typed[idx] = _restore_complex(typed[idx], stored[idx])
    return typed


def _v5_step(node: _Raw, kind: str, key: str, path: str, file: Path) -> _Raw:
    """Returns the scipy value one path step (`kind`, `key`) below `node`."""
    is_array = isinstance(node, np.ndarray) and _v5_object_class(node) is None
    if not isinstance(node, np.ndarray):
        raise KeyError(f"{file}: {path}: a sparse matrix has no elements to index.")
    if kind == "field":
        names = node.dtype.names if is_array else None
        if names is None:
            raise KeyError(f"{file}: {path}: .{key} of a value that is not a struct.")
        if node.size != 1:
            raise KeyError(f"{file}: {path}: .{key} of a struct array; index it first.")
        if key not in names:
            raise KeyError(f"{file}: {path}: no field {key!r}.")
        value: np.ndarray = node.reshape(-1)[0][key]
        return value
    if kind == "cell":
        if not is_array or node.dtype != object or node.dtype.names is not None:
            raise KeyError(f"{file}: {path}: {{{key}}} of a value that is not a cell.")
        items = _flat(node)
        return items[_index(key, len(items), path)]
    if not is_array or node.dtype.names is None:
        raise KeyError(f"{file}: {path}: ({key}) of a value that is not a struct.")
    i = _index(key, node.size, path)
    return node.reshape(-1, order="F")[i : i + 1].reshape(1, 1)


def _v5_describe(raw: _Raw, path: str) -> MatlabEntry:
    """Returns the entry of a scipy value."""
    cls = _v5_object_class(raw)
    if cls is not None:
        return MatlabEntry(path, (), cls)
    if not isinstance(raw, np.ndarray):
        logical = raw.dtype == np.bool_ or raw.dtype == np.uint8
        shape = tuple(int(n) for n in raw.shape)
        return MatlabEntry(path, shape, "logical" if logical else "double", True)
    shape = tuple(int(n) for n in raw.shape)
    if raw.dtype.names is not None:
        return MatlabEntry(path, shape, "struct")
    if raw.dtype == object:
        return MatlabEntry(path, shape, "cell")
    if raw.dtype.kind == "U":
        return MatlabEntry(path, shape, "char")
    return MatlabEntry(path, shape, _CLASS_OF_DTYPE.get(raw.dtype, str(raw.dtype)))


def _v5_children(raw: _Raw, path: str) -> list[tuple[_Raw, str]]:
    """Returns the struct fields, cells or struct array elements below a scipy value.

    A cell or struct array with more than MAX_LISTED_ELEMENTS elements gives none.
    """
    if not isinstance(raw, np.ndarray) or _v5_object_class(raw) is not None:
        return []
    if raw.dtype.names is not None:
        if raw.size == 1:
            record = raw.reshape(-1)[0]
            return [(record[k], f"{path}.{k}") for k in raw.dtype.names]
        if raw.size > MAX_LISTED_ELEMENTS:
            return []
        flat = raw.reshape(-1, order="F")
        return [
            (flat[i : i + 1].reshape(1, 1), f"{path}({i + 1})") for i in range(raw.size)
        ]
    if raw.dtype == object and raw.size <= MAX_LISTED_ELEMENTS:
        return [(v, f"{path}{{{i + 1}}}") for i, v in enumerate(_flat(raw))]
    return []


def _v5_entries(raw: _Raw, path: str, recursive: bool) -> list[MatlabEntry]:
    """Returns the entry of a scipy value, then those below it if `recursive`."""
    found = [_v5_describe(raw, path)]
    if recursive:
        for child, child_path in _v5_children(raw, path):
            found += _v5_entries(child, child_path, recursive)
    return found


def _v5_value(raw: _Raw, path: str) -> MatlabValue:
    """Converts a value as scipy reads it to a Python value."""
    from scipy import sparse

    cls = _v5_object_class(raw)
    if cls is not None:
        raise _object_error(path, cls)
    if not isinstance(raw, np.ndarray):
        matrix = sparse.csc_array(raw)
        # MATLAB's sparse matrices are double or logical; scipy reads logical as uint8.
        return matrix.astype(bool) if matrix.dtype == np.uint8 else matrix
    if raw.dtype.names is not None:
        if raw.size == 0:
            return []
        records: list[MatlabValue] = [
            {k: _v5_value(r[k], f"{path}.{k}") for k in raw.dtype.names}
            for r in raw.reshape(-1, order="F")
        ]
        return records[0] if raw.shape == (1, 1) else list(records)
    if raw.dtype == object:
        return [_v5_value(v, f"{path}{{{i + 1}}}") for i, v in enumerate(_flat(raw))]
    if raw.dtype.kind == "U":
        if raw.size == 0:
            return ""
        if raw.ndim != 2:
            raise ValueError(
                f"{path}: char arrays of more than two dimensions are not read."
            )
        rows = ["".join(row) for row in raw]
        return rows[0] if len(rows) == 1 else rows
    return np.asarray(raw)
