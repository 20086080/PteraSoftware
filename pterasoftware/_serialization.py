"""Contains the functions for serializing and deserializing Ptera Software objects."""

from __future__ import annotations

import base64
import gzip
import json
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from . import _logging

# noinspection PyProtectedMember
from ._panel import Panel

# This module is inherently coupled to the internals of every class in the package
# (it reads __slots__, knows class structure, and imports all classes into its
# registry), so importing from a sibling private module is acceptable here.
# noinspection PyProtectedMember
from ._vortices._line_vortex import LineVortex

# noinspection PyProtectedMember
from ._vortices.horseshoe_vortex import HorseshoeVortex
from ._vortices.ring_vortex import RingVortex
from .geometry.airfoil import Airfoil
from .geometry.airplane import Airplane
from .geometry.wing import Wing
from .geometry.wing_cross_section import WingCrossSection

# noinspection PyProtectedMember
from .movements._functions import oscillating_linspaces, oscillating_sinspaces
from .operating_point import OperatingPoint
from .problems import SteadyProblem

_logger = _logging.get_logger("_serialization")

# Maps serializable callable names to their function objects and vice versa.
_CALLABLE_NAME_TO_FUNC = {
    "sine": oscillating_sinspaces,
    "uniform": oscillating_linspaces,
}
_CALLABLE_FUNC_TO_NAME = {v: k for k, v in _CALLABLE_NAME_TO_FUNC.items()}

# Increments only when the serialization structure changes (slots added/removed/
# renamed, class registry changed, encoding strategy changed).
_FORMAT_VERSION = 1

# Maximum decompressed size in bytes when reading gzip files. Prevents gzip bombs
# from exhausting memory.
_MAX_DECOMPRESSED_SIZE = 1_073_741_824  # 1 GB

# Maps class names to their types for deserialization dispatch.
_CLASS_REGISTRY: dict[str, type] = {
    "LineVortex": LineVortex,
    "RingVortex": RingVortex,
    "HorseshoeVortex": HorseshoeVortex,
    "Airfoil": Airfoil,
    "OperatingPoint": OperatingPoint,
    "WingCrossSection": WingCrossSection,
    "Panel": Panel,
    "Wing": Wing,
    "Airplane": Airplane,
    "SteadyProblem": SteadyProblem,
}


def save(path: str | Path, obj: object) -> None:
    """Saves a Ptera Software object to a JSON file.

    If the path ends with ".json.gz", the output is gzip compressed automatically.

    :param path: The file path to save to. Should end with ".json" or ".json.gz".
    :param obj: The Ptera Software object to save.
    :return: None
    """
    path = Path(path)
    if not path.name.endswith(".json") and not path.name.endswith(".json.gz"):
        raise ValueError(
            f"Path must end with '.json' or '.json.gz', got '{path.name}'."
        )
    _logger.info("Saving %s to %s.", type(obj).__name__, path)

    data = _object_to_dict(obj)

    # Add the metadata header fields to the serialized dict.
    provenance = _get_provenance()
    header = {"_format_version": _FORMAT_VERSION, **provenance}
    data = {**header, **data}

    json_bytes = json.dumps(data).encode("utf-8")

    if path.name.endswith(".json.gz"):
        with gzip.open(path, "wb") as f:
            f.write(json_bytes)
    else:
        with open(path, "wb") as f:
            f.write(json_bytes)

    file_size = path.stat().st_size
    _logger.info("Saved %s to %s (%d bytes).", type(obj).__name__, path, file_size)


def load(path: str | Path) -> object:
    """Loads a Ptera Software object from a JSON file.

    If the path ends with ".json.gz", the input is gzip decompressed automatically.

    :param path: The file path to load from.
    :return: The deserialized Ptera Software object.
    """
    path = Path(path)
    if not path.name.endswith(".json") and not path.name.endswith(".json.gz"):
        raise ValueError(
            f"Path must end with '.json' or '.json.gz', got '{path.name}'."
        )
    _logger.info("Loading from %s.", path)

    if path.name.endswith(".json.gz"):
        with gzip.open(path, "rb") as f:
            raw = f.read(_MAX_DECOMPRESSED_SIZE + 1)
            if len(raw) > _MAX_DECOMPRESSED_SIZE:
                raise ValueError(
                    f"Decompressed file exceeds the maximum allowed size of "
                    f"{_MAX_DECOMPRESSED_SIZE} bytes."
                )
    else:
        with open(path, "rb") as f:
            raw = f.read()

    data = json.loads(raw)

    # Check format version compatibility.
    file_version = data.get("_format_version")
    if file_version != _FORMAT_VERSION:
        raise ValueError(
            f"Format version mismatch: file has version {file_version}, but the "
            f"current code expects version {_FORMAT_VERSION}."
        )

    # Log provenance warnings.
    _log_load_warnings(data)

    obj = _object_from_dict(data)
    _logger.info("Loaded %s from %s.", type(obj).__name__, path)
    return obj


def _get_provenance() -> dict:
    """Returns a dict of provenance metadata for the serialized file.

    The provenance fields are informational only and are never checked at load time. The
    git derived fields (_commit and _dirty) are best effort and are set to None if the
    code is running outside a git repository.

    :return: A dict with provenance metadata.
    """
    try:
        pkg_version = version("PteraSoftware")
    except PackageNotFoundError:
        pkg_version = None

    commit = None
    dirty = None
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode("ascii")
            .strip()
        )
        status = (
            subprocess.check_output(
                ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
            )
            .decode("ascii")
            .strip()
        )
        dirty = len(status) > 0
    except (FileNotFoundError, subprocess.CalledProcessError):
        _logger.warning("Git is not available. Provenance fields will be null.")

    return {
        "_pterasoftware_version": pkg_version,
        "_commit": commit,
        "_dirty": dirty,
        "_saved_at": datetime.now(timezone.utc).isoformat(),
    }


def _log_load_warnings(data: dict) -> None:
    """Logs warnings about provenance metadata during deserialization.

    :param data: The top level dict loaded from the JSON file.
    :return: None
    """
    if data.get("_dirty"):
        _logger.warning(
            "The file was saved with uncommitted changes (_dirty=true). The _commit "
            "hash may not fully represent the code state at save time."
        )

    file_commit = data.get("_commit")
    if file_commit is not None:
        try:
            current_commit = (
                subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
                )
                .decode("ascii")
                .strip()
            )
            if file_commit != current_commit:
                _logger.warning(
                    "The file was saved at commit %s, but the current HEAD is %s.",
                    file_commit[:12],
                    current_commit[:12],
                )
            current_status = (
                subprocess.check_output(
                    ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
                )
                .decode("ascii")
                .strip()
            )
            if len(current_status) > 0:
                _logger.warning("The current working tree has uncommitted changes.")
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass


def _object_to_dict(obj: object, *, _skip_slots: frozenset[str] = frozenset()) -> dict:
    """Generically serializes a Ptera Software object to a dict.

    Iterates over the class's __slots__ to discover all attributes, including cache
    slots. JSON keys are the slot names themselves (e.g., "_rho", "_name", "forces_W").
    Slots listed in _skip_slots are serialized as null (used by the shared reference
    optimization for UnsteadyProblem and solver classes).

    :param obj: The Ptera Software object to serialize.
    :param _skip_slots: A frozenset of slot names to serialize as null.
    :return: A dict with "_type" set to the class name and one key per slot.
    """
    cls = type(obj)
    class_name = cls.__name__
    if class_name not in _CLASS_REGISTRY:
        raise TypeError(f"_object_to_dict does not handle {class_name}.")

    result: dict = {"_type": class_name}
    for slot_name in getattr(cls, "__slots__"):
        if slot_name in _skip_slots:
            result[slot_name] = None
        else:
            result[slot_name] = _serialize_value(getattr(obj, slot_name))
    return result


def _object_from_dict(data: dict) -> object:
    """Generically deserializes a Ptera Software object from a dict.

    Uses the "_type" tag to look up the class in the registry. Creates an empty instance
    via object.__new__(cls), bypassing __init__ entirely. Then restores every serialized
    attribute (including caches) via object.__setattr__(). After restoring all slots,
    calls _reconstruct_shared_references() to rebuild any slots that were skipped during
    serialization.

    :param data: The dict produced by _object_to_dict.
    :return: The reconstructed Ptera Software object.
    """
    type_tag = data["_type"]
    cls = _CLASS_REGISTRY.get(type_tag)
    if cls is None:
        raise TypeError(f"Unknown class in _object_from_dict: '{type_tag}'.")

    obj: object = object.__new__(cls)
    for slot_name in getattr(cls, "__slots__"):
        object.__setattr__(obj, slot_name, _deserialize_value(data[slot_name]))
    _reconstruct_shared_references(obj)
    return obj


def _reconstruct_shared_references(obj: object) -> None:
    """Rebuilds shared references that were skipped during serialization.

    Called after _object_from_dict restores all serialized slots. Handles
    UnsteadyProblem (Movement <-> SteadyProblem aliases), steady solvers (solver ->
    SteadyProblem aliases), and the unsteady solver (solver -> UnsteadyProblem aliases).

    :param obj: The deserialized Ptera Software object.
    :return: None
    """
    # No shared references to reconstruct for simple classes like LineVortex,
    # RingVortex, HorseshoeVortex, etc. Extended in later steps for
    # UnsteadyProblem and solver classes.


def _ndarray_to_dict(arr: np.ndarray) -> dict:
    """Converts a NumPy ndarray to a JSON serializable dict.

    For numeric and bool dtypes, the array data is encoded as a base64 string with dtype
    and shape metadata. For dtype=object arrays (e.g., Wing._panels,
    Wing.wake_ring_vortices), each element is serialized individually via
    _serialize_value. The writeable flag is recorded so that deserialization can restore
    the original mutability.

    :param arr: The ndarray to serialize.
    :return: A dict representing the serialized ndarray.
    """
    if arr.dtype == object:
        return {
            "_type": "ndarray",
            "dtype": "object",
            "shape": list(arr.shape),
            "items": [_serialize_value(item) for item in arr.ravel()],
            "writeable": bool(arr.flags.writeable),
        }

    return {
        "_type": "ndarray",
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
        "data": base64.b64encode(arr.tobytes()).decode("ascii"),
        "writeable": bool(arr.flags.writeable),
    }


def _ndarray_from_dict(d: dict) -> np.ndarray:
    """Reconstructs a NumPy ndarray from a dict produced by _ndarray_to_dict.

    Dispatches on dtype: base64 decode for numeric and bool dtypes, element by element
    deserialization for dtype=object (reshaping to the original shape). After
    reconstruction, restores the writeable flag from the dict's "writeable" field. If
    the field is absent, the array defaults to writeable.

    :param d: The dict produced by _ndarray_to_dict.
    :return: The reconstructed ndarray.
    """
    shape = d["shape"]
    writeable = d.get("writeable", True)

    if d["dtype"] == "object":
        items = [_deserialize_value(item) for item in d["items"]]
        arr = np.empty(len(items), dtype=object)
        for i, item in enumerate(items):
            arr[i] = item
        arr = arr.reshape(shape)
    else:
        raw = base64.b64decode(d["data"])
        arr = np.frombuffer(raw, dtype=np.dtype(d["dtype"])).reshape(shape).copy()

    if not writeable:
        arr.flags.writeable = False

    return arr


def _serialize_value(value: object) -> object:
    """Serializes a single value based on its runtime type.

    Dispatch order: None, bool (before int since bool is a subclass of int), int, float,
    str, ndarray, tuple, list, callable. int and float are wrapped in {"_type": ...,
    "value": ...} dicts rather than serialized as bare JSON numbers to eliminate the
    int/float ambiguity that arises because JSON has a single number type. None, bool,
    and str remain bare JSON values because they map to unambiguous JSON types (null,
    boolean, string).

    :param value: The value to serialize.
    :return: The JSON serializable representation of the value.
    """
    if value is None:
        return None

    # Check bool before int because bool is a subclass of int.
    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    if isinstance(value, (int, np.integer)):
        return {"_type": "int", "value": int(value)}

    if isinstance(value, (float, np.floating)):
        float_value = float(value)
        if not np.isfinite(float_value):
            raise ValueError(f"Cannot serialize non finite float: {float_value}.")
        return {"_type": "float", "value": float_value}

    if isinstance(value, str):
        return value

    if isinstance(value, np.ndarray):
        return _ndarray_to_dict(value)

    if isinstance(value, tuple):
        return {
            "_type": "tuple",
            "items": [_serialize_value(item) for item in value],
        }

    if isinstance(value, list):
        return {
            "_type": "list",
            "items": [_serialize_value(item) for item in value],
        }

    if type(value).__name__ in _CLASS_REGISTRY:
        return _object_to_dict(value)

    if callable(value):
        name = _CALLABLE_FUNC_TO_NAME.get(value)
        if name is None:
            raise ValueError(
                "Only 'sine' and 'uniform' spacing functions are serializable. Custom "
                "callables cannot be serialized."
            )
        return {"_type": "callable", "name": name}

    raise TypeError(f"_serialize_value does not handle {type(value).__name__}.")


def _deserialize_value(data: object) -> object:
    """Deserializes a single value from its JSON representation.

    The format is self describing via _type tags. None, bool, and str are bare JSON
    values. All other types are wrapped in dicts with a _type key. Bare JSON numbers
    (int or float without a _type wrapper) raise a ValueError because all numeric values
    are wrapped during serialization.

    :param data: The serialized data.
    :return: The deserialized value.
    """
    if data is None:
        return None

    if isinstance(data, bool):
        return data

    if isinstance(data, str):
        return data

    if isinstance(data, dict):
        type_tag = data.get("_type")
        if type_tag is None:
            raise ValueError(
                "Dict without '_type' key encountered during deserialization."
            )
        if type_tag == "int":
            return int(data["value"])
        if type_tag == "float":
            return float(data["value"])
        if type_tag == "ndarray":
            return _ndarray_from_dict(data)
        if type_tag == "tuple":
            return tuple(_deserialize_value(item) for item in data["items"])
        if type_tag == "list":
            return [_deserialize_value(item) for item in data["items"]]
        if type_tag == "callable":
            name = data["name"]
            func = _CALLABLE_NAME_TO_FUNC.get(name)
            if func is None:
                raise ValueError(f"Unknown callable name: '{name}'.")
            return func
        if type_tag in _CLASS_REGISTRY:
            return _object_from_dict(data)
        raise TypeError(f"Unknown _type tag: '{type_tag}'.")

    if isinstance(data, (int, float)):
        raise ValueError(
            f"Bare JSON number {data} encountered during deserialization. All numeric "
            "values should be wrapped in _type dicts."
        )

    raise TypeError(f"_deserialize_value does not handle {type(data).__name__}.")
