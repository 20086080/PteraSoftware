"""Contains the functions for serializing and deserializing Ptera Software objects."""

from __future__ import annotations

import base64

import numpy as np

from . import _logging

_logger = _logging.get_logger("_serialization")


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

    :param value: The value to serialize.
    :return: The JSON serializable representation of the value.
    """
    if value is None:
        return None

    raise TypeError(f"_serialize_value does not yet handle {type(value).__name__}.")


def _deserialize_value(data: object) -> object:
    """Deserializes a single value from its JSON representation.

    :param data: The serialized data.
    :return: The deserialized value.
    """
    if data is None:
        return None

    raise TypeError(f"_deserialize_value does not yet handle {type(data).__name__}.")
