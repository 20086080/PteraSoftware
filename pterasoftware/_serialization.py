"""Contains the functions for serializing and deserializing Ptera Software objects."""

from __future__ import annotations

import base64

import numpy as np

from . import _logging

# This module is inherently coupled to the internals of every class in the package
# (it reads __slots__, knows class structure, and imports all classes into its
# registry), so importing from a sibling private module is acceptable here.
# noinspection PyProtectedMember
from .movements._functions import oscillating_linspaces, oscillating_sinspaces

_logger = _logging.get_logger("_serialization")

# Maps serializable callable names to their function objects and vice versa.
_CALLABLE_NAME_TO_FUNC = {
    "sine": oscillating_sinspaces,
    "uniform": oscillating_linspaces,
}
_CALLABLE_FUNC_TO_NAME = {v: k for k, v in _CALLABLE_NAME_TO_FUNC.items()}


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
        raise TypeError(f"Unknown _type tag: '{type_tag}'.")

    if isinstance(data, (int, float)):
        raise ValueError(
            f"Bare JSON number {data} encountered during deserialization. All numeric "
            "values should be wrapped in _type dicts."
        )

    raise TypeError(f"_deserialize_value does not handle {type(data).__name__}.")
