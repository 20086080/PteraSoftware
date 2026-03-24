"""This module contains classes to test functions in the serialization module."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import numpy.testing as npt

# noinspection PyProtectedMember
from pterasoftware._panel import Panel

# noinspection PyProtectedMember
from pterasoftware._serialization import (
    _FORMAT_VERSION,
    _deserialize_value,
    _ndarray_from_dict,
    _ndarray_to_dict,
    _object_from_dict,
    _object_to_dict,
    _serialize_value,
    load,
    save,
)

# noinspection PyProtectedMember
from pterasoftware._vortices._line_vortex import LineVortex

# noinspection PyProtectedMember
from pterasoftware._vortices.horseshoe_vortex import HorseshoeVortex

# noinspection PyProtectedMember
from pterasoftware._vortices.ring_vortex import RingVortex
from pterasoftware.geometry.airfoil import Airfoil
from pterasoftware.geometry.airplane import Airplane
from pterasoftware.geometry.wing import Wing
from pterasoftware.geometry.wing_cross_section import WingCrossSection

# noinspection PyProtectedMember
from pterasoftware.movements._functions import (
    oscillating_linspaces,
    oscillating_sinspaces,
)
from pterasoftware.operating_point import OperatingPoint
from pterasoftware.problems import SteadyProblem


class TestNdarrayRoundTrip(unittest.TestCase):
    """This class contains methods for testing _ndarray_to_dict and
    _ndarray_from_dict round trips.
    """

    def test_float64_1d(self):
        """Tests round trip for a 1D float64 array.

        :return: None
        """
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        result = _ndarray_from_dict(_ndarray_to_dict(arr))
        npt.assert_array_equal(result, arr)
        self.assertEqual(result.dtype, np.float64)

    def test_float64_2d(self):
        """Tests round trip for a 2D float64 array.

        :return: None
        """
        arr = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
        result = _ndarray_from_dict(_ndarray_to_dict(arr))
        npt.assert_array_equal(result, arr)
        self.assertEqual(result.shape, (3, 2))

    def test_float64_3d(self):
        """Tests round trip for a 3D float64 array.

        :return: None
        """
        arr = np.arange(24.0, dtype=np.float64).reshape((2, 4, 3))
        result = _ndarray_from_dict(_ndarray_to_dict(arr))
        npt.assert_array_equal(result, arr)
        self.assertEqual(result.shape, (2, 4, 3))

    def test_int64_1d(self):
        """Tests round trip for a 1D int64 array.

        :return: None
        """
        arr = np.array([10, 20, 30], dtype=np.int64)
        result = _ndarray_from_dict(_ndarray_to_dict(arr))
        npt.assert_array_equal(result, arr)
        self.assertEqual(result.dtype, np.int64)

    def test_bool_1d(self):
        """Tests round trip for a 1D bool array.

        :return: None
        """
        arr = np.array([True, False, True, False], dtype=np.bool_)
        result = _ndarray_from_dict(_ndarray_to_dict(arr))
        npt.assert_array_equal(result, arr)
        self.assertEqual(result.dtype, np.bool_)

    def test_empty_float64(self):
        """Tests round trip for an empty float64 array.

        :return: None
        """
        arr = np.array([], dtype=np.float64)
        result = _ndarray_from_dict(_ndarray_to_dict(arr))
        npt.assert_array_equal(result, arr)
        self.assertEqual(result.shape, (0,))
        self.assertEqual(result.dtype, np.float64)

    def test_empty_2d(self):
        """Tests round trip for an empty 2D array with a non trivial second dimension.

        :return: None
        """
        arr = np.empty((0, 3), dtype=np.float64)
        result = _ndarray_from_dict(_ndarray_to_dict(arr))
        self.assertEqual(result.shape, (0, 3))
        self.assertEqual(result.dtype, np.float64)

    def test_writeable_preserved(self):
        """Tests that a writable array remains writable after round trip.

        :return: None
        """
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        self.assertTrue(arr.flags.writeable)
        result = _ndarray_from_dict(_ndarray_to_dict(arr))
        self.assertTrue(result.flags.writeable)

    def test_read_only_preserved(self):
        """Tests that a read only array remains read only after round trip.

        :return: None
        """
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        arr.flags.writeable = False
        result = _ndarray_from_dict(_ndarray_to_dict(arr))
        npt.assert_array_equal(result, arr)
        self.assertFalse(result.flags.writeable)

    def test_missing_writeable_defaults_to_writeable(self):
        """Tests that a missing writeable field defaults to a writable array.

        :return: None
        """
        arr = np.array([1.0, 2.0], dtype=np.float64)
        d = _ndarray_to_dict(arr)
        del d["writeable"]
        result = _ndarray_from_dict(d)
        self.assertTrue(result.flags.writeable)

    def test_dtype_object_with_none(self):
        """Tests round trip for a dtype=object array containing None values.

        :return: None
        """
        arr = np.empty(3, dtype=object)
        arr[0] = None
        arr[1] = None
        arr[2] = None
        result = _ndarray_from_dict(_ndarray_to_dict(arr))
        self.assertEqual(result.dtype, object)
        self.assertEqual(result.shape, (3,))
        for i in range(3):
            self.assertIsNone(result[i])

    def test_dtype_object_2d_shape(self):
        """Tests that a 2D dtype=object array preserves its shape after round trip.

        :return: None
        """
        arr = np.empty((2, 3), dtype=object)
        for i in range(2):
            for j in range(3):
                arr[i, j] = None
        result = _ndarray_from_dict(_ndarray_to_dict(arr))
        self.assertEqual(result.shape, (2, 3))
        self.assertEqual(result.dtype, object)

    def test_dtype_object_read_only_preserved(self):
        """Tests that a read only dtype=object array remains read only after round trip.

        :return: None
        """
        arr = np.empty(2, dtype=object)
        arr[0] = None
        arr[1] = None
        arr.flags.writeable = False
        result = _ndarray_from_dict(_ndarray_to_dict(arr))
        self.assertFalse(result.flags.writeable)

    def test_dtype_object_writeable_preserved(self):
        """Tests that a writable dtype=object array remains writable after round trip.

        :return: None
        """
        arr = np.empty(2, dtype=object)
        arr[0] = None
        arr[1] = None
        result = _ndarray_from_dict(_ndarray_to_dict(arr))
        self.assertTrue(result.flags.writeable)


class TestNdarrayToDict(unittest.TestCase):
    """This class contains methods for testing _ndarray_to_dict output structure."""

    def test_numeric_dict_keys(self):
        """Tests that a numeric array produces a dict with the expected keys.

        :return: None
        """
        arr = np.array([1.0], dtype=np.float64)
        d = _ndarray_to_dict(arr)
        self.assertEqual(d["_type"], "ndarray")
        self.assertEqual(d["dtype"], "float64")
        self.assertEqual(d["shape"], [1])
        self.assertIn("data", d)
        self.assertIn("writeable", d)
        self.assertNotIn("items", d)

    def test_object_dict_keys(self):
        """Tests that a dtype=object array produces a dict with the expected keys.

        :return: None
        """
        arr = np.empty(1, dtype=object)
        arr[0] = None
        d = _ndarray_to_dict(arr)
        self.assertEqual(d["_type"], "ndarray")
        self.assertEqual(d["dtype"], "object")
        self.assertEqual(d["shape"], [1])
        self.assertIn("items", d)
        self.assertIn("writeable", d)
        self.assertNotIn("data", d)

    def test_base64_data_is_string(self):
        """Tests that the base64 encoded data is a string.

        :return: None
        """
        arr = np.array([1.0, 2.0], dtype=np.float64)
        d = _ndarray_to_dict(arr)
        self.assertIsInstance(d["data"], str)


class TestSerializeValue(unittest.TestCase):
    """This class contains methods for testing _serialize_value."""

    def test_none(self):
        """Tests that None serializes to None.

        :return: None
        """
        self.assertIsNone(_serialize_value(None))

    def test_bool_true(self):
        """Tests that True serializes to True.

        :return: None
        """
        result = _serialize_value(True)
        self.assertIs(result, True)

    def test_bool_false(self):
        """Tests that False serializes to False.

        :return: None
        """
        result = _serialize_value(False)
        self.assertIs(result, False)

    def test_np_bool(self):
        """Tests that a numpy bool serializes to a Python bool.

        :return: None
        """
        result = _serialize_value(np.bool_(True))
        self.assertIs(result, True)
        self.assertIsInstance(result, bool)

    def test_bool_not_wrapped_as_int(self):
        """Tests that bool values are not wrapped as int dicts.

        :return: None
        """
        result = _serialize_value(True)
        self.assertNotIsInstance(result, dict)

    def test_int(self):
        """Tests that a Python int serializes to a wrapped dict.

        :return: None
        """
        result = _serialize_value(42)
        self.assertEqual(result, {"_type": "int", "value": 42})

    def test_np_int64(self):
        """Tests that a numpy int64 serializes to a wrapped dict with a Python int.

        :return: None
        """
        result = _serialize_value(np.int64(7))
        self.assertEqual(result, {"_type": "int", "value": 7})
        assert isinstance(result, dict)
        self.assertIsInstance(result["value"], int)

    def test_float(self):
        """Tests that a Python float serializes to a wrapped dict.

        :return: None
        """
        result = _serialize_value(3.14)
        self.assertEqual(result, {"_type": "float", "value": 3.14})

    def test_np_float64(self):
        """Tests that a numpy float64 serializes to a wrapped dict with a Python float.

        :return: None
        """
        result = _serialize_value(np.float64(2.718))
        self.assertEqual(result, {"_type": "float", "value": 2.718})
        assert isinstance(result, dict)
        self.assertIsInstance(result["value"], float)

    def test_float_inf_raises(self):
        """Tests that serializing inf raises a ValueError.

        :return: None
        """
        with self.assertRaises(ValueError):
            _serialize_value(float("inf"))

    def test_float_negative_inf_raises(self):
        """Tests that serializing negative inf raises a ValueError.

        :return: None
        """
        with self.assertRaises(ValueError):
            _serialize_value(float("-inf"))

    def test_float_nan_raises(self):
        """Tests that serializing NaN raises a ValueError.

        :return: None
        """
        with self.assertRaises(ValueError):
            _serialize_value(float("nan"))

    def test_str(self):
        """Tests that a string serializes to itself.

        :return: None
        """
        result = _serialize_value("hello")
        self.assertEqual(result, "hello")

    def test_ndarray(self):
        """Tests that a numpy array delegates to _ndarray_to_dict.

        :return: None
        """
        arr = np.array([1.0, 2.0], dtype=np.float64)
        result = _serialize_value(arr)
        assert isinstance(result, dict)
        self.assertEqual(result["_type"], "ndarray")

    def test_tuple(self):
        """Tests that a tuple serializes to a dict with items.

        :return: None
        """
        result = _serialize_value((1, 2.0, "three"))
        assert isinstance(result, dict)
        self.assertEqual(result["_type"], "tuple")
        self.assertEqual(len(result["items"]), 3)

    def test_tuple_empty(self):
        """Tests that an empty tuple serializes correctly.

        :return: None
        """
        result = _serialize_value(())
        self.assertEqual(result, {"_type": "tuple", "items": []})

    def test_list(self):
        """Tests that a list serializes to a dict with items.

        :return: None
        """
        result = _serialize_value([1, 2.0, "three"])
        assert isinstance(result, dict)
        self.assertEqual(result["_type"], "list")
        self.assertEqual(len(result["items"]), 3)

    def test_list_empty(self):
        """Tests that an empty list serializes correctly.

        :return: None
        """
        result = _serialize_value([])
        self.assertEqual(result, {"_type": "list", "items": []})

    def test_nested_tuple(self):
        """Tests that a nested tuple serializes recursively.

        :return: None
        """
        result = _serialize_value((1, (2, 3)))
        assert isinstance(result, dict)
        self.assertEqual(result["_type"], "tuple")
        inner = result["items"][1]
        assert isinstance(inner, dict)
        self.assertEqual(inner["_type"], "tuple")
        self.assertEqual(len(inner["items"]), 2)

    def test_callable_sine(self):
        """Tests that the oscillating_sinspaces function serializes by name.

        :return: None
        """
        result = _serialize_value(oscillating_sinspaces)
        self.assertEqual(result, {"_type": "callable", "name": "sine"})

    def test_callable_uniform(self):
        """Tests that the oscillating_linspaces function serializes by name.

        :return: None
        """
        result = _serialize_value(oscillating_linspaces)
        self.assertEqual(result, {"_type": "callable", "name": "uniform"})

    def test_callable_custom_raises(self):
        """Tests that a custom callable raises a ValueError.

        :return: None
        """
        with self.assertRaises(ValueError):
            _serialize_value(lambda x: x)

    def test_unsupported_type_raises(self):
        """Tests that an unsupported type raises a TypeError.

        :return: None
        """
        with self.assertRaises(TypeError):
            _serialize_value(set())


class TestDeserializeValue(unittest.TestCase):
    """This class contains methods for testing _deserialize_value."""

    def test_none(self):
        """Tests that None deserializes to None.

        :return: None
        """
        self.assertIsNone(_deserialize_value(None))

    def test_bool_true(self):
        """Tests that True deserializes to True.

        :return: None
        """
        self.assertIs(_deserialize_value(True), True)

    def test_bool_false(self):
        """Tests that False deserializes to False.

        :return: None
        """
        self.assertIs(_deserialize_value(False), False)

    def test_str(self):
        """Tests that a string deserializes to itself.

        :return: None
        """
        self.assertEqual(_deserialize_value("hello"), "hello")

    def test_int(self):
        """Tests that a wrapped int dict deserializes to an int.

        :return: None
        """
        result = _deserialize_value({"_type": "int", "value": 42})
        self.assertEqual(result, 42)
        self.assertIsInstance(result, int)

    def test_float(self):
        """Tests that a wrapped float dict deserializes to a float.

        :return: None
        """
        result = _deserialize_value({"_type": "float", "value": 3.14})
        self.assertEqual(result, 3.14)
        self.assertIsInstance(result, float)

    def test_ndarray(self):
        """Tests that an ndarray dict deserializes to a numpy array.

        :return: None
        """
        arr = np.array([1.0, 2.0], dtype=np.float64)
        d = _ndarray_to_dict(arr)
        result = _deserialize_value(d)
        npt.assert_array_equal(result, arr)

    def test_tuple(self):
        """Tests that a tuple dict deserializes to a tuple.

        :return: None
        """
        data = {
            "_type": "tuple",
            "items": [
                {"_type": "int", "value": 1},
                {"_type": "float", "value": 2.0},
                "three",
            ],
        }
        result = _deserialize_value(data)
        self.assertEqual(result, (1, 2.0, "three"))
        self.assertIsInstance(result, tuple)

    def test_list(self):
        """Tests that a list dict deserializes to a list.

        :return: None
        """
        data = {
            "_type": "list",
            "items": [
                {"_type": "int", "value": 1},
                {"_type": "float", "value": 2.0},
            ],
        }
        result = _deserialize_value(data)
        self.assertEqual(result, [1, 2.0])
        self.assertIsInstance(result, list)

    def test_callable_sine(self):
        """Tests that a callable dict with name "sine" deserializes to
        oscillating_sinspaces.

        :return: None
        """
        result = _deserialize_value({"_type": "callable", "name": "sine"})
        self.assertIs(result, oscillating_sinspaces)

    def test_callable_uniform(self):
        """Tests that a callable dict with name "uniform" deserializes to
        oscillating_linspaces.

        :return: None
        """
        result = _deserialize_value({"_type": "callable", "name": "uniform"})
        self.assertIs(result, oscillating_linspaces)

    def test_callable_unknown_name_raises(self):
        """Tests that an unknown callable name raises a ValueError.

        :return: None
        """
        with self.assertRaises(ValueError):
            _deserialize_value({"_type": "callable", "name": "unknown"})

    def test_bare_int_raises(self):
        """Tests that a bare JSON int raises a ValueError.

        :return: None
        """
        with self.assertRaises(ValueError):
            _deserialize_value(42)

    def test_bare_float_raises(self):
        """Tests that a bare JSON float raises a ValueError.

        :return: None
        """
        with self.assertRaises(ValueError):
            _deserialize_value(3.14)

    def test_dict_without_type_raises(self):
        """Tests that a dict without a _type key raises a ValueError.

        :return: None
        """
        with self.assertRaises(ValueError):
            _deserialize_value({"key": "value"})

    def test_unknown_type_tag_raises(self):
        """Tests that an unknown _type tag raises a TypeError.

        :return: None
        """
        with self.assertRaises(TypeError):
            _deserialize_value({"_type": "unknown"})


class TestValueRoundTrip(unittest.TestCase):
    """This class contains methods for testing _serialize_value and _deserialize_value
    round trips.
    """

    def test_none(self):
        """Tests round trip for None.

        :return: None
        """
        self.assertIsNone(_deserialize_value(_serialize_value(None)))

    def test_bool(self):
        """Tests round trip for bool values.

        :return: None
        """
        for value in [True, False]:
            with self.subTest(value=value):
                self.assertIs(_deserialize_value(_serialize_value(value)), value)

    def test_int(self):
        """Tests round trip for int values.

        :return: None
        """
        for value in [0, 1, -1, 42, -999]:
            with self.subTest(value=value):
                result = _deserialize_value(_serialize_value(value))
                self.assertEqual(result, value)
                self.assertIsInstance(result, int)

    def test_np_int64(self):
        """Tests round trip for numpy int64 values.

        :return: None
        """
        result = _deserialize_value(_serialize_value(np.int64(7)))
        self.assertEqual(result, 7)
        self.assertIsInstance(result, int)

    def test_float(self):
        """Tests round trip for float values.

        :return: None
        """
        for value in [0.0, 1.5, -3.14, 1e-10, 1e10]:
            with self.subTest(value=value):
                result = _deserialize_value(_serialize_value(value))
                self.assertEqual(result, value)
                self.assertIsInstance(result, float)

    def test_np_float64(self):
        """Tests round trip for numpy float64 values.

        :return: None
        """
        result = _deserialize_value(_serialize_value(np.float64(2.718)))
        self.assertEqual(result, 2.718)
        self.assertIsInstance(result, float)

    def test_str(self):
        """Tests round trip for string values.

        :return: None
        """
        for value in ["", "hello", "cosine", "uniform"]:
            with self.subTest(value=value):
                self.assertEqual(_deserialize_value(_serialize_value(value)), value)

    def test_tuple(self):
        """Tests round trip for a tuple with mixed types.

        :return: None
        """
        value = (1, 2.0, "three", None, True)
        result = _deserialize_value(_serialize_value(value))
        self.assertEqual(result, value)
        self.assertIsInstance(result, tuple)

    def test_list(self):
        """Tests round trip for a list with mixed types.

        :return: None
        """
        value = [1, 2.0, "three", None, False]
        result = _deserialize_value(_serialize_value(value))
        self.assertEqual(result, value)
        self.assertIsInstance(result, list)

    def test_nested_containers(self):
        """Tests round trip for nested tuples and lists.

        :return: None
        """
        value = ([1, 2], (3.0, "four"), [None, True])
        result = _deserialize_value(_serialize_value(value))
        assert isinstance(result, tuple)
        self.assertEqual(result[0], [1, 2])
        self.assertIsInstance(result[0], list)
        self.assertEqual(result[1], (3.0, "four"))
        self.assertIsInstance(result[1], tuple)

    def test_callable_sine(self):
        """Tests round trip for the oscillating_sinspaces function.

        :return: None
        """
        self.assertIs(
            _deserialize_value(_serialize_value(oscillating_sinspaces)),
            oscillating_sinspaces,
        )

    def test_callable_uniform(self):
        """Tests round trip for the oscillating_linspaces function.

        :return: None
        """
        self.assertIs(
            _deserialize_value(_serialize_value(oscillating_linspaces)),
            oscillating_linspaces,
        )


class TestObjectToDict(unittest.TestCase):
    """This class contains methods for testing _object_to_dict."""

    def test_line_vortex_type_tag(self):
        """Tests that a LineVortex serializes with the correct _type tag.

        :return: None
        """
        lv = LineVortex(
            Slvp_GP1_CgP1=np.array([0.0, 0.0, 0.0]),
            Elvp_GP1_CgP1=np.array([1.0, 0.0, 0.0]),
            strength=1.0,
        )
        d = _object_to_dict(lv)
        self.assertEqual(d["_type"], "LineVortex")

    def test_line_vortex_slot_keys(self):
        """Tests that all LineVortex __slots__ appear as keys in the serialized dict.

        :return: None
        """
        lv = LineVortex(
            Slvp_GP1_CgP1=np.array([0.0, 0.0, 0.0]),
            Elvp_GP1_CgP1=np.array([1.0, 0.0, 0.0]),
            strength=1.0,
        )
        d = _object_to_dict(lv)
        for slot_name in LineVortex.__slots__:
            self.assertIn(slot_name, d)

    def test_line_vortex_strength(self):
        """Tests that a LineVortex's strength is serialized correctly.

        :return: None
        """
        lv = LineVortex(
            Slvp_GP1_CgP1=np.array([0.0, 0.0, 0.0]),
            Elvp_GP1_CgP1=np.array([1.0, 0.0, 0.0]),
            strength=5.0,
        )
        d = _object_to_dict(lv)
        self.assertEqual(d["strength"], {"_type": "float", "value": 5.0})

    def test_line_vortex_arrays(self):
        """Tests that a LineVortex's endpoint arrays are serialized as ndarray dicts.

        :return: None
        """
        start = np.array([1.0, 2.0, 3.0])
        end = np.array([4.0, 5.0, 6.0])
        lv = LineVortex(
            Slvp_GP1_CgP1=start,
            Elvp_GP1_CgP1=end,
            strength=1.0,
        )
        d = _object_to_dict(lv)
        assert isinstance(d["_Slvp_GP1_CgP1"], dict)
        self.assertEqual(d["_Slvp_GP1_CgP1"]["_type"], "ndarray")
        assert isinstance(d["_Elvp_GP1_CgP1"], dict)
        self.assertEqual(d["_Elvp_GP1_CgP1"]["_type"], "ndarray")

    def test_line_vortex_none_caches(self):
        """Tests that uncomputed cache slots serialize as None.

        :return: None
        """
        lv = LineVortex(
            Slvp_GP1_CgP1=np.array([0.0, 0.0, 0.0]),
            Elvp_GP1_CgP1=np.array([1.0, 0.0, 0.0]),
            strength=1.0,
        )
        d = _object_to_dict(lv)
        self.assertIsNone(d["_vector_GP1"])
        self.assertIsNone(d["_Clvp_GP1_CgP1"])

    def test_line_vortex_populated_caches(self):
        """Tests that computed cache slots serialize as ndarray dicts.

        :return: None
        """
        lv = LineVortex(
            Slvp_GP1_CgP1=np.array([0.0, 0.0, 0.0]),
            Elvp_GP1_CgP1=np.array([1.0, 0.0, 0.0]),
            strength=1.0,
        )
        # Trigger cache population by accessing the properties.
        _ = lv.vector_GP1
        _ = lv.Clvp_GP1_CgP1
        d = _object_to_dict(lv)
        assert isinstance(d["_vector_GP1"], dict)
        self.assertEqual(d["_vector_GP1"]["_type"], "ndarray")
        assert isinstance(d["_Clvp_GP1_CgP1"], dict)
        self.assertEqual(d["_Clvp_GP1_CgP1"]["_type"], "ndarray")

    def test_line_vortex_via_serialize_value(self):
        """Tests that _serialize_value dispatches LineVortex to _object_to_dict.

        :return: None
        """
        lv = LineVortex(
            Slvp_GP1_CgP1=np.array([0.0, 0.0, 0.0]),
            Elvp_GP1_CgP1=np.array([1.0, 0.0, 0.0]),
            strength=1.0,
        )
        result = _serialize_value(lv)
        assert isinstance(result, dict)
        self.assertEqual(result["_type"], "LineVortex")

    def test_skip_slots(self):
        """Tests that slots in _skip_slots are serialized as None.

        :return: None
        """
        lv = LineVortex(
            Slvp_GP1_CgP1=np.array([0.0, 0.0, 0.0]),
            Elvp_GP1_CgP1=np.array([1.0, 0.0, 0.0]),
            strength=1.0,
        )
        d = _object_to_dict(lv, _skip_slots=frozenset({"strength"}))
        self.assertIsNone(d["strength"])
        assert isinstance(d["_Slvp_GP1_CgP1"], dict)
        self.assertEqual(d["_Slvp_GP1_CgP1"]["_type"], "ndarray")

    def test_unregistered_class_raises(self):
        """Tests that an unregistered class raises a TypeError.

        :return: None
        """
        with self.assertRaises(TypeError):
            _object_to_dict("not a Ptera Software object")


class TestObjectFromDict(unittest.TestCase):
    """This class contains methods for testing _object_from_dict."""

    def test_line_vortex_round_trip(self):
        """Tests that a LineVortex survives a full round trip.

        :return: None
        """
        start = np.array([1.0, 2.0, 3.0])
        end = np.array([4.0, 5.0, 6.0])
        lv = LineVortex(Slvp_GP1_CgP1=start, Elvp_GP1_CgP1=end, strength=5.0)
        d = _object_to_dict(lv)
        result = _object_from_dict(d)
        assert isinstance(result, LineVortex)
        npt.assert_array_equal(result.Slvp_GP1_CgP1, start)
        npt.assert_array_equal(result.Elvp_GP1_CgP1, end)
        self.assertEqual(result.strength, 5.0)

    def test_line_vortex_array_writeable_flags(self):
        """Tests that deserialized LineVortex arrays preserve their writeable flags.

        :return: None
        """
        lv = LineVortex(
            Slvp_GP1_CgP1=np.array([0.0, 0.0, 0.0]),
            Elvp_GP1_CgP1=np.array([1.0, 0.0, 0.0]),
            strength=1.0,
        )
        result = _object_from_dict(_object_to_dict(lv))
        assert isinstance(result, LineVortex)
        self.assertFalse(result.Slvp_GP1_CgP1.flags.writeable)
        self.assertFalse(result.Elvp_GP1_CgP1.flags.writeable)

    def test_line_vortex_none_caches_round_trip(self):
        """Tests that uncomputed caches remain None after round trip.

        :return: None
        """
        lv = LineVortex(
            Slvp_GP1_CgP1=np.array([0.0, 0.0, 0.0]),
            Elvp_GP1_CgP1=np.array([1.0, 0.0, 0.0]),
            strength=1.0,
        )
        result = _object_from_dict(_object_to_dict(lv))
        assert isinstance(result, LineVortex)
        # Access private cache slots directly to verify they are None.
        self.assertIsNone(object.__getattribute__(result, "_vector_GP1"))
        self.assertIsNone(object.__getattribute__(result, "_Clvp_GP1_CgP1"))

    def test_line_vortex_populated_caches_round_trip(self):
        """Tests that populated caches survive round trip with correct values.

        :return: None
        """
        lv = LineVortex(
            Slvp_GP1_CgP1=np.array([0.0, 0.0, 0.0]),
            Elvp_GP1_CgP1=np.array([1.0, 0.0, 0.0]),
            strength=1.0,
        )
        # Trigger cache population.
        expected_vector = lv.vector_GP1.copy()
        expected_center = lv.Clvp_GP1_CgP1.copy()
        result = _object_from_dict(_object_to_dict(lv))
        assert isinstance(result, LineVortex)
        npt.assert_array_equal(result.vector_GP1, expected_vector)
        npt.assert_array_equal(result.Clvp_GP1_CgP1, expected_center)

    def test_line_vortex_via_deserialize_value(self):
        """Tests that _deserialize_value dispatches a LineVortex dict to
        _object_from_dict.

        :return: None
        """
        lv = LineVortex(
            Slvp_GP1_CgP1=np.array([0.0, 0.0, 0.0]),
            Elvp_GP1_CgP1=np.array([1.0, 0.0, 0.0]),
            strength=1.0,
        )
        d = _object_to_dict(lv)
        result = _deserialize_value(d)
        assert isinstance(result, LineVortex)
        self.assertEqual(result.strength, 1.0)

    def test_line_vortex_full_round_trip_via_value_functions(self):
        """Tests a full round trip through _serialize_value and _deserialize_value.

        :return: None
        """
        start = np.array([1.0, 2.0, 3.0])
        end = np.array([4.0, 5.0, 6.0])
        lv = LineVortex(Slvp_GP1_CgP1=start, Elvp_GP1_CgP1=end, strength=7.5)
        result = _deserialize_value(_serialize_value(lv))
        assert isinstance(result, LineVortex)
        npt.assert_array_equal(result.Slvp_GP1_CgP1, start)
        npt.assert_array_equal(result.Elvp_GP1_CgP1, end)
        self.assertEqual(result.strength, 7.5)

    def test_unknown_class_raises(self):
        """Tests that an unknown class name raises a TypeError.

        :return: None
        """
        with self.assertRaises(TypeError):
            _object_from_dict({"_type": "UnknownClass"})


class TestSaveLoad(unittest.TestCase):
    """This class contains methods for testing save and load."""

    def test_json_round_trip(self):
        """Tests that save and load produce a correct round trip via JSON.

        :return: None
        """
        start = np.array([1.0, 2.0, 3.0])
        end = np.array([4.0, 5.0, 6.0])
        lv = LineVortex(Slvp_GP1_CgP1=start, Elvp_GP1_CgP1=end, strength=5.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.json"
            save(path, lv)
            result = load(path)
        assert isinstance(result, LineVortex)
        npt.assert_array_equal(result.Slvp_GP1_CgP1, start)
        npt.assert_array_equal(result.Elvp_GP1_CgP1, end)
        self.assertEqual(result.strength, 5.0)

    def test_gzip_round_trip(self):
        """Tests that save and load produce a correct round trip via gzip JSON.

        :return: None
        """
        start = np.array([1.0, 2.0, 3.0])
        end = np.array([4.0, 5.0, 6.0])
        lv = LineVortex(Slvp_GP1_CgP1=start, Elvp_GP1_CgP1=end, strength=5.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.json.gz"
            save(path, lv)
            result = load(path)
        assert isinstance(result, LineVortex)
        npt.assert_array_equal(result.Slvp_GP1_CgP1, start)
        npt.assert_array_equal(result.Elvp_GP1_CgP1, end)
        self.assertEqual(result.strength, 5.0)

    def test_file_contains_format_version(self):
        """Tests that the saved file contains the format version.

        :return: None
        """
        import json

        lv = LineVortex(
            Slvp_GP1_CgP1=np.array([0.0, 0.0, 0.0]),
            Elvp_GP1_CgP1=np.array([1.0, 0.0, 0.0]),
            strength=1.0,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.json"
            save(path, lv)
            with open(path) as f:
                data = json.load(f)
        self.assertEqual(data["_format_version"], _FORMAT_VERSION)

    def test_file_contains_provenance(self):
        """Tests that the saved file contains provenance metadata.

        :return: None
        """
        import json

        lv = LineVortex(
            Slvp_GP1_CgP1=np.array([0.0, 0.0, 0.0]),
            Elvp_GP1_CgP1=np.array([1.0, 0.0, 0.0]),
            strength=1.0,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.json"
            save(path, lv)
            with open(path) as f:
                data = json.load(f)
        self.assertIn("_saved_at", data)
        self.assertIn("_pterasoftware_version", data)
        self.assertIn("_commit", data)
        self.assertIn("_dirty", data)

    def test_format_version_mismatch_raises(self):
        """Tests that loading a file with a mismatched format version raises.

        :return: None
        """
        import json

        lv = LineVortex(
            Slvp_GP1_CgP1=np.array([0.0, 0.0, 0.0]),
            Elvp_GP1_CgP1=np.array([1.0, 0.0, 0.0]),
            strength=1.0,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.json"
            save(path, lv)
            with open(path) as f:
                data = json.load(f)
            data["_format_version"] = 9999
            with open(path, "w") as f:
                json.dump(data, f)
            with self.assertRaises(ValueError):
                load(path)

    def test_save_accepts_string_path(self):
        """Tests that save accepts a string path in addition to a Path object.

        :return: None
        """
        lv = LineVortex(
            Slvp_GP1_CgP1=np.array([0.0, 0.0, 0.0]),
            Elvp_GP1_CgP1=np.array([1.0, 0.0, 0.0]),
            strength=1.0,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "test.json")
            save(path, lv)
            result = load(path)
        assert isinstance(result, LineVortex)

    def test_save_invalid_extension_raises(self):
        """Tests that save raises a ValueError for an unsupported file extension.

        :return: None
        """
        lv = LineVortex(
            Slvp_GP1_CgP1=np.array([0.0, 0.0, 0.0]),
            Elvp_GP1_CgP1=np.array([1.0, 0.0, 0.0]),
            strength=1.0,
        )
        with self.assertRaises(ValueError):
            save("test.txt", lv)

    def test_load_invalid_extension_raises(self):
        """Tests that load raises a ValueError for an unsupported file extension.

        :return: None
        """
        with self.assertRaises(ValueError):
            load("test.dat")

    def test_gzip_bomb_protection(self):
        """Tests that loading a gzip file that exceeds the size limit raises.

        :return: None
        """
        lv = LineVortex(
            Slvp_GP1_CgP1=np.array([0.0, 0.0, 0.0]),
            Elvp_GP1_CgP1=np.array([1.0, 0.0, 0.0]),
            strength=1.0,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.json.gz"
            save(path, lv)
            with patch("pterasoftware._serialization._MAX_DECOMPRESSED_SIZE", 10):
                with self.assertRaises(ValueError):
                    load(path)


class TestRingVortexRoundTrip(unittest.TestCase):
    """This class contains methods for testing RingVortex serialization round trips."""

    def test_round_trip(self):
        """Tests that a RingVortex survives a full round trip.

        :return: None
        """
        fr = np.array([1.0, 0.0, 0.0])
        fl = np.array([1.0, 1.0, 0.0])
        bl = np.array([0.0, 1.0, 0.0])
        br = np.array([0.0, 0.0, 0.0])
        rv = RingVortex(
            Frrvp_GP1_CgP1=fr,
            Flrvp_GP1_CgP1=fl,
            Blrvp_GP1_CgP1=bl,
            Brrvp_GP1_CgP1=br,
            strength=3.0,
        )
        rv.age = 0.5
        result = _deserialize_value(_serialize_value(rv))
        assert isinstance(result, RingVortex)
        npt.assert_array_equal(result.Frrvp_GP1_CgP1, fr)
        npt.assert_array_equal(result.Flrvp_GP1_CgP1, fl)
        npt.assert_array_equal(result.Blrvp_GP1_CgP1, bl)
        npt.assert_array_equal(result.Brrvp_GP1_CgP1, br)
        self.assertEqual(result.strength, 3.0)
        self.assertEqual(result.age, 0.5)

    def test_populated_leg_caches_round_trip(self):
        """Tests that populated LineVortex leg caches survive round trip.

        :return: None
        """
        rv = RingVortex(
            Frrvp_GP1_CgP1=np.array([1.0, 0.0, 0.0]),
            Flrvp_GP1_CgP1=np.array([1.0, 1.0, 0.0]),
            Blrvp_GP1_CgP1=np.array([0.0, 1.0, 0.0]),
            Brrvp_GP1_CgP1=np.array([0.0, 0.0, 0.0]),
            strength=1.0,
        )
        # Trigger cache population.
        _ = rv.front_leg
        _ = rv.left_leg
        _ = rv.back_leg
        _ = rv.right_leg
        result = _deserialize_value(_serialize_value(rv))
        assert isinstance(result, RingVortex)
        assert isinstance(result.front_leg, LineVortex)
        npt.assert_array_equal(
            result.front_leg.Slvp_GP1_CgP1, rv.front_leg.Slvp_GP1_CgP1
        )

    def test_none_caches_round_trip(self):
        """Tests that uncomputed caches remain None after round trip.

        :return: None
        """
        rv = RingVortex(
            Frrvp_GP1_CgP1=np.array([1.0, 0.0, 0.0]),
            Flrvp_GP1_CgP1=np.array([1.0, 1.0, 0.0]),
            Blrvp_GP1_CgP1=np.array([0.0, 1.0, 0.0]),
            Brrvp_GP1_CgP1=np.array([0.0, 0.0, 0.0]),
            strength=1.0,
        )
        result = _deserialize_value(_serialize_value(rv))
        assert isinstance(result, RingVortex)
        self.assertIsNone(object.__getattribute__(result, "_front_leg"))
        self.assertIsNone(object.__getattribute__(result, "_Crvp_GP1_CgP1"))
        self.assertIsNone(object.__getattribute__(result, "_area"))


class TestHorseshoeVortexRoundTrip(unittest.TestCase):
    """This class contains methods for testing HorseshoeVortex serialization round
    trips.
    """

    def test_round_trip(self):
        """Tests that a HorseshoeVortex survives a full round trip.

        :return: None
        """
        fr = np.array([1.0, 0.0, 0.0])
        fl = np.array([1.0, 1.0, 0.0])
        vec = np.array([-1.0, 0.0, 0.0])
        hv = HorseshoeVortex(
            Frhvp_GP1_CgP1=fr,
            Flhvp_GP1_CgP1=fl,
            leftLegVector_GP1=vec,
            left_right_leg_lengths=20.0,
            strength=4.0,
        )
        result = _deserialize_value(_serialize_value(hv))
        assert isinstance(result, HorseshoeVortex)
        npt.assert_array_equal(result.Frhvp_GP1_CgP1, fr)
        npt.assert_array_equal(result.Flhvp_GP1_CgP1, fl)
        npt.assert_allclose(result.leftLegVector_GP1, vec / np.linalg.norm(vec))
        self.assertEqual(result.left_right_leg_lengths, 20.0)
        self.assertEqual(result.strength, 4.0)

    def test_populated_leg_caches_round_trip(self):
        """Tests that populated LineVortex leg caches survive round trip.

        :return: None
        """
        hv = HorseshoeVortex(
            Frhvp_GP1_CgP1=np.array([1.0, 0.0, 0.0]),
            Flhvp_GP1_CgP1=np.array([1.0, 1.0, 0.0]),
            leftLegVector_GP1=np.array([-1.0, 0.0, 0.0]),
            left_right_leg_lengths=20.0,
            strength=1.0,
        )
        # Trigger cache population.
        _ = hv.right_leg
        _ = hv.finite_leg
        _ = hv.left_leg
        result = _deserialize_value(_serialize_value(hv))
        assert isinstance(result, HorseshoeVortex)
        assert isinstance(result.finite_leg, LineVortex)
        npt.assert_array_equal(
            result.finite_leg.Slvp_GP1_CgP1, hv.finite_leg.Slvp_GP1_CgP1
        )

    def test_none_caches_round_trip(self):
        """Tests that uncomputed caches remain None after round trip.

        :return: None
        """
        hv = HorseshoeVortex(
            Frhvp_GP1_CgP1=np.array([1.0, 0.0, 0.0]),
            Flhvp_GP1_CgP1=np.array([1.0, 1.0, 0.0]),
            leftLegVector_GP1=np.array([-1.0, 0.0, 0.0]),
            left_right_leg_lengths=20.0,
            strength=1.0,
        )
        result = _deserialize_value(_serialize_value(hv))
        assert isinstance(result, HorseshoeVortex)
        self.assertIsNone(object.__getattribute__(result, "_right_leg"))
        self.assertIsNone(object.__getattribute__(result, "_finite_leg"))
        self.assertIsNone(object.__getattribute__(result, "_left_leg"))
        self.assertIsNone(object.__getattribute__(result, "_Brhvp_GP1_CgP1"))
        self.assertIsNone(object.__getattribute__(result, "_Blhvp_GP1_CgP1"))


class TestAirfoilRoundTrip(unittest.TestCase):
    """This class contains methods for testing Airfoil serialization round trips."""

    def test_round_trip(self):
        """Tests that an Airfoil survives a full round trip.

        :return: None
        """
        airfoil = Airfoil(name="NACA0012")
        result = _deserialize_value(_serialize_value(airfoil))
        assert isinstance(result, Airfoil)
        self.assertEqual(result.name, "NACA0012")
        npt.assert_array_equal(result.outline_A_lp, airfoil.outline_A_lp)
        self.assertEqual(result.resample, airfoil.resample)
        self.assertEqual(result.n_points_per_side, airfoil.n_points_per_side)

    def test_mcl_round_trip(self):
        """Tests that the mean camber line array survives round trip.

        :return: None
        """
        airfoil = Airfoil(name="NACA2412")
        result = _deserialize_value(_serialize_value(airfoil))
        assert isinstance(result, Airfoil)
        assert result.mcl_A_lp is not None
        assert airfoil.mcl_A_lp is not None
        npt.assert_array_equal(result.mcl_A_lp, airfoil.mcl_A_lp)

    def test_writeable_flags(self):
        """Tests that deserialized Airfoil arrays preserve their writeable flags.

        :return: None
        """
        airfoil = Airfoil(name="NACA0012")
        result = _deserialize_value(_serialize_value(airfoil))
        assert isinstance(result, Airfoil)
        self.assertFalse(result.outline_A_lp.flags.writeable)

    def test_save_load_round_trip(self):
        """Tests that an Airfoil survives a save/load round trip.

        :return: None
        """
        airfoil = Airfoil(name="NACA0012")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "airfoil.json"
            save(path, airfoil)
            result = load(path)
        assert isinstance(result, Airfoil)
        self.assertEqual(result.name, "NACA0012")
        npt.assert_array_equal(result.outline_A_lp, airfoil.outline_A_lp)


class TestOperatingPointRoundTrip(unittest.TestCase):
    """This class contains methods for testing OperatingPoint serialization round
    trips.
    """

    def test_round_trip(self):
        """Tests that an OperatingPoint survives a full round trip.

        :return: None
        """
        op = OperatingPoint(rho=1.225, vCg__E=10.0, alpha=5.0, beta=0.0)
        result = _deserialize_value(_serialize_value(op))
        assert isinstance(result, OperatingPoint)
        self.assertEqual(result.rho, 1.225)
        self.assertEqual(result.vCg__E, 10.0)
        self.assertEqual(result.alpha, 5.0)
        self.assertEqual(result.beta, 0.0)

    def test_with_surface_effect(self):
        """Tests that an OperatingPoint with surface effect parameters survives round
        trip.

        :return: None
        """
        op = OperatingPoint(
            surfaceNormal_E=(0.0, 0.0, 1.0),
            surfacePoint_E_Eo=(0.0, 0.0, -1.0),
        )
        result = _deserialize_value(_serialize_value(op))
        assert isinstance(result, OperatingPoint)
        assert result.surfaceNormal_E is not None
        assert op.surfaceNormal_E is not None
        npt.assert_array_equal(result.surfaceNormal_E, op.surfaceNormal_E)
        assert result.surfacePoint_E_Eo is not None
        assert op.surfacePoint_E_Eo is not None
        npt.assert_array_equal(result.surfacePoint_E_Eo, op.surfacePoint_E_Eo)

    def test_none_surface_params(self):
        """Tests that None surface parameters remain None after round trip.

        :return: None
        """
        op = OperatingPoint()
        result = _deserialize_value(_serialize_value(op))
        assert isinstance(result, OperatingPoint)
        self.assertIsNone(result.surfaceNormal_E)
        self.assertIsNone(result.surfacePoint_E_Eo)

    def test_none_caches_round_trip(self):
        """Tests that uncomputed caches remain None after round trip.

        :return: None
        """
        op = OperatingPoint()
        result = _deserialize_value(_serialize_value(op))
        assert isinstance(result, OperatingPoint)
        self.assertIsNone(object.__getattribute__(result, "_qInf__E"))
        self.assertIsNone(object.__getattribute__(result, "_T_pas_GP1_CgP1_to_W_CgP1"))
        self.assertIsNone(object.__getattribute__(result, "_vInf_GP1__E"))


class TestWingCrossSectionRoundTrip(unittest.TestCase):
    """This class contains methods for testing WingCrossSection serialization round
    trips.
    """

    def test_round_trip(self):
        """Tests that a WingCrossSection survives a full round trip.

        :return: None
        """
        wing_cross_section = WingCrossSection(
            airfoil=Airfoil(name="NACA0012"),
            num_spanwise_panels=8,
            chord=1.0,
        )
        result = _deserialize_value(_serialize_value(wing_cross_section))
        assert isinstance(result, WingCrossSection)
        self.assertEqual(result.airfoil.name, "NACA0012")
        self.assertEqual(result.num_spanwise_panels, 8)
        self.assertEqual(result.chord, 1.0)

    def test_tip_wing_cross_section(self):
        """Tests that a tip WingCrossSection (num_spanwise_panels=None) survives round
        trip.

        :return: None
        """
        wing_cross_section = WingCrossSection(
            airfoil=Airfoil(name="NACA0012"),
            num_spanwise_panels=None,
        )
        result = _deserialize_value(_serialize_value(wing_cross_section))
        assert isinstance(result, WingCrossSection)
        self.assertIsNone(result.num_spanwise_panels)

    def test_nested_airfoil(self):
        """Tests that the nested Airfoil object survives round trip.

        :return: None
        """
        airfoil = Airfoil(name="NACA2412")
        wing_cross_section = WingCrossSection(
            airfoil=airfoil,
            num_spanwise_panels=8,
        )
        result = _deserialize_value(_serialize_value(wing_cross_section))
        assert isinstance(result, WingCrossSection)
        npt.assert_array_equal(result.airfoil.outline_A_lp, airfoil.outline_A_lp)


class TestPanelRoundTrip(unittest.TestCase):
    """This class contains methods for testing Panel serialization round trips."""

    def test_round_trip(self):
        """Tests that a Panel survives a full round trip.

        :return: None
        """
        panel = Panel(
            Frpp_G_Cg=np.array([1.0, 0.0, 0.0]),
            Flpp_G_Cg=np.array([1.0, 1.0, 0.0]),
            Blpp_G_Cg=np.array([0.0, 1.0, 0.0]),
            Brpp_G_Cg=np.array([0.0, 0.0, 0.0]),
            is_leading_edge=True,
            is_trailing_edge=False,
        )
        result = _deserialize_value(_serialize_value(panel))
        assert isinstance(result, Panel)
        npt.assert_array_equal(result.Frpp_G_Cg, np.array([1.0, 0.0, 0.0]))
        npt.assert_array_equal(result.Flpp_G_Cg, np.array([1.0, 1.0, 0.0]))
        self.assertTrue(result.is_leading_edge)
        self.assertFalse(result.is_trailing_edge)

    def test_none_mutable_attrs(self):
        """Tests that None mutable attributes remain None after round trip.

        :return: None
        """
        panel = Panel(
            Frpp_G_Cg=np.array([1.0, 0.0, 0.0]),
            Flpp_G_Cg=np.array([1.0, 1.0, 0.0]),
            Blpp_G_Cg=np.array([0.0, 1.0, 0.0]),
            Brpp_G_Cg=np.array([0.0, 0.0, 0.0]),
            is_leading_edge=True,
            is_trailing_edge=False,
        )
        result = _deserialize_value(_serialize_value(panel))
        assert isinstance(result, Panel)
        self.assertIsNone(result.ring_vortex)
        self.assertIsNone(result.horseshoe_vortex)
        self.assertIsNone(result.forces_GP1)

    def test_writeable_flags(self):
        """Tests that deserialized Panel arrays preserve their writeable flags.

        :return: None
        """
        panel = Panel(
            Frpp_G_Cg=np.array([1.0, 0.0, 0.0]),
            Flpp_G_Cg=np.array([1.0, 1.0, 0.0]),
            Blpp_G_Cg=np.array([0.0, 1.0, 0.0]),
            Brpp_G_Cg=np.array([0.0, 0.0, 0.0]),
            is_leading_edge=True,
            is_trailing_edge=False,
        )
        result = _deserialize_value(_serialize_value(panel))
        assert isinstance(result, Panel)
        self.assertFalse(result.Frpp_G_Cg.flags.writeable)


def _make_meshed_airplane() -> Airplane:
    """Creates a simple meshed Airplane for testing.

    :return: A meshed Airplane with one Wing containing 2x2 Panels.
    """
    import pterasoftware as ps

    wing = Wing(
        wing_cross_sections=[
            WingCrossSection(
                airfoil=Airfoil(name="NACA0012"),
                num_spanwise_panels=2,
                chord=1.0,
            ),
            WingCrossSection(
                airfoil=Airfoil(name="NACA0012"),
                num_spanwise_panels=None,
                chord=1.0,
                Lp_Wcsp_Lpp=(0.0, 5.0, 0.0),
            ),
        ],
        num_chordwise_panels=2,
        chordwise_spacing="uniform",
    )
    airplane = Airplane(wings=[wing])
    op = ps.operating_point.OperatingPoint()
    ps.problems.SteadyProblem(airplanes=[airplane], operating_point=op)
    return airplane


def _make_steady_problem() -> SteadyProblem:
    """Creates a simple SteadyProblem for testing.

    :return: A SteadyProblem with one meshed Airplane and 2x2 Panels.
    """
    import pterasoftware as ps

    wing = Wing(
        wing_cross_sections=[
            WingCrossSection(
                airfoil=Airfoil(name="NACA0012"),
                num_spanwise_panels=2,
                chord=1.0,
            ),
            WingCrossSection(
                airfoil=Airfoil(name="NACA0012"),
                num_spanwise_panels=None,
                chord=1.0,
                Lp_Wcsp_Lpp=(0.0, 5.0, 0.0),
            ),
        ],
        num_chordwise_panels=2,
        chordwise_spacing="uniform",
    )
    airplane = Airplane(wings=[wing])
    op = ps.operating_point.OperatingPoint(rho=1.225, vCg__E=10.0, alpha=5.0)
    return ps.problems.SteadyProblem(airplanes=[airplane], operating_point=op)


class TestWingRoundTrip(unittest.TestCase):
    """This class contains methods for testing Wing serialization round trips."""

    def test_round_trip(self):
        """Tests that a meshed Wing survives a full round trip.

        :return: None
        """
        airplane = _make_meshed_airplane()
        wing = airplane.wings[0]
        result = _deserialize_value(_serialize_value(wing))
        assert isinstance(result, Wing)
        self.assertEqual(result.name, wing.name)
        self.assertEqual(result.num_chordwise_panels, wing.num_chordwise_panels)
        self.assertEqual(result.chordwise_spacing, wing.chordwise_spacing)

    def test_panels_dtype_object_round_trip(self):
        """Tests that the dtype=object _panels array survives round trip with Panel
        objects.

        :return: None
        """
        airplane = _make_meshed_airplane()
        wing = airplane.wings[0]
        assert wing.panels is not None
        result = _deserialize_value(_serialize_value(wing))
        assert isinstance(result, Wing)
        assert result.panels is not None
        self.assertEqual(result.panels.shape, wing.panels.shape)
        self.assertEqual(result.panels.dtype, object)
        for i in range(result.panels.shape[0]):
            for j in range(result.panels.shape[1]):
                assert isinstance(result.panels[i, j], Panel)
                npt.assert_array_equal(
                    result.panels[i, j].Frpp_G_Cg,
                    wing.panels[i, j].Frpp_G_Cg,
                )

    def test_wing_cross_sections_tuple_round_trip(self):
        """Tests that the _wing_cross_sections tuple survives round trip.

        :return: None
        """
        airplane = _make_meshed_airplane()
        wing = airplane.wings[0]
        result = _deserialize_value(_serialize_value(wing))
        assert isinstance(result, Wing)
        self.assertEqual(len(result.wing_cross_sections), len(wing.wing_cross_sections))
        for orig, loaded in zip(wing.wing_cross_sections, result.wing_cross_sections):
            assert isinstance(loaded, WingCrossSection)
            self.assertEqual(loaded.chord, orig.chord)


class TestAirplaneRoundTrip(unittest.TestCase):
    """This class contains methods for testing Airplane serialization round trips."""

    def test_round_trip(self):
        """Tests that a meshed Airplane survives a full round trip.

        :return: None
        """
        airplane = _make_meshed_airplane()
        result = _deserialize_value(_serialize_value(airplane))
        assert isinstance(result, Airplane)
        self.assertEqual(result.name, airplane.name)
        self.assertEqual(len(result.wings), len(airplane.wings))
        self.assertEqual(result.s_ref, airplane.s_ref)
        self.assertEqual(result.c_ref, airplane.c_ref)
        self.assertEqual(result.b_ref, airplane.b_ref)

    def test_nested_wing_panels(self):
        """Tests that an Airplane's nested Wing Panel arrays survive round trip.

        :return: None
        """
        airplane = _make_meshed_airplane()
        result = _deserialize_value(_serialize_value(airplane))
        assert isinstance(result, Airplane)
        for orig_wing, loaded_wing in zip(airplane.wings, result.wings):
            assert isinstance(loaded_wing, Wing)
            assert loaded_wing.panels is not None
            assert orig_wing.panels is not None
            self.assertEqual(loaded_wing.panels.shape, orig_wing.panels.shape)

    def test_save_load_round_trip(self):
        """Tests that a meshed Airplane survives a save/load round trip.

        :return: None
        """
        airplane = _make_meshed_airplane()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "airplane.json"
            save(path, airplane)
            result = load(path)
        assert isinstance(result, Airplane)
        self.assertEqual(result.name, airplane.name)
        self.assertEqual(len(result.wings), len(airplane.wings))


class TestSteadyProblemRoundTrip(unittest.TestCase):
    """This class contains methods for testing SteadyProblem serialization round
    trips.
    """

    def test_round_trip(self):
        """Tests that a SteadyProblem survives a full round trip.

        :return: None
        """
        problem = _make_steady_problem()
        result = _deserialize_value(_serialize_value(problem))
        assert isinstance(result, SteadyProblem)
        self.assertEqual(len(result.airplanes), 1)
        assert isinstance(result.airplanes[0], Airplane)
        assert isinstance(result.operating_point, OperatingPoint)
        self.assertEqual(result.operating_point.rho, 1.225)
        self.assertEqual(result.operating_point.alpha, 5.0)

    def test_panels_have_gp1_coordinates(self):
        """Tests that deserialized Panels retain their GP1 coordinates set by the
        SteadyProblem.

        :return: None
        """
        problem = _make_steady_problem()
        orig_panel = problem.airplanes[0].wings[0].panels[0, 0]
        result = _deserialize_value(_serialize_value(problem))
        assert isinstance(result, SteadyProblem)
        loaded_panel = result.airplanes[0].wings[0].panels[0, 0]
        assert orig_panel.Frpp_GP1_CgP1 is not None
        assert loaded_panel.Frpp_GP1_CgP1 is not None
        npt.assert_array_equal(loaded_panel.Frpp_GP1_CgP1, orig_panel.Frpp_GP1_CgP1)

    def test_save_load_round_trip(self):
        """Tests that a SteadyProblem survives a save/load round trip.

        :return: None
        """
        problem = _make_steady_problem()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "problem.json"
            save(path, problem)
            result = load(path)
        assert isinstance(result, SteadyProblem)
        self.assertEqual(len(result.airplanes), 1)
        assert isinstance(result.operating_point, OperatingPoint)
