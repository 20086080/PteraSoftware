"""This module contains classes to test functions in the serialization module."""

import unittest

import numpy as np
import numpy.testing as npt

# noinspection PyProtectedMember
from pterasoftware._serialization import (
    _deserialize_value,
    _ndarray_from_dict,
    _ndarray_to_dict,
    _serialize_value,
)

# noinspection PyProtectedMember
from pterasoftware.movements._functions import (
    oscillating_linspaces,
    oscillating_sinspaces,
)


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
