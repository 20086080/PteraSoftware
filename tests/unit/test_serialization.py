"""This module contains classes to test functions in the serialization module."""

import unittest

import numpy as np
import numpy.testing as npt

# noinspection PyProtectedMember
from pterasoftware._serialization import _ndarray_from_dict, _ndarray_to_dict


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
