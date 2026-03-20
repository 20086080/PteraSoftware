"""This module contains classes to test __slots__ enforcement on slotted classes."""

import copy
import unittest

import numpy as np
import numpy.testing as npt

import pterasoftware as ps

# noinspection PyProtectedMember
from pterasoftware._vortices import _line_vortex
from tests.unit.fixtures import (
    geometry_fixtures,
    line_vortex_fixtures,
    operating_point_fixtures,
)


class TestLineVortexSlots(unittest.TestCase):
    """This class contains tests to verify __slots__ enforcement on LineVortex."""

    def setUp(self):
        """Set up test fixtures for LineVortex slots tests."""
        self.line_vortex = line_vortex_fixtures.make_basic_line_vortex_fixture()

    def test_slots_defined(self):
        """Test that __slots__ is defined on LineVortex."""
        self.assertTrue(hasattr(_line_vortex.LineVortex, "__slots__"))

    def test_no_instance_dict(self):
        """Test that LineVortex instances have no __dict__."""
        self.assertFalse(hasattr(self.line_vortex, "__dict__"))

    def test_dynamic_attribute_raises(self):
        """Test that dynamic attribute assignment raises AttributeError."""
        with self.assertRaises(AttributeError):
            self.line_vortex.nonexistent_attribute = 42

    def test_property_access(self):
        """Test that all properties remain accessible after adding __slots__."""
        # Immutable properties.
        npt.assert_array_equal(
            self.line_vortex.Slvp_GP1_CgP1,
            np.array([0.0, 0.0, 0.0]),
        )
        npt.assert_array_equal(
            self.line_vortex.Elvp_GP1_CgP1,
            np.array([1.0, 0.0, 0.0]),
        )

        # Mutable attribute.
        self.assertEqual(self.line_vortex.strength, 1.0)

        # Cached computed properties.
        npt.assert_array_equal(
            self.line_vortex.vector_GP1,
            np.array([1.0, 0.0, 0.0]),
        )
        npt.assert_array_equal(
            self.line_vortex.Clvp_GP1_CgP1,
            np.array([0.5, 0.0, 0.0]),
        )

    def test_deepcopy(self):
        """Test that copy.deepcopy produces a correct independent copy."""
        # Access cached properties before copying to test cache copying.
        _ = self.line_vortex.vector_GP1
        _ = self.line_vortex.Clvp_GP1_CgP1

        copied = copy.deepcopy(self.line_vortex)

        # Verify the copy is a separate instance.
        self.assertIsNot(copied, self.line_vortex)

        # Verify all property values match.
        npt.assert_array_equal(copied.Slvp_GP1_CgP1, self.line_vortex.Slvp_GP1_CgP1)
        npt.assert_array_equal(copied.Elvp_GP1_CgP1, self.line_vortex.Elvp_GP1_CgP1)
        self.assertEqual(copied.strength, self.line_vortex.strength)
        npt.assert_array_equal(copied.vector_GP1, self.line_vortex.vector_GP1)
        npt.assert_array_equal(copied.Clvp_GP1_CgP1, self.line_vortex.Clvp_GP1_CgP1)

        # Verify arrays are independent (not shared references).
        self.assertIsNot(copied.Slvp_GP1_CgP1, self.line_vortex.Slvp_GP1_CgP1)
        self.assertIsNot(copied.Elvp_GP1_CgP1, self.line_vortex.Elvp_GP1_CgP1)


class TestAirfoilSlots(unittest.TestCase):
    """This class contains tests to verify __slots__ enforcement on Airfoil."""

    def setUp(self):
        """Set up test fixtures for Airfoil slots tests."""
        self.airfoil = geometry_fixtures.make_naca0012_airfoil_fixture()

    def test_slots_defined(self):
        """Test that __slots__ is defined on Airfoil."""
        self.assertTrue(hasattr(ps.geometry.airfoil.Airfoil, "__slots__"))

    def test_no_instance_dict(self):
        """Test that Airfoil instances have no __dict__."""
        self.assertFalse(hasattr(self.airfoil, "__dict__"))

    def test_dynamic_attribute_raises(self):
        """Test that dynamic attribute assignment raises AttributeError."""
        with self.assertRaises(AttributeError):
            self.airfoil.nonexistent_attribute = 42

    def test_property_access(self):
        """Test that all properties remain accessible after adding __slots__."""
        self.assertEqual(self.airfoil.name, "naca0012")
        self.assertIsInstance(self.airfoil.outline_A_lp, np.ndarray)
        self.assertIsInstance(self.airfoil.resample, bool)
        self.assertIsInstance(self.airfoil.n_points_per_side, int)
        self.assertIsNotNone(self.airfoil.mcl_A_lp)

    def test_deepcopy_method(self):
        """Test that __deepcopy__ produces a correct independent copy."""
        copied = copy.deepcopy(self.airfoil)

        # Verify the copy is a separate instance.
        self.assertIsNot(copied, self.airfoil)

        # Verify all property values match.
        self.assertEqual(copied.name, self.airfoil.name)
        npt.assert_array_equal(copied.outline_A_lp, self.airfoil.outline_A_lp)
        self.assertEqual(copied.resample, self.airfoil.resample)
        self.assertEqual(copied.n_points_per_side, self.airfoil.n_points_per_side)
        npt.assert_array_equal(copied.mcl_A_lp, self.airfoil.mcl_A_lp)

        # Verify arrays are independent (not shared references).
        self.assertIsNot(copied.outline_A_lp, self.airfoil.outline_A_lp)
        self.assertIsNot(copied.mcl_A_lp, self.airfoil.mcl_A_lp)

    def test_deepcopy_no_dict(self):
        """Test that a deep copied Airfoil has no __dict__."""
        copied = copy.deepcopy(self.airfoil)
        self.assertFalse(hasattr(copied, "__dict__"))


class TestOperatingPointSlots(unittest.TestCase):
    """This class contains tests to verify __slots__ enforcement on OperatingPoint."""

    def setUp(self):
        """Set up test fixtures for OperatingPoint slots tests."""
        self.basic_op = operating_point_fixtures.make_basic_operating_point_fixture()
        self.surface_op = (
            operating_point_fixtures.make_with_ground_surface_operating_point_fixture()
        )

    def test_slots_defined(self):
        """Test that __slots__ is defined on OperatingPoint."""
        self.assertTrue(hasattr(ps.operating_point.OperatingPoint, "__slots__"))

    def test_no_instance_dict(self):
        """Test that OperatingPoint instances have no __dict__."""
        self.assertFalse(hasattr(self.basic_op, "__dict__"))

    def test_dynamic_attribute_raises(self):
        """Test that dynamic attribute assignment raises AttributeError."""
        with self.assertRaises(AttributeError):
            self.basic_op.nonexistent_attribute = 42

    def test_property_access_basic(self):
        """Test that immutable properties are accessible on a basic OperatingPoint."""
        self.assertEqual(self.basic_op.rho, 1.225)
        self.assertEqual(self.basic_op.vCg__E, 10.0)
        self.assertEqual(self.basic_op.alpha, 5.0)
        self.assertEqual(self.basic_op.beta, 0.0)
        npt.assert_array_equal(
            self.basic_op.angles_E_to_BP1_izyx,
            np.array([0.0, 0.0, 0.0]),
        )
        npt.assert_array_equal(
            self.basic_op.CgP1_E_Eo,
            np.array([0.0, 0.0, 0.0]),
        )
        self.assertIsNone(self.basic_op.surfaceNormal_E)
        self.assertIsNone(self.basic_op.surfacePoint_E_Eo)
        self.assertEqual(self.basic_op.externalFX_W, 0.0)
        self.assertEqual(self.basic_op.nu, 15.06e-6)

    def test_property_access_cached(self):
        """Test that cached computed properties are accessible."""
        # Scalar cached property.
        self.assertIsInstance(self.basic_op.qInf__E, float)

        # Transformation matrix cached properties.
        self.assertEqual(self.basic_op.T_pas_GP1_CgP1_to_BP1_CgP1.shape, (4, 4))
        self.assertEqual(self.basic_op.T_pas_BP1_CgP1_to_GP1_CgP1.shape, (4, 4))
        self.assertEqual(self.basic_op.T_pas_BP1_CgP1_to_W_CgP1.shape, (4, 4))
        self.assertEqual(self.basic_op.T_pas_W_CgP1_to_BP1_CgP1.shape, (4, 4))
        self.assertEqual(self.basic_op.T_pas_GP1_CgP1_to_W_CgP1.shape, (4, 4))
        self.assertEqual(self.basic_op.T_pas_W_CgP1_to_GP1_CgP1.shape, (4, 4))
        self.assertEqual(self.basic_op.T_pas_E_CgP1_to_BP1_CgP1.shape, (4, 4))
        self.assertEqual(self.basic_op.T_pas_BP1_CgP1_to_E_CgP1.shape, (4, 4))
        self.assertEqual(self.basic_op.T_pas_E_CgP1_to_GP1_CgP1.shape, (4, 4))
        self.assertEqual(self.basic_op.T_pas_GP1_CgP1_to_E_CgP1.shape, (4, 4))

        # Freestream cached properties.
        self.assertEqual(self.basic_op.vInfHat_GP1__E.shape, (3,))
        self.assertEqual(self.basic_op.vInf_GP1__E.shape, (3,))

    def test_property_access_surface(self):
        """Test that surface effect properties are accessible on a surface enabled
        OperatingPoint.
        """
        self.assertIsNotNone(self.surface_op.surfaceNormal_E)
        self.assertIsNotNone(self.surface_op.surfacePoint_E_Eo)
        self.assertIsNotNone(self.surface_op.surfaceNormal_GP1)
        self.assertIsNotNone(self.surface_op.surfacePoint_GP1_CgP1)
        self.assertIsNotNone(self.surface_op.surfaceReflect_T_act_GP1_CgP1)

    def test_property_access_no_surface(self):
        """Test that surface effect properties return None when no surface is
        defined.
        """
        self.assertIsNone(self.basic_op.surfaceNormal_GP1)
        self.assertIsNone(self.basic_op.surfacePoint_GP1_CgP1)
        self.assertIsNone(self.basic_op.surfaceReflect_T_act_GP1_CgP1)

    def test_deepcopy(self):
        """Test that copy.deepcopy produces a correct independent copy."""
        # Access cached properties before copying.
        _ = self.basic_op.qInf__E
        _ = self.basic_op.T_pas_GP1_CgP1_to_W_CgP1
        _ = self.basic_op.vInf_GP1__E

        copied = copy.deepcopy(self.basic_op)

        # Verify the copy is a separate instance.
        self.assertIsNot(copied, self.basic_op)

        # Verify immutable property values match.
        self.assertEqual(copied.rho, self.basic_op.rho)
        self.assertEqual(copied.vCg__E, self.basic_op.vCg__E)
        self.assertEqual(copied.alpha, self.basic_op.alpha)
        self.assertEqual(copied.beta, self.basic_op.beta)
        npt.assert_array_equal(
            copied.angles_E_to_BP1_izyx, self.basic_op.angles_E_to_BP1_izyx
        )
        npt.assert_array_equal(copied.CgP1_E_Eo, self.basic_op.CgP1_E_Eo)
        self.assertEqual(copied.externalFX_W, self.basic_op.externalFX_W)
        self.assertEqual(copied.nu, self.basic_op.nu)

        # Verify cached property values match.
        self.assertEqual(copied.qInf__E, self.basic_op.qInf__E)
        npt.assert_array_equal(
            copied.T_pas_GP1_CgP1_to_W_CgP1,
            self.basic_op.T_pas_GP1_CgP1_to_W_CgP1,
        )
        npt.assert_array_equal(copied.vInf_GP1__E, self.basic_op.vInf_GP1__E)

        # Verify arrays are independent (not shared references).
        self.assertIsNot(
            copied.angles_E_to_BP1_izyx, self.basic_op.angles_E_to_BP1_izyx
        )
        self.assertIsNot(copied.CgP1_E_Eo, self.basic_op.CgP1_E_Eo)

    def test_deepcopy_with_surface(self):
        """Test that copy.deepcopy works correctly on a surface enabled
        OperatingPoint.
        """
        # Access surface cached properties before copying.
        _ = self.surface_op.surfaceNormal_GP1
        _ = self.surface_op.surfacePoint_GP1_CgP1
        _ = self.surface_op.surfaceReflect_T_act_GP1_CgP1

        copied = copy.deepcopy(self.surface_op)

        # Verify the copy is a separate instance.
        self.assertIsNot(copied, self.surface_op)

        # Verify surface properties match.
        npt.assert_array_equal(copied.surfaceNormal_E, self.surface_op.surfaceNormal_E)
        npt.assert_array_equal(
            copied.surfacePoint_E_Eo, self.surface_op.surfacePoint_E_Eo
        )
        npt.assert_array_equal(
            copied.surfaceNormal_GP1, self.surface_op.surfaceNormal_GP1
        )
        npt.assert_array_equal(
            copied.surfacePoint_GP1_CgP1,
            self.surface_op.surfacePoint_GP1_CgP1,
        )
        npt.assert_array_equal(
            copied.surfaceReflect_T_act_GP1_CgP1,
            self.surface_op.surfaceReflect_T_act_GP1_CgP1,
        )
