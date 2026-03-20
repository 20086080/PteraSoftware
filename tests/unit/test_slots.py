"""This module contains classes to test __slots__ enforcement on slotted classes."""

import copy
import unittest

import numpy as np
import numpy.testing as npt

import pterasoftware as ps

# noinspection PyProtectedMember
from pterasoftware import _panel
from pterasoftware._vortices import _line_vortex
from tests.unit.fixtures import (
    geometry_fixtures,
    horseshoe_vortex_fixtures,
    line_vortex_fixtures,
    operating_point_fixtures,
    panel_fixtures,
    ring_vortex_fixtures,
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


class TestRingVortexSlots(unittest.TestCase):
    """This class contains tests to verify __slots__ enforcement on RingVortex."""

    def setUp(self):
        """Set up test fixtures for RingVortex slots tests."""
        self.ring_vortex = ring_vortex_fixtures.make_basic_ring_vortex_fixture()

    def test_slots_defined(self):
        """Test that __slots__ is defined on RingVortex."""
        self.assertTrue(hasattr(ps._vortices.ring_vortex.RingVortex, "__slots__"))

    def test_no_instance_dict(self):
        """Test that RingVortex instances have no __dict__."""
        self.assertFalse(hasattr(self.ring_vortex, "__dict__"))

    def test_dynamic_attribute_raises(self):
        """Test that dynamic attribute assignment raises AttributeError."""
        with self.assertRaises(AttributeError):
            self.ring_vortex.nonexistent_attribute = 42

    def test_property_access(self):
        """Test that all properties remain accessible after adding __slots__."""
        # Immutable corner point properties.
        npt.assert_array_equal(
            self.ring_vortex.Frrvp_GP1_CgP1,
            np.array([0.0, 0.5, 0.0]),
        )
        npt.assert_array_equal(
            self.ring_vortex.Flrvp_GP1_CgP1,
            np.array([0.0, -0.5, 0.0]),
        )
        npt.assert_array_equal(
            self.ring_vortex.Blrvp_GP1_CgP1,
            np.array([1.0, -0.5, 0.0]),
        )
        npt.assert_array_equal(
            self.ring_vortex.Brrvp_GP1_CgP1,
            np.array([1.0, 0.5, 0.0]),
        )

        # Mutable attributes.
        self.assertEqual(self.ring_vortex.strength, 1.0)
        self.assertEqual(self.ring_vortex.age, 0.0)

        # Cached computed properties.
        self.assertEqual(self.ring_vortex.Crvp_GP1_CgP1.shape, (3,))
        self.assertIsInstance(self.ring_vortex.area, float)
        self.assertIsInstance(self.ring_vortex.front_leg, _line_vortex.LineVortex)
        self.assertIsInstance(self.ring_vortex.left_leg, _line_vortex.LineVortex)
        self.assertIsInstance(self.ring_vortex.back_leg, _line_vortex.LineVortex)
        self.assertIsInstance(self.ring_vortex.right_leg, _line_vortex.LineVortex)

    def test_deepcopy(self):
        """Test that copy.deepcopy produces a correct independent copy."""
        # Access cached properties before copying.
        _ = self.ring_vortex.Crvp_GP1_CgP1
        _ = self.ring_vortex.front_leg
        _ = self.ring_vortex.area

        copied = copy.deepcopy(self.ring_vortex)

        # Verify the copy is a separate instance.
        self.assertIsNot(copied, self.ring_vortex)

        # Verify property values match.
        npt.assert_array_equal(copied.Frrvp_GP1_CgP1, self.ring_vortex.Frrvp_GP1_CgP1)
        npt.assert_array_equal(copied.Flrvp_GP1_CgP1, self.ring_vortex.Flrvp_GP1_CgP1)
        self.assertEqual(copied.strength, self.ring_vortex.strength)
        self.assertEqual(copied.age, self.ring_vortex.age)
        npt.assert_array_equal(copied.Crvp_GP1_CgP1, self.ring_vortex.Crvp_GP1_CgP1)
        self.assertEqual(copied.area, self.ring_vortex.area)

        # Verify arrays are independent.
        self.assertIsNot(copied.Frrvp_GP1_CgP1, self.ring_vortex.Frrvp_GP1_CgP1)


class TestHorseshoeVortexSlots(unittest.TestCase):
    """This class contains tests to verify __slots__ enforcement on
    HorseshoeVortex.
    """

    def setUp(self):
        """Set up test fixtures for HorseshoeVortex slots tests."""
        self.horseshoe_vortex = (
            horseshoe_vortex_fixtures.make_basic_horseshoe_vortex_fixture()
        )

    def test_slots_defined(self):
        """Test that __slots__ is defined on HorseshoeVortex."""
        self.assertTrue(
            hasattr(ps._vortices.horseshoe_vortex.HorseshoeVortex, "__slots__")
        )

    def test_no_instance_dict(self):
        """Test that HorseshoeVortex instances have no __dict__."""
        self.assertFalse(hasattr(self.horseshoe_vortex, "__dict__"))

    def test_dynamic_attribute_raises(self):
        """Test that dynamic attribute assignment raises AttributeError."""
        with self.assertRaises(AttributeError):
            self.horseshoe_vortex.nonexistent_attribute = 42

    def test_property_access(self):
        """Test that all properties remain accessible after adding __slots__."""
        # Immutable properties.
        npt.assert_array_equal(
            self.horseshoe_vortex.Frhvp_GP1_CgP1,
            np.array([0.0, 0.5, 0.0]),
        )
        npt.assert_array_equal(
            self.horseshoe_vortex.Flhvp_GP1_CgP1,
            np.array([0.0, -0.5, 0.0]),
        )
        self.assertEqual(self.horseshoe_vortex.leftLegVector_GP1.shape, (3,))
        self.assertEqual(self.horseshoe_vortex.left_right_leg_lengths, 20.0)

        # Mutable attribute.
        self.assertEqual(self.horseshoe_vortex.strength, 1.0)

        # Cached computed properties.
        self.assertEqual(self.horseshoe_vortex.Brhvp_GP1_CgP1.shape, (3,))
        self.assertEqual(self.horseshoe_vortex.Blhvp_GP1_CgP1.shape, (3,))
        self.assertIsInstance(self.horseshoe_vortex.right_leg, _line_vortex.LineVortex)
        self.assertIsInstance(self.horseshoe_vortex.finite_leg, _line_vortex.LineVortex)
        self.assertIsInstance(self.horseshoe_vortex.left_leg, _line_vortex.LineVortex)

    def test_deepcopy(self):
        """Test that copy.deepcopy produces a correct independent copy."""
        # Access cached properties before copying.
        _ = self.horseshoe_vortex.Brhvp_GP1_CgP1
        _ = self.horseshoe_vortex.right_leg

        copied = copy.deepcopy(self.horseshoe_vortex)

        # Verify the copy is a separate instance.
        self.assertIsNot(copied, self.horseshoe_vortex)

        # Verify property values match.
        npt.assert_array_equal(
            copied.Frhvp_GP1_CgP1, self.horseshoe_vortex.Frhvp_GP1_CgP1
        )
        npt.assert_array_equal(
            copied.Flhvp_GP1_CgP1, self.horseshoe_vortex.Flhvp_GP1_CgP1
        )
        self.assertEqual(copied.strength, self.horseshoe_vortex.strength)
        npt.assert_array_equal(
            copied.Brhvp_GP1_CgP1, self.horseshoe_vortex.Brhvp_GP1_CgP1
        )
        npt.assert_array_equal(
            copied.Blhvp_GP1_CgP1, self.horseshoe_vortex.Blhvp_GP1_CgP1
        )

        # Verify arrays are independent.
        self.assertIsNot(copied.Frhvp_GP1_CgP1, self.horseshoe_vortex.Frhvp_GP1_CgP1)


class TestWingCrossSectionSlots(unittest.TestCase):
    """This class contains tests to verify __slots__ enforcement on
    WingCrossSection.
    """

    def setUp(self):
        """Set up test fixtures for WingCrossSection slots tests."""
        self.wing_cross_section = (
            geometry_fixtures.make_basic_wing_cross_section_fixture()
        )

    def test_slots_defined(self):
        """Test that __slots__ is defined on WingCrossSection."""
        self.assertTrue(
            hasattr(ps.geometry.wing_cross_section.WingCrossSection, "__slots__")
        )

    def test_no_instance_dict(self):
        """Test that WingCrossSection instances have no __dict__."""
        self.assertFalse(hasattr(self.wing_cross_section, "__dict__"))

    def test_dynamic_attribute_raises(self):
        """Test that dynamic attribute assignment raises AttributeError."""
        with self.assertRaises(AttributeError):
            self.wing_cross_section.nonexistent_attribute = 42

    def test_property_access(self):
        """Test that all properties remain accessible after adding __slots__."""
        # Immutable properties.
        self.assertIsInstance(
            self.wing_cross_section.airfoil, ps.geometry.airfoil.Airfoil
        )
        self.assertEqual(self.wing_cross_section.num_spanwise_panels, 8)
        self.assertEqual(self.wing_cross_section.chord, 1.5)
        npt.assert_array_equal(
            self.wing_cross_section.Lp_Wcsp_Lpp,
            np.array([0.2, 0.5, 0.1]),
        )
        npt.assert_array_equal(
            self.wing_cross_section.angles_Wcsp_to_Wcs_ixyz,
            np.array([5.0, -2.0, 3.0]),
        )
        self.assertEqual(self.wing_cross_section.control_surface_hinge_point, 0.75)
        self.assertEqual(self.wing_cross_section.control_surface_deflection, 5.0)
        self.assertEqual(self.wing_cross_section.spanwise_spacing, "cosine")

        # Mutable attribute.
        self.assertEqual(
            self.wing_cross_section.control_surface_symmetry_type, "symmetric"
        )

        # Set once attributes (not yet set by a parent Wing).
        self.assertFalse(self.wing_cross_section.validated)
        self.assertIsNone(self.wing_cross_section.symmetry_type)

        # Cached transformation properties return None when not validated.
        self.assertIsNone(self.wing_cross_section.T_pas_Wcsp_Lpp_to_Wcs_Lp)
        self.assertIsNone(self.wing_cross_section.T_pas_Wcs_Lp_to_Wcsp_Lpp)

    def test_deepcopy_method(self):
        """Test that __deepcopy__ produces a correct independent copy."""
        copied = copy.deepcopy(self.wing_cross_section)

        # Verify the copy is a separate instance.
        self.assertIsNot(copied, self.wing_cross_section)

        # Verify property values match.
        self.assertEqual(copied.chord, self.wing_cross_section.chord)
        self.assertEqual(
            copied.num_spanwise_panels, self.wing_cross_section.num_spanwise_panels
        )
        npt.assert_array_equal(copied.Lp_Wcsp_Lpp, self.wing_cross_section.Lp_Wcsp_Lpp)
        npt.assert_array_equal(
            copied.angles_Wcsp_to_Wcs_ixyz,
            self.wing_cross_section.angles_Wcsp_to_Wcs_ixyz,
        )
        self.assertEqual(
            copied.control_surface_symmetry_type,
            self.wing_cross_section.control_surface_symmetry_type,
        )
        self.assertEqual(
            copied.spanwise_spacing, self.wing_cross_section.spanwise_spacing
        )

        # Verify the Airfoil is a separate instance.
        self.assertIsNot(copied.airfoil, self.wing_cross_section.airfoil)

        # Verify arrays are independent.
        self.assertIsNot(copied.Lp_Wcsp_Lpp, self.wing_cross_section.Lp_Wcsp_Lpp)
        self.assertIsNot(
            copied.angles_Wcsp_to_Wcs_ixyz,
            self.wing_cross_section.angles_Wcsp_to_Wcs_ixyz,
        )

    def test_deepcopy_no_dict(self):
        """Test that a deep copied WingCrossSection has no __dict__."""
        copied = copy.deepcopy(self.wing_cross_section)
        self.assertFalse(hasattr(copied, "__dict__"))


class TestPanelSlots(unittest.TestCase):
    """This class contains tests to verify __slots__ enforcement on Panel."""

    def setUp(self):
        """Set up test fixtures for Panel slots tests."""
        self.panel = panel_fixtures.make_basic_panel_fixture()
        self.fully_configured_panel = (
            panel_fixtures.make_fully_configured_panel_fixture()
        )

    def test_slots_defined(self):
        """Test that __slots__ is defined on Panel."""
        self.assertTrue(hasattr(_panel.Panel, "__slots__"))

    def test_no_instance_dict(self):
        """Test that Panel instances have no __dict__."""
        self.assertFalse(hasattr(self.panel, "__dict__"))

    def test_dynamic_attribute_raises(self):
        """Test that dynamic attribute assignment raises AttributeError."""
        with self.assertRaises(AttributeError):
            self.panel.nonexistent_attribute = 42

    def test_property_access_immutable(self):
        """Test that immutable properties are accessible."""
        self.assertEqual(self.panel.Frpp_G_Cg.shape, (3,))
        self.assertEqual(self.panel.Flpp_G_Cg.shape, (3,))
        self.assertEqual(self.panel.Blpp_G_Cg.shape, (3,))
        self.assertEqual(self.panel.Brpp_G_Cg.shape, (3,))
        self.assertIsInstance(self.panel.is_leading_edge, bool)
        self.assertIsInstance(self.panel.is_trailing_edge, bool)

    def test_property_access_cached(self):
        """Test that cached computed properties are accessible."""
        self.assertEqual(self.panel.rightLeg_G.shape, (3,))
        self.assertEqual(self.panel.frontLeg_G.shape, (3,))
        self.assertEqual(self.panel.leftLeg_G.shape, (3,))
        self.assertEqual(self.panel.backLeg_G.shape, (3,))
        self.assertEqual(self.panel.Frbvp_G_Cg.shape, (3,))
        self.assertEqual(self.panel.Flbvp_G_Cg.shape, (3,))
        self.assertEqual(self.panel.Cpp_G_Cg.shape, (3,))
        self.assertEqual(self.panel.unitNormal_G.shape, (3,))
        self.assertIsInstance(self.panel.area, float)
        self.assertIsInstance(self.panel.aspect_ratio, float)

    def test_property_access_set_once_unset(self):
        """Test that set once properties return None when not yet set."""
        self.assertIsNone(self.panel.Frpp_GP1_CgP1)
        self.assertIsNone(self.panel.is_right_edge)
        self.assertIsNone(self.panel.is_left_edge)
        self.assertIsNone(self.panel.local_chordwise_position)
        self.assertIsNone(self.panel.local_spanwise_position)

    def test_property_access_fully_configured(self):
        """Test that all properties are accessible on a fully configured Panel."""
        self.assertIsNotNone(self.fully_configured_panel.Frpp_GP1_CgP1)
        self.assertIsNotNone(self.fully_configured_panel.is_right_edge)
        self.assertIsNotNone(self.fully_configured_panel.is_left_edge)
        self.assertIsNotNone(self.fully_configured_panel.local_chordwise_position)
        self.assertIsNotNone(self.fully_configured_panel.local_spanwise_position)

    def test_mutable_attributes(self):
        """Test that mutable attributes are accessible and default to None."""
        self.assertIsNone(self.panel.ring_vortex)
        self.assertIsNone(self.panel.horseshoe_vortex)
        self.assertIsNone(self.panel.forces_GP1)
        self.assertIsNone(self.panel.moments_GP1_CgP1)
        self.assertIsNone(self.panel.forces_W)
        self.assertIsNone(self.panel.moments_W_CgP1)

    def test_deepcopy_method(self):
        """Test that __deepcopy__ produces a correct independent copy."""
        # Access cached properties before copying.
        _ = self.panel.rightLeg_G
        _ = self.panel.Cpp_G_Cg
        _ = self.panel.area

        copied = copy.deepcopy(self.panel)

        # Verify the copy is a separate instance.
        self.assertIsNot(copied, self.panel)

        # Verify immutable property values match.
        npt.assert_array_equal(copied.Frpp_G_Cg, self.panel.Frpp_G_Cg)
        npt.assert_array_equal(copied.Flpp_G_Cg, self.panel.Flpp_G_Cg)
        npt.assert_array_equal(copied.Blpp_G_Cg, self.panel.Blpp_G_Cg)
        npt.assert_array_equal(copied.Brpp_G_Cg, self.panel.Brpp_G_Cg)
        self.assertEqual(copied.is_leading_edge, self.panel.is_leading_edge)
        self.assertEqual(copied.is_trailing_edge, self.panel.is_trailing_edge)

        # Verify cached property values match.
        npt.assert_array_equal(copied.rightLeg_G, self.panel.rightLeg_G)
        npt.assert_array_equal(copied.Cpp_G_Cg, self.panel.Cpp_G_Cg)
        self.assertEqual(copied.area, self.panel.area)

        # Verify arrays are independent.
        self.assertIsNot(copied.Frpp_G_Cg, self.panel.Frpp_G_Cg)
        self.assertIsNot(copied.rightLeg_G, self.panel.rightLeg_G)

        # Verify solver state is reset.
        self.assertIsNone(copied.ring_vortex)
        self.assertIsNone(copied.horseshoe_vortex)
        self.assertIsNone(copied.forces_GP1)

    def test_deepcopy_no_dict(self):
        """Test that a deep copied Panel has no __dict__."""
        copied = copy.deepcopy(self.panel)
        self.assertFalse(hasattr(copied, "__dict__"))


class TestWingSlots(unittest.TestCase):
    """This class contains tests to verify __slots__ enforcement on Wing."""

    def setUp(self):
        """Set up test fixtures for Wing slots tests."""
        # Use a Wing from an Airplane so it has been meshed (symmetry_type set).
        airplane = geometry_fixtures.make_first_airplane_fixture()
        self.wing = airplane.wings[0]
        # Also keep an unmeshed Wing for testing pre-mesh state.
        self.unmeshed_wing = geometry_fixtures.make_type_1_wing_fixture()

    def test_slots_defined(self):
        """Test that __slots__ is defined on Wing."""
        self.assertTrue(hasattr(ps.geometry.wing.Wing, "__slots__"))

    def test_no_instance_dict(self):
        """Test that Wing instances have no __dict__."""
        self.assertFalse(hasattr(self.wing, "__dict__"))

    def test_dynamic_attribute_raises(self):
        """Test that dynamic attribute assignment raises AttributeError."""
        with self.assertRaises(AttributeError):
            self.wing.nonexistent_attribute = 42

    def test_property_access_immutable(self):
        """Test that immutable properties are accessible."""
        self.assertIsInstance(self.wing.wing_cross_sections, tuple)
        self.assertIsInstance(self.wing.name, str)
        self.assertEqual(self.wing.Ler_Gs_Cgs.shape, (3,))
        self.assertEqual(self.wing.angles_Gs_to_Wn_ixyz.shape, (3,))
        self.assertIsInstance(self.wing.num_chordwise_panels, int)
        self.assertIsInstance(self.wing.chordwise_spacing, str)

    def test_property_access_mutable_symmetry(self):
        """Test that mutable symmetry attributes are accessible."""
        self.assertIsInstance(self.wing.symmetric, bool)
        self.assertIsInstance(self.wing.mirror_only, bool)

    def test_property_access_set_once_unset(self):
        """Test that set once properties return None when not yet meshed."""
        self.assertIsNone(self.unmeshed_wing.symmetry_type)
        self.assertIsNone(self.unmeshed_wing.num_spanwise_panels)
        self.assertIsNone(self.unmeshed_wing.num_panels)
        self.assertIsNone(self.unmeshed_wing.panels)

    def test_property_access_cached(self):
        """Test that cached transformation properties are accessible."""
        self.assertEqual(self.wing.T_pas_G_Cg_to_Wn_Ler.shape, (4, 4))
        self.assertEqual(self.wing.T_pas_Wn_Ler_to_G_Cg.shape, (4, 4))
        self.assertEqual(self.wing.WnX_G.shape, (3,))
        self.assertEqual(self.wing.WnY_G.shape, (3,))
        self.assertEqual(self.wing.WnZ_G.shape, (3,))
        self.assertIsInstance(self.wing.children_T_pas_Wn_Ler_to_Wcs_Lp, list)
        self.assertIsInstance(self.wing.children_T_pas_Wcs_Lp_to_Wn_Ler, list)
        self.assertIsInstance(self.wing.children_T_pas_G_Cg_to_Wcs_Lp, list)
        self.assertIsInstance(self.wing.children_T_pas_Wcs_Lp_to_G_Cg, list)

    def test_deepcopy_method(self):
        """Test that __deepcopy__ produces a correct independent copy."""
        # Access cached properties before copying.
        original_T = self.wing.T_pas_G_Cg_to_Wn_Ler
        original_WnX = self.wing.WnX_G
        _ = self.wing.children_T_pas_Wn_Ler_to_Wcs_Lp
        self.assertIsNotNone(original_T)
        self.assertIsNotNone(original_WnX)

        copied = copy.deepcopy(self.wing)

        # Verify the copy is a separate instance.
        self.assertIsNot(copied, self.wing)

        # Verify immutable property values match.
        self.assertEqual(copied.name, self.wing.name)
        npt.assert_array_equal(copied.Ler_Gs_Cgs, self.wing.Ler_Gs_Cgs)
        npt.assert_array_equal(
            copied.angles_Gs_to_Wn_ixyz, self.wing.angles_Gs_to_Wn_ixyz
        )
        self.assertEqual(copied.num_chordwise_panels, self.wing.num_chordwise_panels)
        self.assertEqual(copied.chordwise_spacing, self.wing.chordwise_spacing)

        # Verify cached property values match.
        npt.assert_array_equal(
            copied.T_pas_G_Cg_to_Wn_Ler, self.wing.T_pas_G_Cg_to_Wn_Ler
        )
        npt.assert_array_equal(copied.WnX_G, self.wing.WnX_G)

        # Verify WingCrossSections are independent.
        self.assertIsNot(
            copied.wing_cross_sections[0], self.wing.wing_cross_sections[0]
        )

        # Verify arrays are independent.
        self.assertIsNot(copied.Ler_Gs_Cgs, self.wing.Ler_Gs_Cgs)
        self.assertIsNot(copied.T_pas_G_Cg_to_Wn_Ler, self.wing.T_pas_G_Cg_to_Wn_Ler)

    def test_deepcopy_no_dict(self):
        """Test that a deep copied Wing has no __dict__."""
        copied = copy.deepcopy(self.wing)
        self.assertFalse(hasattr(copied, "__dict__"))


class TestAirplaneSlots(unittest.TestCase):
    """This class contains tests to verify __slots__ enforcement on Airplane."""

    def setUp(self):
        """Set up test fixtures for Airplane slots tests."""
        self.airplane = geometry_fixtures.make_first_airplane_fixture()

    def test_slots_defined(self):
        """Test that __slots__ is defined on Airplane."""
        self.assertTrue(hasattr(ps.geometry.airplane.Airplane, "__slots__"))

    def test_no_instance_dict(self):
        """Test that Airplane instances have no __dict__."""
        self.assertFalse(hasattr(self.airplane, "__dict__"))

    def test_dynamic_attribute_raises(self):
        """Test that dynamic attribute assignment raises AttributeError."""
        with self.assertRaises(AttributeError):
            self.airplane.nonexistent_attribute = 42

    def test_property_access_immutable(self):
        """Test that immutable properties are accessible."""
        self.assertIsInstance(self.airplane.wings, tuple)
        self.assertEqual(self.airplane.name, "First Test Airplane")
        npt.assert_array_equal(self.airplane.Cg_GP1_CgP1, np.array([0.0, 0.0, 0.0]))
        self.assertEqual(self.airplane.weight, 1500.0)
        self.assertIsInstance(self.airplane.s_ref, float)
        self.assertIsInstance(self.airplane.c_ref, float)
        self.assertIsInstance(self.airplane.b_ref, float)

    def test_property_access_cached(self):
        """Test that cached derived properties are accessible."""
        self.assertIsInstance(self.airplane.num_panels, int)
        self.assertEqual(self.airplane.T_pas_G_Cg_to_GP1_CgP1.shape, (4, 4))

    def test_mutable_attributes(self):
        """Test that mutable load attributes are accessible and default to None."""
        self.assertIsNone(self.airplane.forces_W)
        self.assertIsNone(self.airplane.forceCoefficients_W)
        self.assertIsNone(self.airplane.moments_W_CgP1)
        self.assertIsNone(self.airplane.momentCoefficients_W_CgP1)

    def test_deepcopy_method(self):
        """Test that __deepcopy__ produces a correct independent copy."""
        # Access cached properties before copying.
        _ = self.airplane.num_panels
        _ = self.airplane.T_pas_G_Cg_to_GP1_CgP1

        copied = copy.deepcopy(self.airplane)

        # Verify the copy is a separate instance.
        self.assertIsNot(copied, self.airplane)

        # Verify immutable property values match.
        self.assertEqual(copied.name, self.airplane.name)
        npt.assert_array_equal(copied.Cg_GP1_CgP1, self.airplane.Cg_GP1_CgP1)
        self.assertEqual(copied.weight, self.airplane.weight)
        self.assertEqual(copied.s_ref, self.airplane.s_ref)
        self.assertEqual(copied.c_ref, self.airplane.c_ref)
        self.assertEqual(copied.b_ref, self.airplane.b_ref)

        # Verify cached property values match.
        self.assertEqual(copied.num_panels, self.airplane.num_panels)
        npt.assert_array_equal(
            copied.T_pas_G_Cg_to_GP1_CgP1,
            self.airplane.T_pas_G_Cg_to_GP1_CgP1,
        )

        # Verify Wings are independent.
        self.assertIsNot(copied.wings[0], self.airplane.wings[0])

        # Verify arrays are independent.
        self.assertIsNot(copied.Cg_GP1_CgP1, self.airplane.Cg_GP1_CgP1)
        self.assertIsNot(
            copied.T_pas_G_Cg_to_GP1_CgP1,
            self.airplane.T_pas_G_Cg_to_GP1_CgP1,
        )

        # Verify solver state is reset.
        self.assertIsNone(copied.forces_W)
        self.assertIsNone(copied.forceCoefficients_W)
        self.assertIsNone(copied.moments_W_CgP1)
        self.assertIsNone(copied.momentCoefficients_W_CgP1)

    def test_deep_copy_with_cg(self):
        """Test that deep_copy_with_Cg_GP1_CgP1 works with __slots__."""
        copied = self.airplane.deep_copy_with_Cg_GP1_CgP1([2.0, 1.0, 0.5])

        # Verify the copy is a separate instance.
        self.assertIsNot(copied, self.airplane)

        # Verify the new position.
        npt.assert_array_equal(copied.Cg_GP1_CgP1, np.array([2.0, 1.0, 0.5]))

        # Verify other immutable properties are preserved.
        self.assertEqual(copied.name, self.airplane.name)
        self.assertEqual(copied.weight, self.airplane.weight)

        # Verify no __dict__.
        self.assertFalse(hasattr(copied, "__dict__"))

    def test_deepcopy_no_dict(self):
        """Test that a deep copied Airplane has no __dict__."""
        copied = copy.deepcopy(self.airplane)
        self.assertFalse(hasattr(copied, "__dict__"))
