"""This module contains functions to create FreeFlightWingCrossSectionMovements for
use in tests."""

import pterasoftware as ps

from . import geometry_fixtures


def make_basic_free_flight_wing_cross_section_movement_fixture():
    """This method makes a fixture that is a FreeFlightWingCrossSectionMovement with
    general-purpose moderate values.

    :return basic_free_flight_wing_cross_section_movement_fixture:
        FreeFlightWingCrossSectionMovement
        This is the FreeFlightWingCrossSectionMovement with general-purpose values.
    """
    # Initialize the constructing fixture.
    # Use the tip fixture to ensure Lp values stay non negative during oscillation.
    base_wing_cross_section = geometry_fixtures.make_tip_wing_cross_section_fixture()

    # Create the basic FreeFlightWingCrossSectionMovement.
    basic_free_flight_wing_cross_section_movement_fixture = ps.movements.free_flight_wing_cross_section_movement.FreeFlightWingCrossSectionMovement(
        base_wing_cross_section=base_wing_cross_section,
        ampLp_Wcsp_Lpp=(0.4, 0.3, 0.15),
        periodLp_Wcsp_Lpp=(2.0, 2.0, 2.0),
        spacingLp_Wcsp_Lpp=("sine", "sine", "sine"),
        phaseLp_Wcsp_Lpp=(0.0, 0.0, 0.0),
        ampAngles_Wcsp_to_Wcs_ixyz=(15.0, 10.0, 5.0),
        periodAngles_Wcsp_to_Wcs_ixyz=(2.0, 2.0, 2.0),
        spacingAngles_Wcsp_to_Wcs_ixyz=("sine", "sine", "sine"),
        phaseAngles_Wcsp_to_Wcs_ixyz=(0.0, 0.0, 0.0),
    )

    # Return the FreeFlightWingCrossSectionMovement fixture.
    return basic_free_flight_wing_cross_section_movement_fixture
