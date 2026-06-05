"""This module contains functions to create FreeFlightWingMovements for use in
tests."""

import pterasoftware as ps

from . import geometry_fixtures


def make_basic_free_flight_wing_movement_fixture():
    """This method makes a fixture that is a FreeFlightWingMovement with general-purpose
    moderate values.

    :return basic_free_flight_wing_movement_fixture: FreeFlightWingMovement
        This is the FreeFlightWingMovement with general-purpose values. The root
        WingCrossSection's movement is static, while the tip WingCrossSection's movement
        oscillates, so that a representative oscillating Wing can be generated.
    """
    # Initialize the constructing fixtures. The root WingCrossSection's movement stays
    # static (keeping its required zero Lp_Wcsp_Lpp), while the tip's movement
    # oscillates.
    base_wing = geometry_fixtures.make_origin_wing_fixture()
    root_wing_cross_section, tip_wing_cross_section = base_wing.wing_cross_sections
    wing_cross_section_movements = [
        ps.movements.free_flight_wing_cross_section_movement.FreeFlightWingCrossSectionMovement(
            base_wing_cross_section=root_wing_cross_section,
        ),
        ps.movements.free_flight_wing_cross_section_movement.FreeFlightWingCrossSectionMovement(
            base_wing_cross_section=tip_wing_cross_section,
            ampLp_Wcsp_Lpp=(0.4, 0.3, 0.15),
            periodLp_Wcsp_Lpp=(2.0, 2.0, 2.0),
            spacingLp_Wcsp_Lpp=("sine", "sine", "sine"),
            phaseLp_Wcsp_Lpp=(0.0, 0.0, 0.0),
            ampAngles_Wcsp_to_Wcs_ixyz=(15.0, 10.0, 5.0),
            periodAngles_Wcsp_to_Wcs_ixyz=(2.0, 2.0, 2.0),
            spacingAngles_Wcsp_to_Wcs_ixyz=("sine", "sine", "sine"),
            phaseAngles_Wcsp_to_Wcs_ixyz=(0.0, 0.0, 0.0),
        ),
    ]

    # Create the basic FreeFlightWingMovement.
    basic_free_flight_wing_movement_fixture = (
        ps.movements.free_flight_wing_movement.FreeFlightWingMovement(
            base_wing=base_wing,
            wing_cross_section_movements=wing_cross_section_movements,
            ampLer_Gs_Cgs=(0.1, 0.05, 0.08),
            periodLer_Gs_Cgs=(2.0, 2.0, 2.0),
            spacingLer_Gs_Cgs=("sine", "sine", "sine"),
            phaseLer_Gs_Cgs=(0.0, 0.0, 0.0),
            ampAngles_Gs_to_Wn_ixyz=(5.0, 3.0, 2.0),
            periodAngles_Gs_to_Wn_ixyz=(2.0, 2.0, 2.0),
            spacingAngles_Gs_to_Wn_ixyz=("sine", "sine", "sine"),
            phaseAngles_Gs_to_Wn_ixyz=(0.0, 0.0, 0.0),
        )
    )

    # Return the FreeFlightWingMovement fixture.
    return basic_free_flight_wing_movement_fixture
