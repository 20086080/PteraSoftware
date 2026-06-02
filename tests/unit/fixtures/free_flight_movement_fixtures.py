"""This module contains functions to create FreeFlightMovements for use in tests."""

import pterasoftware as ps

from . import (
    free_flight_airplane_movement_fixtures,
    free_flight_operating_point_movement_fixtures,
    geometry_fixtures,
)


def make_basic_free_flight_movement_fixture():
    """This method makes a fixture that is a FreeFlightMovement with general-purpose
    moderate values.

    :return basic_free_flight_movement_fixture: FreeFlightMovement
        This is the FreeFlightMovement with general-purpose values. Its single
        FreeFlightAirplaneMovement oscillates, so the FreeFlightMovement is non static.
    """
    # Initialize the constructing fixtures.
    airplane_movements = [
        free_flight_airplane_movement_fixtures.make_basic_free_flight_airplane_movement_fixture()
    ]
    operating_point_movement = (
        free_flight_operating_point_movement_fixtures.make_basic_free_flight_operating_point_movement_fixture()
    )

    # Create the basic FreeFlightMovement.
    basic_free_flight_movement_fixture = (
        ps.movements.free_flight_movement.FreeFlightMovement(
            airplane_movements=airplane_movements,
            operating_point_movement=operating_point_movement,
            delta_time=0.1,
            prescribed_num_steps=2,
            free_num_steps=3,
        )
    )

    # Return the FreeFlightMovement fixture.
    return basic_free_flight_movement_fixture


def make_static_free_flight_movement_fixture():
    """This method makes a fixture that is a FreeFlightMovement with all static
    components.

    :return static_free_flight_movement_fixture: FreeFlightMovement
        This is the FreeFlightMovement with no prescribed motion. Because the
        FreeFlightOperatingPointMovement never oscillates, the only possible motion comes
        from the FreeFlightAirplaneMovement, which is static here, so the
        FreeFlightMovement is static.
    """
    # Initialize the constructing fixtures. The base Airplane is the first Airplane in a
    # simulation, so its Cg_GP1_CgP1 parameter is all zeros. The base Wing is a type 1
    # Wing (no symmetry), so its static motion cannot change its symmetry type.
    base_airplane = geometry_fixtures.make_first_airplane_fixture()
    base_wing = geometry_fixtures.make_origin_wing_fixture()
    root_wing_cross_section, tip_wing_cross_section = base_wing.wing_cross_sections

    # Create static FreeFlightWingCrossSectionMovements (default zero oscillation).
    wing_cross_section_movements = [
        ps.movements.free_flight_wing_cross_section_movement.FreeFlightWingCrossSectionMovement(
            base_wing_cross_section=root_wing_cross_section,
        ),
        ps.movements.free_flight_wing_cross_section_movement.FreeFlightWingCrossSectionMovement(
            base_wing_cross_section=tip_wing_cross_section,
        ),
    ]

    # Create a static FreeFlightWingMovement (default zero oscillation).
    wing_movement = ps.movements.free_flight_wing_movement.FreeFlightWingMovement(
        base_wing=base_wing,
        wing_cross_section_movements=wing_cross_section_movements,
    )

    # Create a static FreeFlightAirplaneMovement (default zero oscillation).
    airplane_movement = (
        ps.movements.free_flight_airplane_movement.FreeFlightAirplaneMovement(
            base_airplane=base_airplane,
            wing_movements=[wing_movement],
        )
    )

    operating_point_movement = (
        free_flight_operating_point_movement_fixtures.make_basic_free_flight_operating_point_movement_fixture()
    )

    # Create the static FreeFlightMovement.
    static_free_flight_movement_fixture = (
        ps.movements.free_flight_movement.FreeFlightMovement(
            airplane_movements=[airplane_movement],
            operating_point_movement=operating_point_movement,
            delta_time=0.1,
            prescribed_num_steps=2,
            free_num_steps=3,
        )
    )

    # Return the FreeFlightMovement fixture.
    return static_free_flight_movement_fixture


def make_free_flight_movement_with_multiple_airplanes_fixture():
    """This method makes a fixture that is a FreeFlightMovement with multiple
    FreeFlightAirplaneMovements.

    :return free_flight_movement_with_multiple_airplanes_fixture: FreeFlightMovement
        This is the FreeFlightMovement with two FreeFlightAirplaneMovements.
    """
    # Initialize the constructing fixtures with two oscillating
    # FreeFlightAirplaneMovements.
    airplane_movements = [
        free_flight_airplane_movement_fixtures.make_basic_free_flight_airplane_movement_fixture(),
        free_flight_airplane_movement_fixtures.make_basic_free_flight_airplane_movement_fixture(),
    ]
    operating_point_movement = (
        free_flight_operating_point_movement_fixtures.make_basic_free_flight_operating_point_movement_fixture()
    )

    # Create the FreeFlightMovement with multiple FreeFlightAirplaneMovements.
    free_flight_movement_with_multiple_airplanes_fixture = (
        ps.movements.free_flight_movement.FreeFlightMovement(
            airplane_movements=airplane_movements,
            operating_point_movement=operating_point_movement,
            delta_time=0.1,
            prescribed_num_steps=2,
            free_num_steps=3,
        )
    )

    # Return the FreeFlightMovement fixture.
    return free_flight_movement_with_multiple_airplanes_fixture
