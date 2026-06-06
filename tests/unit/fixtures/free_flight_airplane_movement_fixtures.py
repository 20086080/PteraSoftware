"""This module contains functions to create FreeFlightAirplaneMovements for use in
tests."""

import pterasoftware as ps

from . import free_flight_wing_movement_fixtures, geometry_fixtures


def make_basic_free_flight_airplane_movement_fixture():
    """This method makes a fixture that is a FreeFlightAirplaneMovement with
    general-purpose moderate values.

    :return basic_free_flight_airplane_movement_fixture: FreeFlightAirplaneMovement
        This is the FreeFlightAirplaneMovement with general-purpose values. Its single
        FreeFlightWingMovement oscillates, so that a representative oscillating Airplane
        can be generated.
    """
    # Initialize the constructing fixtures. The base Airplane is the first Airplane in
    # a simulation, so its Cg_GP1_CgP1 parameter is all zeros.
    base_airplane = geometry_fixtures.make_first_airplane_fixture()
    wing_movements = [
        free_flight_wing_movement_fixtures.make_basic_free_flight_wing_movement_fixture()
    ]

    # Create the basic FreeFlightAirplaneMovement.
    basic_free_flight_airplane_movement_fixture = (
        ps.movements.free_flight_airplane_movement.FreeFlightAirplaneMovement(
            base_airplane=base_airplane,
            wing_movements=wing_movements,
            ampCg_GP1_CgP1=(0.0, 0.0, 0.0),
            periodCg_GP1_CgP1=(0.0, 0.0, 0.0),
            spacingCg_GP1_CgP1=("sine", "sine", "sine"),
            phaseCg_GP1_CgP1=(0.0, 0.0, 0.0),
        )
    )

    # Return the FreeFlightAirplaneMovement fixture.
    return basic_free_flight_airplane_movement_fixture
