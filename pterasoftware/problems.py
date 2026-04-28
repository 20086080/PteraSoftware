"""Contains the SteadyProblem, UnsteadyProblem, and FreeFlightUnsteadyProblem classes.

**Contains the following classes:**

SteadyProblem: A class used to contain steady aerodynamics problems.

UnsteadyProblem: A class used to contain unsteady aerodynamics problems.

FreeFlightUnsteadyProblem: A class used to contain problems with coupled unsteady
aerodynamics and rigid body dynamics.

**Contains the following functions:**

None
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import numpy as np

from . import (
    _core,
    _mujoco_model,
    _parameter_validation,
    _transformations,
    geometry,
    movements,
)
from . import operating_point as operating_point_mod
from .movements import free_flight_movement

if TYPE_CHECKING:
    from ._coupled_unsteady_ring_vortex_lattice_method import (
        CoupledUnsteadyRingVortexLatticeMethodSolver,
    )


class SteadyProblem:
    """A class used to contain steady aerodynamics problems.

    **Contains the following methods:**

    reynolds_numbers: A tuple of Reynolds numbers, one for each Airplane in the
    SteadyProblem.
    """

    __slots__ = (
        "_airplanes",
        "_operating_point",
        "_reynolds_numbers",
    )

    def __init__(
        self,
        airplanes: list[geometry.airplane.Airplane],
        operating_point: operating_point_mod.OperatingPoint,
    ) -> None:
        """The initialization method.

        :param airplanes: The list of the Airplanes for this SteadyProblem.
        :param operating_point: The OperatingPoint for this SteadyProblem.
        :return: None
        """
        # Validate and store immutable attributes.
        if not isinstance(airplanes, list):
            raise TypeError("airplanes must be a list.")
        if len(airplanes) < 1:
            raise ValueError("airplanes must have at least one element.")
        for airplane in airplanes:
            if not isinstance(airplane, geometry.airplane.Airplane):
                raise TypeError("Every element in airplanes must be an Airplane.")
        # Store as tuple to prevent external mutation via .append(), .pop(), etc.
        self._airplanes: tuple[geometry.airplane.Airplane, ...] = tuple(airplanes)

        if not isinstance(operating_point, operating_point_mod.OperatingPoint):
            raise TypeError("operating_point must be an OperatingPoint.")
        self._operating_point = operating_point

        # Initialize the caches for the properties derived from the immutable
        # attributes.
        self._reynolds_numbers: tuple[float, ...] | None = None

        # Validate that the first Airplane has Cg_GP1_CgP1 set to zeros.
        self._airplanes[0].validate_first_airplane_constraints()

        # Populate GP1_CgP1 coordinates for all Airplanes' Panels. This finds the
        # Panels' positions in the first Airplane's geometry axes, relative to the
        # first Airplane's CG based on their locally defined positions.
        for airplane in self._airplanes:
            # Compute the passive transformation matrix from this Airplane's local
            # geometry axes, relative to its CG, to the first Airplane's geometry axes,
            # relative to the first Airplane's CG.
            T_pas_G_Cg_to_GP1_CgP1 = airplane.T_pas_G_Cg_to_GP1_CgP1

            for wing in airplane.wings:
                assert wing.panels is not None

                for panel in np.ravel(wing.panels):
                    panel.Frpp_GP1_CgP1 = _transformations.apply_T_to_vectors(
                        T_pas_G_Cg_to_GP1_CgP1, panel.Frpp_G_Cg, has_point=True
                    )
                    panel.Flpp_GP1_CgP1 = _transformations.apply_T_to_vectors(
                        T_pas_G_Cg_to_GP1_CgP1, panel.Flpp_G_Cg, has_point=True
                    )
                    panel.Blpp_GP1_CgP1 = _transformations.apply_T_to_vectors(
                        T_pas_G_Cg_to_GP1_CgP1, panel.Blpp_G_Cg, has_point=True
                    )
                    panel.Brpp_GP1_CgP1 = _transformations.apply_T_to_vectors(
                        T_pas_G_Cg_to_GP1_CgP1, panel.Brpp_G_Cg, has_point=True
                    )

    # --- Immutable: read only properties ---
    @property
    def airplanes(self) -> tuple[geometry.airplane.Airplane, ...]:
        return self._airplanes

    @property
    def operating_point(self) -> operating_point_mod.OperatingPoint:
        return self._operating_point

    # --- Immutable derived: manual lazy caching ---
    @property
    def reynolds_numbers(self) -> tuple[float, ...]:
        """A tuple of Reynolds numbers, one for each Airplane in the SteadyProblem.

        **Notes:**

        The Reynolds number is calculated as: Re = (V x L) / nu, where V is the
        freestream speed, observed from the Earth frame (vCg__E from OperatingPoint,
        m/s), L is the characteristic length (c_ref from Airplane, m), and nu is the
        kinematic viscosity (nu from OperatingPoint, m^2/s).

        These Reynolds numbers only consider the freestream speed, not any apparent
        velocity due to prescribed motion, so be careful interpreting it for cases where
        this SteadyProblem corresponds to one time step in an UnsteadyProblem.

        :return: A tuple of Reynolds numbers, one for each Airplane.
        """
        if self._reynolds_numbers is None:
            v = self._operating_point.vCg__E
            nu = self._operating_point.nu

            reynolds_list = []
            for airplane in self._airplanes:
                c_ref = airplane.c_ref
                assert c_ref is not None, "Airplane c_ref must be set to calculate Re"
                re = (v * c_ref) / nu
                reynolds_list.append(re)

            # Store as tuple to prevent external mutation.
            self._reynolds_numbers = tuple(reynolds_list)
        return self._reynolds_numbers


class UnsteadyProblem(_core.CoreUnsteadyProblem):
    """A class used to contain unsteady aerodynamics problems.

    **Contains the following methods:**

    only_final_results: Determines whether the solver will only calculate loads for the
    final time step or final cycle.

    num_steps: The number of time steps.

    delta_time: The time step size in seconds.

    first_averaging_step: The first time step included in cycle averaging.

    first_results_step: The first time step for which loads are calculated.

    max_wake_rows: The maximum chordwise wake rows per Wing.

    movement: The Movement that contains this UnsteadyProblem's OperatingPointMovement
    and AirplaneMovements.

    steady_problems: A tuple of SteadyProblems, one for each time step.
    """

    __slots__ = (
        "_movement",
        "_steady_problems",
    )

    def __init__(
        self,
        movement: movements.movement.Movement,
        only_final_results: bool | np.bool_ = False,
    ) -> None:
        """The initialization method.

        :param movement: The Movement that contains this UnsteadyProblem's
            OperatingPointMovement and AirplaneMovements.
        :param only_final_results: Determines whether the Solver will only calculate
            loads for the final time step (for static Movements) or (for non static
            Movements) for will only calculate loads for the time steps in the final
            complete motion cycle (of the Movement's sub Movement with the longest
            period), which increases simulation speed. Can be a bool or a numpy bool and
            will be converted internally to a bool. The default is False.
        :return: None
        """
        # Validate and store the Movement before calling super().__init__() because
        # the Movement provides the parameters that the core class needs.
        if not isinstance(movement, movements.movement.Movement):
            raise TypeError("movement must be a Movement.")
        self._movement = movement

        # Delegate shared initialization (validation, first_averaging_step computation,
        # load list initialization) to the core class.
        super().__init__(
            only_final_results=only_final_results,
            delta_time=self._movement.delta_time,
            num_steps=self._movement.num_steps,
            max_wake_rows=self._movement.max_wake_rows,
            lcm_period=self._movement.lcm_period,
        )

        # Initialize an empty list to hold the SteadyProblems as they are generated.
        steady_problems_temp: list[SteadyProblem] = []

        # Iterate through the UnsteadyProblem's time steps.
        for step_id in range(self._num_steps):

            # Get the Airplanes and the OperatingPoint associated with this time step.
            these_airplanes = []
            for this_base_airplane in movement.airplanes:
                these_airplanes.append(this_base_airplane[step_id])
            this_operating_point = movement.operating_points[step_id]

            # Initialize the SteadyProblem at this time step.
            this_steady_problem = SteadyProblem(
                airplanes=these_airplanes, operating_point=this_operating_point
            )

            # Append this SteadyProblem to the temporary list.
            steady_problems_temp.append(this_steady_problem)

        # Store as tuple to prevent external mutation via .append(), .pop(), etc.
        self._steady_problems: tuple[SteadyProblem, ...] = tuple(steady_problems_temp)

    # --- Immutable: read only properties ---
    @property
    def movement(self) -> movements.movement.Movement:
        return self._movement

    @property
    def steady_problems(self) -> tuple[SteadyProblem, ...]:
        return self._steady_problems


class _CoupledUnsteadyProblem(_core.CoreUnsteadyProblem):
    """A class for coupled unsteady aerodynamics problems.

    This class extends CoreUnsteadyProblem to manage SteadyProblems for coupled
    simulations where the geometry at each time step depends on the solver's results
    from previous time steps.

    **Contains the following methods:**

    movement: The CoreMovement that defines the motion parameters for this problem.

    steady_problems: A tuple of SteadyProblems, one for each time step that has been
    initialized so far.

    get_steady_problem: Gets the SteadyProblem at a specified time step.

    initialize_next_problem: Initializes the next time step's SteadyProblem. Must be
    overridden by subclasses.
    """

    __slots__ = (
        "_movement",
        "_steady_problems",
    )

    def __init__(
        self,
        movement: _core.CoreMovement,
        initial_airplanes: list[geometry.airplane.Airplane],
        initial_operating_point: operating_point_mod.OperatingPoint,
    ) -> None:
        """The initialization method.

        Initializes the coupled unsteady problem with the first time step's geometry and
        the motion parameters from the provided CoreMovement.

        :param movement: A CoreMovement object that defines the motion parameters
            (delta_time, num_steps, max_wake_rows, lcm_period) for this problem.
        :param initial_airplanes: The list of Airplanes at the first time step.
        :param initial_operating_point: The OperatingPoint at the first time step.
        :return: None
        """
        self._movement = movement

        # Delegate shared initialization (validation, first_averaging_step computation,
        # load list initialization) to the core class. _CoupledUnsteadyProblems require
        # per step results to feed the coupling hook, so only_final_results is always
        # False.
        super().__init__(
            only_final_results=False,
            delta_time=self._movement.delta_time,
            num_steps=self._movement.num_steps,
            max_wake_rows=self._movement.max_wake_rows,
            lcm_period=self._movement.lcm_period,
        )

        # Coupled-specific state: a mutable list of SteadyProblems that grows as the
        # solver advances. Subclass initialize_next_problem overrides append to this
        # list; external code reads through the steady_problems tuple-view property to
        # preserve the read-only contract inherited from UnsteadyProblem. Seed with a
        # SteadyProblem built from the initial geometry so step zero is always ready.
        self._steady_problems: list[SteadyProblem] = [
            SteadyProblem(
                airplanes=initial_airplanes,
                operating_point=initial_operating_point,
            )
        ]

    # --- Immutable: read only properties ---
    @property
    def movement(self) -> _core.CoreMovement:
        return self._movement

    @property
    def steady_problems(self) -> tuple[SteadyProblem, ...]:
        return tuple(self._steady_problems)

    def get_steady_problem(self, step: int) -> SteadyProblem:
        """Get the SteadyProblem at a given time step.

        :param step: The time step index (zero indexed). Must be greater than or equal
            to zero and less than the total number of time steps.
        :return: The SteadyProblem at the specified time step.
        """
        step = _parameter_validation.int_in_range_return_int(
            step, "step", 0, True, len(self._steady_problems), False
        )

        return self._steady_problems[step]

    def initialize_next_problem(
        self, solver: CoupledUnsteadyRingVortexLatticeMethodSolver
    ) -> None:
        """Initialize the next time step's SteadyProblem.

        Must be overridden by subclasses to compute the geometry for the next time step
        based on the solver's results.

        :param solver: The CoupledUnsteadyRingVortexLatticeMethodSolver instance
            providing aerodynamic data from the current time step.
        :return: None
        """
        raise NotImplementedError("Subclasses must implement initialize_next_problem.")


class FreeFlightUnsteadyProblem(_CoupledUnsteadyProblem):
    """A class used to contain problems with coupled unsteady aerodynamics and rigid
    body dynamics.

    **Contains the following methods:**

    only_final_results: Determines whether the solver will only calculate loads for the
    final time step or final cycle.

    num_steps: The number of time steps.

    delta_time: The time step size in seconds.

    first_averaging_step: The first time step included in cycle averaging.

    first_results_step: The first time step for which loads are calculated.

    max_wake_rows: The maximum chordwise wake rows per Wing.

    movement: The FreeFlightMovement that defines the motion parameters for this
    FreeFlightUnsteadyProblem.

    steady_problems: A tuple of SteadyProblems, one for each time step that has been
    initialized so far.

    get_steady_problem: Gets the SteadyProblem at a specified time step.

    initialize_next_problem: Initializes the next time step's SteadyProblem from rigid
    body dynamics.

    I_BP1_CgP1: The inertia matrix of the Airplane (in the first Airplane's body axes,
    relative to the first Airplane's CG) in kilogram square meters.

    external_forces_fn: A callable that computes additional forces and moments to apply
    to the Airplane during the simulation, or None.

    mujoco_model: The MuJoCoModel used for rigid body dynamics integration.
    """

    __slots__ = (
        "_I_BP1_CgP1",
        "_external_forces_fn",
        "_mujoco_model",
        "forces_W",
        "forceCoefficients_W",
        "moments_W_Cg",
        "momentCoefficients_W_Cg",
    )

    def __init__(
        self,
        movement: movements.free_flight_movement.FreeFlightMovement,
        initial_airplanes: list[geometry.airplane.Airplane],
        initial_operating_point: operating_point_mod.OperatingPoint,
        I_BP1_CgP1: np.ndarray | Sequence[Sequence[float | int]],
        external_forces_fn: (
            Callable[
                [
                    operating_point_mod.OperatingPoint,
                    geometry.airplane.Airplane,
                ],
                tuple[np.ndarray, np.ndarray],
            ]
            | None
        ) = None,
        extra_xml: dict[str, str] | None = None,
        mujoco_assets: dict[str, bytes] | None = None,
    ) -> None:
        """The initialization method.

        :param movement: The FreeFlightMovement that defines the prescribed wing
            geometry and operating point for this FreeFlightUnsteadyProblem.
        :param initial_airplanes: A list containing exactly one Airplane representing
            the initial geometry at the first time step. Multi-airplane free flight is
            not supported in this release.
        :param initial_operating_point: The OperatingPoint at the first time step,
            defining the initial freestream conditions, body orientation, angular
            velocity, and gravity vector.
        :param I_BP1_CgP1: An array-like object of numbers (int or float) with shape
            (3,3) representing the inertia matrix of the Airplane (in the first
            Airplane's body axes, relative to the first Airplane's CG). It must be
            symmetric. Can be a tuple, list, or ndarray. Values are converted to floats
            internally. The units are in kilogram square meters.
        :param external_forces_fn: A callable that computes additional forces and
            moments to apply to the Airplane during the simulation. It takes an
            OperatingPoint and an Airplane and returns a tuple of two (3,) ndarrays of
            floats: the additional force (in wind axes, in Newtons) and the additional
            moment (in wind axes, relative to the first Airplane's CG, in Newton
            meters). Setting this to None applies no additional forces. The default is
            None.
        :param extra_xml: A dict mapping injection point names to XML fragment strings
            to inject into the MuJoCo model's XML. Supported keys are "default",
            "asset", "visual", "worldbody", and "body". Setting this to None injects no
            extra XML. The default is None.
        :param mujoco_assets: A dict mapping virtual filenames to their binary contents
            for the MuJoCo model. Setting this to None provides no extra assets. The
            default is None.
        :return: None
        """
        if len(initial_airplanes) != 1:
            raise ValueError(
                "initial_airplanes must have exactly one element. "
                "Multi-airplane free flight is not supported in this release."
            )

        super().__init__(
            movement=movement,
            initial_airplanes=initial_airplanes,
            initial_operating_point=initial_operating_point,
        )

        I_BP1_CgP1 = _parameter_validation.m_by_n_number_arrayLike_return_float(
            I_BP1_CgP1, "I_BP1_CgP1", 3, 3
        )
        if not np.allclose(I_BP1_CgP1, I_BP1_CgP1.T):
            raise ValueError("I_BP1_CgP1 must be symmetric.")
        self._I_BP1_CgP1 = I_BP1_CgP1
        self._I_BP1_CgP1.flags.writeable = False

        if external_forces_fn is not None and not callable(external_forces_fn):
            raise TypeError("external_forces_fn must be callable or None.")
        self._external_forces_fn = external_forces_fn

        # Initialize empty lists to hold the loads and load coefficients experienced by
        # each time step's Airplane.
        self.forces_W: list[np.ndarray] = []
        self.forceCoefficients_W: list[np.ndarray] = []
        self.moments_W_Cg: list[np.ndarray] = []
        self.momentCoefficients_W_Cg: list[np.ndarray] = []

        self._mujoco_model = _mujoco_model.MuJoCoModel(
            name=initial_airplanes[0].name,
            weight=initial_airplanes[0].weight,
            omegas_BP1__E=initial_operating_point.omegas_BP1__E,
            g_E=initial_operating_point.g_E,
            T_pas_BP1_CgP1_to_E_CgP1=initial_operating_point.T_pas_BP1_CgP1_to_E_CgP1,
            vCg_E__E=-1
            * _transformations.apply_T_to_vectors(
                initial_operating_point.T_pas_GP1_CgP1_to_E_CgP1,
                initial_operating_point.vInf_GP1__E,
                has_point=False,
            ),
            I_BP1_CgP1=self._I_BP1_CgP1,
            delta_time=movement.delta_time,
            extra_xml=extra_xml,
            mujoco_assets=mujoco_assets,
        )

    # --- Immutable: read only properties ---
    @property
    def I_BP1_CgP1(self) -> np.ndarray:
        return self._I_BP1_CgP1

    @property
    def external_forces_fn(
        self,
    ) -> (
        Callable[
            [
                operating_point_mod.OperatingPoint,
                geometry.airplane.Airplane,
            ],
            tuple[np.ndarray, np.ndarray],
        ]
        | None
    ):
        return self._external_forces_fn

    @property
    def mujoco_model(self) -> _mujoco_model.MuJoCoModel:
        return self._mujoco_model

    def initialize_next_problem(
        self, solver: CoupledUnsteadyRingVortexLatticeMethodSolver
    ) -> None:
        """Initializes the next time step's SteadyProblem from rigid body dynamics.

        Transforms aerodynamic loads into Earth axes, applies them (along with weight
        and any external forces) to the MuJoCo model, steps the dynamics forward,
        extracts the new state, and creates the next SteadyProblem with the updated
        OperatingPoint and the prescribed Airplane geometry for the next step.

        :param solver: The CoupledUnsteadyRingVortexLatticeMethodSolver instance
            providing aerodynamic data from the current time step.
        :return: None
        """
        # 1. Get aerodynamic loads from the current Airplane.

        # 2. Add external forces if external_forces_fn is set.

        # 3. Transform loads from wind axes to Earth axes.

        # 4. Add the weight force in Earth axes.

        # 5. Apply loads to MuJoCo and step the dynamics forward.

        # 6. Extract the new state from MuJoCo.

        # 7. Derive alpha, beta, and Euler angles from the new state.

        # 8. Create the new OperatingPoint for the next time step.

        # 9. Get the next Airplane from the movement's pregenerated airplanes.

        # 10. Create the next SteadyProblem and append to _steady_problems.

        # 11. Store load history.

        raise NotImplementedError
