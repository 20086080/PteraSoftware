"""Contains the AirplaneMovement class.

**Contains the following classes:**

AirplaneMovement: A class used to contain an Airplane's movement.

**Contains the following functions:**

None
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import cast

import numpy as np

from .. import _core, _parameter_validation, geometry
from . import _functions
from . import wing_movement as wing_movement_mod


class AirplaneMovement(_core.CoreAirplaneMovement):
    """A class used to contain an Airplane's movement.

    **Contains the following methods:**

    __deepcopy__: Creates a deep copy of this AirplaneMovement.

    all_periods: All unique non zero periods from this AirplaneMovement, its
    WingMovement(s), and their WingCrossSectionMovements.

    max_period: The longest period of AirplaneMovement's own motion, the motion(s) of
    its sub movement object(s), and the motions of its sub sub  movement objects.

    generate_airplane_at_time_step: Creates the Airplane at a single time step.

    generate_airplanes: Creates the Airplane at each time step, and returns them in a
    list.
    """

    __slots__ = ()

    def __init__(
        self,
        base_airplane: geometry.airplane.Airplane,
        wing_movements: list[wing_movement_mod.WingMovement],
        ampCg_GP1_CgP1: np.ndarray | Sequence[float | int] = (0.0, 0.0, 0.0),
        periodCg_GP1_CgP1: np.ndarray | Sequence[float | int] = (0.0, 0.0, 0.0),
        spacingCg_GP1_CgP1: np.ndarray | Sequence[str | Callable[[float], float]] = (
            "sine",
            "sine",
            "sine",
        ),
        phaseCg_GP1_CgP1: np.ndarray | Sequence[float | int] = (0.0, 0.0, 0.0),
    ) -> None:
        """The initialization method.

        :param base_airplane: The base Airplane from which the Airplane at each time
            step will be created.
        :param wing_movements: A list of the WingMovements associated with each of the
            base Airplane's Wings. It must have the same length as the base Airplane's
            list of Wings.
        :param ampCg_GP1_CgP1: An array-like object of non negative numbers (int or
            float) with shape (3,) representing the amplitudes of the AirplaneMovement's
            changes in its Airplanes' Cg_GP1_CgP1 parameters. Can be a tuple, list, or
            ndarray. Values are converted to floats internally. Each amplitude must be
            low enough that it doesn't drive its base value out of the range of valid
            values. Otherwise, this AirplaneMovement will try to create Airplanes with
            invalid parameter values. Because the first Airplane's Cg_GP1_CgP1 parameter
            must be all zeros, this means that the first Airplane's ampCg_GP1_CgP1
            parameter must also be all zeros. The units are in meters. The default is
            (0.0, 0.0, 0.0).
        :param periodCg_GP1_CgP1: An array-like object of non negative numbers (int or
            float) with shape (3,) representing the periods of the AirplaneMovement's
            changes in its Airplanes' Cg_GP1_CgP1 parameters. Can be a tuple, list, or
            ndarray. Values are converted to floats internally. Each element must be 0.0
            if the corresponding element in ampCg_GP1_CgP1 is 0.0 and non zero if not.
            The units are in seconds. The default is (0.0, 0.0, 0.0).
        :param spacingCg_GP1_CgP1: An array-like object of strs or callables with shape
            (3,) representing the spacing of the AirplaneMovement's changes in its
            Airplanes' Cg_GP1_CgP1 parameters. Can be a tuple, list, or ndarray. Each
            element can be the str "sine", the str "uniform", or a callable custom
            spacing function. Custom spacing functions are for advanced users and must
            start at 0.0, return to 0.0 after one period of 2.0 * pi radians, have
            amplitude of 1.0, be periodic, return finite values only, and accept a float
            as input and return a float. Custom functions are scaled by ampCg_GP1_CgP1,
            shifted horizontally and vertically by phaseCg_GP1_CgP1 and the base value,
            and have a period set by periodCg_GP1_CgP1. The default is ("sine", "sine",
            "sine").
        :param phaseCg_GP1_CgP1: An array-like object of numbers (int or float) with
            shape (3,) representing the phase offsets of the elements in the first time
            step's Airplane's Cg_GP1_CgP1 parameter relative to the base Airplane's
            Cg_GP1_CgP1 parameter. Can be a tuple, list, or ndarray. Elements must lie
            in the range (-180.0, 180.0]. Each element must be 0.0 if the corresponding
            element in ampCg_GP1_CgP1 is 0.0 and non zero if not. Values are converted
            to floats internally. The units are in degrees. The default is (0.0, 0.0,
            0.0).
        :return: None
        """
        super().__init__(
            base_airplane=base_airplane,
            wing_movements=wing_movements,
            ampCg_GP1_CgP1=ampCg_GP1_CgP1,
            periodCg_GP1_CgP1=periodCg_GP1_CgP1,
            spacingCg_GP1_CgP1=spacingCg_GP1_CgP1,
            phaseCg_GP1_CgP1=phaseCg_GP1_CgP1,
        )

    # --- Immutable: read only properties ---
    @property
    def wing_movements(self) -> tuple[wing_movement_mod.WingMovement, ...]:
        return cast(
            tuple[wing_movement_mod.WingMovement, ...],
            self._wing_movements,
        )

    # --- Other methods ---
    def generate_airplanes(
        self, num_steps: int, delta_time: float | int
    ) -> list[geometry.airplane.Airplane]:
        """Creates the Airplane at each time step, and returns them in a list.

        For static geometry (no periodic motion), this method optimizes performance by
        creating the first Airplane with full mesh generation, then using deepcopy for
        subsequent time steps. This avoids redundant mesh generation when the geometry
        is identical across all steps.

        :param num_steps: The number of time steps in this movement. It must be a
            positive int.
        :param delta_time: The time between each time step. It must be a positive number
            (float or int), and will be converted internally to a float. The units are
            in seconds.
        :return: The list of Airplanes associated with this AirplaneMovement.
        """
        num_steps = _parameter_validation.int_in_range_return_int(
            num_steps,
            "num_steps",
            min_val=1,
            min_inclusive=True,
        )
        delta_time = _parameter_validation.number_in_range_return_float(
            delta_time, "delta_time", min_val=0.0, min_inclusive=False
        )

        # Check if geometry is static (no periodic motion).
        is_static_geometry = self.max_period == 0.0

        if is_static_geometry:
            # Optimization for static geometry: create first Airplane with full mesh
            # generation, then deepcopy for subsequent steps.
            return self._generate_airplanes_static(num_steps, delta_time)
        else:
            # For variable geometry, use the standard approach.
            return self._generate_airplanes_variable(num_steps, delta_time)

    def _generate_airplanes_static(
        self, num_steps: int, delta_time: float
    ) -> list[geometry.airplane.Airplane]:
        """Generates Airplanes for static geometry using deepcopy optimization.

        Creates the first Airplane with full mesh generation, then uses deepcopy for
        subsequent time steps to avoid redundant mesh generation.

        :param num_steps: The number of time steps.
        :param delta_time: The time between each time step in seconds.
        :return: The list of Airplanes.
        """
        # Create the first Airplane (triggers full mesh generation).
        first_airplane = self.generate_airplane_at_time_step(0, delta_time)

        # Create list with first Airplane.
        airplanes = [first_airplane]

        # Create copies for remaining steps with different Cg_GP1_CgP1 positions.
        for step in range(1, num_steps):
            time = step * delta_time

            # Evaluate Cg_GP1_CgP1 at this step.
            thisCg_GP1_CgP1 = np.zeros(3, dtype=float)
            for dim in range(3):
                this_spacing = self._spacingCg_GP1_CgP1[dim]
                this_amp = self._ampCg_GP1_CgP1[dim]
                this_period = self._periodCg_GP1_CgP1[dim]
                this_phase = self._phaseCg_GP1_CgP1[dim]
                this_base = self._base_airplane.Cg_GP1_CgP1[dim]

                if this_spacing == "sine":
                    thisCg_GP1_CgP1[dim] = _functions.oscillating_sin_at_time(
                        amp=this_amp,
                        period=this_period,
                        phase=this_phase,
                        base=this_base,
                        time=time,
                    )
                elif this_spacing == "uniform":
                    thisCg_GP1_CgP1[dim] = _functions.oscillating_lin_at_time(
                        amp=this_amp,
                        period=this_period,
                        phase=this_phase,
                        base=this_base,
                        time=time,
                    )
                elif callable(this_spacing):
                    thisCg_GP1_CgP1[dim] = _functions.oscillating_custom_at_time(
                        amp=this_amp,
                        period=this_period,
                        phase=this_phase,
                        base=this_base,
                        time=time,
                        custom_function=this_spacing,
                    )
                else:
                    raise ValueError(f"Invalid spacing value: {this_spacing}")

            copied_airplane = first_airplane.deep_copy_with_Cg_GP1_CgP1(thisCg_GP1_CgP1)
            airplanes.append(copied_airplane)

        return airplanes

    def _generate_airplanes_variable(
        self, num_steps: int, delta_time: float
    ) -> list[geometry.airplane.Airplane]:
        """Generates Airplanes for variable (periodic) geometry.

        Uses a conservative optimization approach that validates periodicity before
        applying deepcopy optimization. If validation fails, falls back to standard
        generation.

        :param num_steps: The number of time steps.
        :param delta_time: The time between each time step in seconds.
        :return: The list of Airplanes.
        """
        # Step 1: Calculate geometry LCM period.
        geometry_lcm = self._geometry_lcm_period()

        # Step 2: Pre-validation checks.
        # Check if period aligns cleanly with delta_time.
        steps_per_period_float = geometry_lcm / delta_time
        steps_per_period = int(round(steps_per_period_float))

        alignment_error = abs(steps_per_period_float - steps_per_period)
        if alignment_error > 1e-6:
            # Period doesn't align with time steps. Fall back to standard generation.
            return self._generate_airplanes_variable_standard(num_steps, delta_time)

        # Check if there's meaningful benefit.
        if num_steps <= steps_per_period:
            # No repetition occurs. No benefit from optimization.
            return self._generate_airplanes_variable_standard(num_steps, delta_time)

        # Step 3: Generate first period + one validation step.
        validation_num_steps = steps_per_period + 1

        # Create an empty 2D ndarray to hold Wings for validation steps.
        wings = np.empty((len(self.wing_movements), validation_num_steps), dtype=object)

        # Iterate through the WingMovements.
        for wing_movement_id, wing_movement in enumerate(self.wing_movements):
            this_wings_list_of_wings = np.array(
                wing_movement.generate_wings(
                    num_steps=validation_num_steps, delta_time=delta_time
                )
            )
            wings[wing_movement_id, :] = this_wings_list_of_wings

        # Step 4: Validate periodicity.
        # Compare step 0 geometry to step steps_per_period.
        if not self._geometry_matches(
            wings_step_a=wings[:, 0],
            wings_step_b=wings[:, steps_per_period],
            tolerance=1e-9,
        ):
            # Geometry doesn't repeat as expected. Fall back to standard generation.
            return self._generate_airplanes_variable_standard(num_steps, delta_time)

        # Step 5: Create Airplanes for first period.
        this_name = self._base_airplane.name
        this_weight = self._base_airplane.weight

        first_period_airplanes = []
        for step in range(steps_per_period):
            this_airplane = self.generate_airplane_at_time_step(step, delta_time)
            first_period_airplanes.append(this_airplane)

        # Step 6: Create copies for remaining steps with different Cg_GP1_CgP1.
        airplanes = list(first_period_airplanes)
        for step in range(steps_per_period, num_steps):
            source_step = step % steps_per_period

            # Evaluate Cg_GP1_CgP1 at this step.
            time = step * delta_time
            thisCg_GP1_CgP1 = np.zeros(3, dtype=float)
            for dim in range(3):
                this_spacing = self._spacingCg_GP1_CgP1[dim]
                this_amp = self._ampCg_GP1_CgP1[dim]
                this_period = self._periodCg_GP1_CgP1[dim]
                this_phase = self._phaseCg_GP1_CgP1[dim]
                this_base = self._base_airplane.Cg_GP1_CgP1[dim]

                if this_spacing == "sine":
                    thisCg_GP1_CgP1[dim] = _functions.oscillating_sin_at_time(
                        amp=this_amp,
                        period=this_period,
                        phase=this_phase,
                        base=this_base,
                        time=time,
                    )
                elif this_spacing == "uniform":
                    thisCg_GP1_CgP1[dim] = _functions.oscillating_lin_at_time(
                        amp=this_amp,
                        period=this_period,
                        phase=this_phase,
                        base=this_base,
                        time=time,
                    )
                elif callable(this_spacing):
                    thisCg_GP1_CgP1[dim] = _functions.oscillating_custom_at_time(
                        amp=this_amp,
                        period=this_period,
                        phase=this_phase,
                        base=this_base,
                        time=time,
                        custom_function=this_spacing,
                    )
                else:
                    raise ValueError(f"Invalid spacing value: {this_spacing}")

            copied_airplane = first_period_airplanes[
                source_step
            ].deep_copy_with_Cg_GP1_CgP1(thisCg_GP1_CgP1)
            airplanes.append(copied_airplane)

        return airplanes

    def _generate_airplanes_variable_standard(
        self, num_steps: int, delta_time: float
    ) -> list[geometry.airplane.Airplane]:
        """Generates Airplanes for variable geometry using standard approach.

        Creates new Airplanes for each time step without optimization.

        :param num_steps: The number of time steps.
        :param delta_time: The time between each time step in seconds.
        :return: The list of Airplanes.
        """
        airplanes = []
        for step in range(num_steps):
            airplanes.append(self.generate_airplane_at_time_step(step, delta_time))

        return airplanes

    @staticmethod
    def _lcm(a: float, b: float) -> float:
        """Calculates the least common multiple of two numbers.

        :param a: First number (period in seconds).
        :param b: Second number (period in seconds).
        :return: LCM of a and b. Returns 0.0 if either input is 0.0.
        """
        if a == 0.0 or b == 0.0:
            return 0.0
        # Convert to integers (periods are typically whole multiples of delta_time).
        # Use sufficiently large multiplier to preserve precision.
        multiplier = 1000000
        a_int = int(round(a * multiplier))
        b_int = int(round(b * multiplier))
        lcm_int = abs(a_int * b_int) // math.gcd(a_int, b_int)
        return lcm_int / multiplier

    @staticmethod
    def _lcm_multiple(periods: list[float] | tuple[float, ...]) -> float:
        """Calculates the least common multiple of multiple periods.

        :param periods: A list or tuple of periods in seconds.
        :return: LCM of all periods. Returns 0.0 if all periods are 0.0.
        """
        if not periods or all(p == 0.0 for p in periods):
            return 0.0

        # Filter out zero periods and calculate LCM.
        non_zero_periods = [p for p in periods if p != 0.0]

        result = non_zero_periods[0]
        for period in non_zero_periods[1:]:
            result = AirplaneMovement._lcm(result, period)
        return result

    def _geometry_lcm_period(self) -> float:
        """Calculates the LCM of all geometry related periods.

        Excludes OperatingPoint periods since those don't affect geometry. Uses the
        all_periods property which already collects only geometry related periods.

        :return: The LCM of all geometry related periods in seconds. Returns 0.0 if all
            geometry is static.
        """
        return self._lcm_multiple(self.all_periods)

    @staticmethod
    def _geometry_matches(
        wings_step_a: np.ndarray,
        wings_step_b: np.ndarray,
        tolerance: float = 1e-9,
    ) -> bool:
        """Compares two sets of Wings to verify their geometry matches within tolerance.

        Checks Wing position (Ler_Gs_Cgs), Wing angles (angles_Gs_to_Wn_ixyz), and Panel
        corner positions (Frpp_G_Cg, Flpp_G_Cg, Blpp_G_Cg, Brpp_G_Cg).

        :param wings_step_a: A (num_wings,) ndarray of Wings from the first time step.
        :param wings_step_b: A (num_wings,) ndarray of Wings from the second time step.
        :param tolerance: The tolerance for floating point comparison. The default is
            1e-9.
        :return: True if all geometry attributes match within tolerance, False
            otherwise.
        """
        if len(wings_step_a) != len(wings_step_b):
            return False

        for wing_a, wing_b in zip(wings_step_a, wings_step_b):
            # Check Wing position.
            if not np.allclose(
                wing_a.Ler_Gs_Cgs, wing_b.Ler_Gs_Cgs, atol=tolerance, rtol=0.0
            ):
                return False

            # Check Wing angles.
            if not np.allclose(
                wing_a.angles_Gs_to_Wn_ixyz,
                wing_b.angles_Gs_to_Wn_ixyz,
                atol=tolerance,
                rtol=0.0,
            ):
                return False

            # Check Panel corner positions if Wings are meshed.
            if wing_a.panels is not None and wing_b.panels is not None:
                if wing_a.panels.shape != wing_b.panels.shape:
                    return False

                for i in range(wing_a.panels.shape[0]):
                    for j in range(wing_a.panels.shape[1]):
                        panel_a = wing_a.panels[i, j]
                        panel_b = wing_b.panels[i, j]

                        # Check all four corner positions.
                        if not np.allclose(
                            panel_a.Frpp_G_Cg,
                            panel_b.Frpp_G_Cg,
                            atol=tolerance,
                            rtol=0.0,
                        ):
                            return False
                        if not np.allclose(
                            panel_a.Flpp_G_Cg,
                            panel_b.Flpp_G_Cg,
                            atol=tolerance,
                            rtol=0.0,
                        ):
                            return False
                        if not np.allclose(
                            panel_a.Blpp_G_Cg,
                            panel_b.Blpp_G_Cg,
                            atol=tolerance,
                            rtol=0.0,
                        ):
                            return False
                        if not np.allclose(
                            panel_a.Brpp_G_Cg,
                            panel_b.Brpp_G_Cg,
                            atol=tolerance,
                            rtol=0.0,
                        ):
                            return False

        return True
