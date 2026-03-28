"""Contains the core classes for the movement and problem hierarchies."""

from __future__ import annotations

import math

from . import _parameter_validation
from .movements import airplane_movement as airplane_movement_mod
from .movements import operating_point_movement as operating_point_movement_mod


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


def _lcm_multiple(periods: list[float]) -> float:
    """Calculates the least common multiple of multiple periods.

    :param periods: A list of periods in seconds.
    :return: LCM of all periods. Returns 0.0 if all periods are 0.0.
    """
    if not periods or all(p == 0.0 for p in periods):
        return 0.0
    non_zero_periods = [p for p in periods if p != 0.0]
    if not non_zero_periods:
        return 0.0
    result = non_zero_periods[0]
    for period in non_zero_periods[1:]:
        result = _lcm(result, period)
    return result


class CoreMovement:
    """A core class used to contain the shared foundation of Movement and its feature
    variant siblings.

    CoreMovement holds the fundamental parameters (AirplaneMovements,
    OperatingPointMovement, delta_time, num_steps, and max_wake_rows) and provides the
    shared derived properties that all Movement variants need. Unlike Movement,
    CoreMovement requires delta_time and num_steps to be provided directly and does not
    perform automatic estimation or calculation.

    **Contains the following methods:**

    lcm_period: The least common multiple of all motion periods, ensuring all motions
    complete an integer number of cycles when cycle averaging forces and moments.

    max_period: The longest period of motion of CoreMovement's sub movement objects, the
    motion(s) of its sub sub movement object(s), and the motions of its sub sub sub
    movement objects.

    min_period: The shortest non zero period of motion of CoreMovement's sub movement
    objects, the motion(s) of its sub sub movement object(s), and the motions of its sub
    sub sub movement objects.

    static: Flags if CoreMovement's sub movement objects, its sub sub movement
    object(s), and its sub sub sub movement objects all represent no motion.
    """

    __slots__ = (
        "_airplane_movements",
        "_operating_point_movement",
        "_delta_time",
        "_num_steps",
        "_max_wake_rows",
        "_lcm_period",
        "_max_period",
        "_min_period",
        "_static",
    )

    def __init__(
        self,
        airplane_movements: list[airplane_movement_mod.AirplaneMovement],
        operating_point_movement: operating_point_movement_mod.OperatingPointMovement,
        delta_time: float | int,
        num_steps: int,
        max_wake_rows: int | None = None,
    ) -> None:
        """The initialization method.

        :param airplane_movements: A list of the AirplaneMovements associated with each
            of the UnsteadyProblem's Airplanes.
        :param operating_point_movement: An OperatingPointMovement characterizing any
            changes to the UnsteadyProblem's operating conditions.
        :param delta_time: The time between each time step. It must be a positive number
            (int or float). Values are converted to floats internally. The units are in
            seconds.
        :param num_steps: The number of time steps of the unsteady simulation. It must
            be a positive int.
        :param max_wake_rows: The maximum number of chordwise wake RingVortex rows per
            Wing. Must be a positive int if set. The default is None (no truncation).
        :return: None
        """
        # Validate and store the AirplaneMovements.
        if not isinstance(airplane_movements, list):
            raise TypeError("airplane_movements must be a list.")
        if len(airplane_movements) < 1:
            raise ValueError("airplane_movements must have at least one element.")
        for airplane_movement in airplane_movements:
            if not isinstance(
                airplane_movement, airplane_movement_mod.AirplaneMovement
            ):
                raise TypeError(
                    "Every element in airplane_movements must be an "
                    "AirplaneMovement."
                )
        # Store as tuple to prevent external mutation.
        self._airplane_movements: tuple[airplane_movement_mod.AirplaneMovement, ...] = (
            tuple(airplane_movements)
        )

        # Validate and store the OperatingPointMovement.
        if not isinstance(
            operating_point_movement,
            operating_point_movement_mod.OperatingPointMovement,
        ):
            raise TypeError(
                "operating_point_movement must be an " "OperatingPointMovement."
            )
        self._operating_point_movement = operating_point_movement

        # Initialize the caches for the properties derived from the immutable
        # attributes. These are initialized early because static is accessed
        # below during __init__ to validate max_wake_rows.
        self._lcm_period: float | None = None
        self._max_period: float | None = None
        self._min_period: float | None = None
        self._static: bool | None = None

        # Validate and store delta_time.
        delta_time = _parameter_validation.number_in_range_return_float(
            delta_time, "delta_time", min_val=0.0, min_inclusive=False
        )
        self._delta_time: float = delta_time

        # Validate and store num_steps.
        num_steps = _parameter_validation.int_in_range_return_int(
            num_steps,
            "num_steps",
            min_val=1,
            min_inclusive=True,
        )
        self._num_steps: int = num_steps

        # Validate and store max_wake_rows.
        if max_wake_rows is not None:
            max_wake_rows = _parameter_validation.int_in_range_return_int(
                max_wake_rows,
                "max_wake_rows",
                min_val=1,
                min_inclusive=True,
            )
        self._max_wake_rows = max_wake_rows

    # --- Immutable: read only properties ---
    @property
    def airplane_movements(
        self,
    ) -> tuple[airplane_movement_mod.AirplaneMovement, ...]:
        return self._airplane_movements

    @property
    def operating_point_movement(
        self,
    ) -> operating_point_movement_mod.OperatingPointMovement:
        return self._operating_point_movement

    @property
    def delta_time(self) -> float:
        return self._delta_time

    @property
    def num_steps(self) -> int:
        return self._num_steps

    @property
    def max_wake_rows(self) -> int | None:
        return self._max_wake_rows

    # --- Immutable derived: manual lazy caching ---
    @property
    def lcm_period(self) -> float:
        """The least common multiple of all motion periods, ensuring all motions
        complete an integer number of cycles when cycle averaging forces and moments.

        Using the LCM ensures that when cycle averaging forces and moments, we capture a
        complete cycle of all motions, not just the longest one. For example, if one
        motion has a period of 2.0 s and another has a period of 3.0 s, the LCM is 6.0,
        which contains exactly 3 cycles of the first motion and 2 cycles of the second.

        :return: The LCM period in seconds. If all the motion is static, this will be
            0.0.
        """
        if self._lcm_period is None:
            # Collect all periods from AirplaneMovements.
            all_periods: list[float] = []
            for airplane_movement in self._airplane_movements:
                all_periods.extend(airplane_movement.all_periods)

            # Add the OperatingPointMovement period.
            all_periods.append(self._operating_point_movement.max_period)

            self._lcm_period = _lcm_multiple(all_periods)
        return self._lcm_period

    @property
    def max_period(self) -> float:
        """The longest period of motion of CoreMovement's sub movement objects, the
        motion(s) of its sub sub movement object(s), and the motions of its sub sub sub
        movement objects.

        For cycle averaging calculations, lcm_period should be used instead of
        max_period to ensure all motions complete an integer number of cycles.

        :return: The longest period in seconds. If all the motion is static, this will
            be 0.0.
        """
        if self._max_period is None:
            # Iterate through the AirplaneMovements and find the one with the
            # largest max period.
            airplane_movement_max_periods = []
            for airplane_movement in self._airplane_movements:
                airplane_movement_max_periods.append(airplane_movement.max_period)
            max_airplane_period = max(airplane_movement_max_periods)

            # The global max period is the maximum of the max
            # AirplaneMovement period and the OperatingPointMovement max
            # period.
            self._max_period = max(
                max_airplane_period,
                self._operating_point_movement.max_period,
            )
        return self._max_period

    @property
    def min_period(self) -> float:
        """The shortest non zero period of motion of CoreMovement's sub movement
        objects, the motion(s) of its sub sub movement object(s), and the motions of its
        sub sub sub movement objects.

        :return: The shortest non zero period in seconds. If all the motion is static,
            this will be 0.0.
        """
        if self._min_period is None:
            # Collect all periods from AirplaneMovements.
            all_periods: list[float] = []
            for airplane_movement in self._airplane_movements:
                all_periods.extend(airplane_movement.all_periods)

            # Add the OperatingPointMovement period.
            op_period = self._operating_point_movement.max_period
            if op_period != 0.0:
                all_periods.append(op_period)

            # Filter out zero periods and find the minimum.
            non_zero_periods = [p for p in all_periods if p != 0.0]
            if not non_zero_periods:
                self._min_period = 0.0
            else:
                self._min_period = min(non_zero_periods)
        return self._min_period

    @property
    def static(self) -> bool:
        """Flags if CoreMovement's sub movement objects, its sub sub movement object(s),
        and its sub sub sub movement objects all represent no motion.

        :return: True if CoreMovement's sub movement objects, its sub sub movement
            object(s), and its sub sub sub movement objects all represent no motion.
            False otherwise.
        """
        if self._static is None:
            self._static = self.max_period == 0
        return self._static
