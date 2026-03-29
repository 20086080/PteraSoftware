"""Contains the OperatingPointMovement class.

**Contains the following classes:**

OperatingPointMovement: A class used to contain an OperatingPoint's movements.

**Contains the following functions:**

None
"""

from __future__ import annotations

from collections.abc import Callable

from .. import _core, _parameter_validation
from .. import operating_point as operating_point_mod


class OperatingPointMovement(_core.CoreOperatingPointMovement):
    """A class used to contain an OperatingPoint's movements.

    **Contains the following methods:**

    max_period: OperatingPointMovement's longest period of motion.

    generate_operating_point_at_time_step: Creates the OperatingPoint at a single time
    step.

    generate_operating_points: Creates the OperatingPoint at each time step, and returns
    them in a list.
    """

    __slots__ = ()

    def __init__(
        self,
        base_operating_point: operating_point_mod.OperatingPoint,
        ampVCg__E: float | int = 0.0,
        periodVCg__E: float | int = 0.0,
        spacingVCg__E: str | Callable[[float], float] = "sine",
        phaseVCg__E: float | int = 0.0,
    ) -> None:
        """The initialization method.

        :param base_operating_point: The base OperatingPoint from which the
            OperatingPoint at each time step will be created.
        :param ampVCg__E: The amplitude of the OperatingPointMovement's changes in its
            OperatingPoints' vCg__E parameters. Must be a non negative number (int or
            float), and is converted to a float internally. The amplitude must be low
            enough that it doesn't drive its base value out of the range of valid
            values. Otherwise, this OperatingPointMovement will try to create
            OperatingPoints with invalid parameter values. The units are in meters per
            second. The default is 0.0.
        :param periodVCg__E: The period of the OperatingPointMovement's changes in its
            OperatingPoints' vCg__E parameter. Must be a non negative number (int or
            float), and is converted to a float internally. It must be 0.0 if ampVCg__E
            is 0.0 and non zero if not. The units are in seconds. The default is 0.0.
        :param spacingVCg__E: Determines the spacing of the OperatingPointMovement's
            change in its OperatingPoints' vCg__E parameters. Can be "sine", "uniform",
            or a callable custom spacing function. Custom spacing functions are for
            advanced users and must start at 0.0, return to 0.0 after one period of 2.0
            * pi radians, have amplitude of 1.0, be periodic, return finite values only,
            and accept a float as input and return a float. The custom function is
            scaled by ampVCg__E, shifted horizontally and vertically by phaseVCg__E and
            the base value, and have a period set by periodVCg__E. The default is
            "sine".
        :param phaseVCg__E: The phase offset of the first time step's OperatingPoint's
            vCg__E parameter relative to the base OperatingPoint's vCg__E parameter.
            Must be a number (int or float) in the range (-180.0, 180.0], and will be
            converted to a float internally. It must be 0.0 if ampVCg__E is 0.0 and non
            zero if not. The units are in degrees. The default is 0.0.
        :return: None
        """
        super().__init__(
            base_operating_point=base_operating_point,
            ampVCg__E=ampVCg__E,
            periodVCg__E=periodVCg__E,
            spacingVCg__E=spacingVCg__E,
            phaseVCg__E=phaseVCg__E,
        )

    # --- Other methods ---
    def generate_operating_points(
        self, num_steps: int, delta_time: float | int
    ) -> list[operating_point_mod.OperatingPoint]:
        """Creates the OperatingPoint at each time step, and returns them in a list.

        :param num_steps: The number of time steps in this movement. It must be a
            positive int.
        :param delta_time: The time between each time step. It must be a positive number
            (int or float), and will be converted internally to a float. The units are
            in seconds.
        :return: The list of OperatingPoints associated with this
            OperatingPointMovement.
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

        operating_points = []
        for step in range(num_steps):
            operating_points.append(
                self.generate_operating_point_at_time_step(step, delta_time)
            )

        return operating_points
