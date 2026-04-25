"""Contains the MuJoCoModel class."""

from __future__ import annotations

from collections.abc import Sequence

import mujoco
import numpy as np

from . import _parameter_validation, _transformations


class MuJoCoModel:
    """A class used to interface with MuJoCo for free flight simulations.

    **Contains the following methods:**

    apply_loads: Applies loads to the model.

    step: Advances the MuJoCo simulation by one time step.

    get_state: Extracts the current position, orientation, velocity, and angular
    velocity from the model.

    reset: Resets the model's state to the initial conditions, time to zero seconds, and
    removes any applied loads.

    **Notes:**

    Wraps MuJoCo models and data objects to provide a clean interface for applying
    aerodynamic loads to the first Airplane, advancing the MuJoCo simulation, and
    extracting the current state of the first Airplane.
    """

    __slots__ = (
        "_xml_str",
        "_model",
        "_data",
        "_body_id",
        "_initial_key_frame_id",
        "_initial_qpos",
        "_initial_qvel",
    )

    def __init__(
        self,
        name: str,
        weight: float | int,
        omegas_BP1__E: np.ndarray | Sequence[float | int],
        g_E: np.ndarray | Sequence[float | int],
        T_pas_BP1_CgP1_to_E_CgP1: np.ndarray | Sequence[Sequence[float | int]],
        vCg_E__E: np.ndarray | Sequence[float | int],
        I_BP1_CgP1: np.ndarray | Sequence[Sequence[float | int]],
        delta_time: float | int,
        extra_xml: dict[str, str] | None = None,
        mujoco_assets: dict[str, bytes] | None = None,
    ) -> None:
        """The initialization method.

        :param name: The name of the Airplane. Used as the MuJoCo body name.
        :param weight: The weight of the Airplane in Newtons.
        :param omegas_BP1__E: An array-like of 3 numbers (int or float) representing the
            initial angular velocity of the first Airplane's body axes (in the first
            Airplane's body axes, observed from the Earth frame). Can be a tuple, list,
            or ndarray. Values are converted to floats internally. The units are in
            degrees per second.
        :param g_E: An array-like of 3 numbers (int or float) representing the
            gravitational acceleration vector (in Earth axes). Can be a tuple, list, or
            ndarray. Values are converted to floats internally. The units are in meters
            per second squared.
        :param T_pas_BP1_CgP1_to_E_CgP1: A (4,4) array-like of numbers (int or float)
            representing the passive transformation matrix from the first Airplane's
            body axes, relative to the first Airplane's CG, to Earth axes, relative to
            the first Airplane's CG. Can be a nested tuple, list, or ndarray. Values are
            converted to floats internally.
        :param vCg_E__E: An array-like of 3 numbers (int or float) representing the
            initial velocity of the first Airplane's CG (in Earth axes, observed from
            the Earth frame). Can be a tuple, list, or ndarray. Values are converted to
            floats internally. The units are in meters per second.
        :param I_BP1_CgP1: A (3,3) array-like of numbers (int or float) representing the
            inertia matrix of the first Airplane. It is in the first Airplane's body
            axes, relative to the first Airplane's CG. Can be a nested tuple, list, or
            ndarray. Values are converted to floats internally. The units are in
            kilogram meters squared.
        :param delta_time: The time, in seconds, between each time step. It must be a
            positive number (int or float).
        :param extra_xml: A dict mapping injection point names to XML fragment strings
            to inject into the generated MuJoCo XML. Supported keys are "default",
            "asset", and "visual" (inserted as top level elements), "worldbody"
            (inserted inside the worldbody element, before the body), and "body"
            (inserted inside the body element, after the inertial element). The default
            is None, which injects no extra XML.
        :param mujoco_assets: A dict mapping virtual filenames to their binary contents.
            These are passed to MuJoCo's from_xml_string as the assets parameter,
            allowing meshes and other binary files to be loaded without writing to disk.
            The default is None, which provides no extra assets.
        :return: None
        """
        omegas_BP1__E = _parameter_validation.threeD_number_vectorLike_return_float(
            omegas_BP1__E, "omegas_BP1__E"
        )
        g_E = _parameter_validation.threeD_number_vectorLike_return_float(g_E, "g_E")
        T_pas_BP1_CgP1_to_E_CgP1 = (
            _parameter_validation.fourByFour_number_arrayLike_return_float(
                T_pas_BP1_CgP1_to_E_CgP1, "T_pas_BP1_CgP1_to_E_CgP1"
            )
        )
        vCg_E__E = _parameter_validation.threeD_number_vectorLike_return_float(
            vCg_E__E, "vCg_E__E"
        )
        I_BP1_CgP1 = _parameter_validation.m_by_n_number_arrayLike_return_float(
            I_BP1_CgP1, "I_BP1_CgP1", 3, 3
        )

        start_key_frame_name: str = "start"

        omegasRad_BP1__E = np.deg2rad(omegas_BP1__E)

        mass = weight / np.linalg.norm(g_E)
        omegaXRad_BP1__E, omegaYRad_BP1__E, omegaZRad_BP1__E = omegasRad_BP1__E[:]

        R_pas_BP1_to_E = T_pas_BP1_CgP1_to_E_CgP1[:3, :3]

        R_act_BP1_to_E = np.linalg.inv(R_pas_BP1_to_E)

        R_act_E_to_BP1 = R_act_BP1_to_E.T

        # REFACTOR: Add section on quaternions to ANGLES_VECTORS_AND_TRANSFORMATIONS.md.
        quat_act_E_to_BP1_wxyz = _transformations.R_to_quat_wxyz(R_act_E_to_BP1)

        IXX_BP1_CgP1, IXY_BP1_CgP1, IXZ_BP1_CgP1 = I_BP1_CgP1[0]
        IYX_BP1_CgP1, IYY_BP1_CgP1, IYZ_BP1_CgP1 = I_BP1_CgP1[1]
        IZX_BP1_CgP1, IZY_BP1_CgP1, IZZ_BP1_CgP1 = I_BP1_CgP1[2]

        IXY_BP1_CgP1 = (IXY_BP1_CgP1 + IYX_BP1_CgP1) / 2
        IXZ_BP1_CgP1 = (IXZ_BP1_CgP1 + IZX_BP1_CgP1) / 2
        IYZ_BP1_CgP1 = (IYZ_BP1_CgP1 + IZY_BP1_CgP1) / 2

        (
            quatW_act_E_to_BP1,
            quatX_act_E_to_BP1,
            quatY_act_E_to_BP1,
            quatZ_act_E_to_BP1,
        ) = quat_act_E_to_BP1_wxyz[:]

        vCgX_E__E, vCgY_E__E, vCgZ_E__E = vCg_E__E[:]

        # Gravity in the MuJoCo model is turned off as it is applied by the
        # FreeFlightSolver.
        gravity_str = "0.0 0.0 0.0"
        inertia_str = (
            f"{IXX_BP1_CgP1} {IYY_BP1_CgP1} {IZZ_BP1_CgP1} {IXY_BP1_CgP1} "
            f"{IXZ_BP1_CgP1} {IYZ_BP1_CgP1}"
        )
        qpos_str = (
            f"0.0 0.0 0.0 {quatW_act_E_to_BP1} {quatX_act_E_to_BP1} {quatY_act_E_to_BP1} "
            f"{quatZ_act_E_to_BP1}"
        )
        qvel_str = (
            f"{vCgX_E__E} {vCgY_E__E} {vCgZ_E__E} {omegaXRad_BP1__E} "
            f"{omegaYRad_BP1__E} {omegaZRad_BP1__E}"
        )

        # Build the extra XML fragments to inject. If extra_xml is None, use an empty
        # dict so the .get calls below return empty strings.
        _extra = extra_xml if extra_xml is not None else {}
        extra_default = _extra.get("default", "")
        extra_asset = _extra.get("asset", "")
        extra_visual = _extra.get("visual", "")
        extra_worldbody = _extra.get("worldbody", "")
        extra_body = _extra.get("body", "")

        # Initialize the immutable attributes.
        self._xml_str: str = f"""
        <mujoco model="{name}">
          <option timestep="{delta_time}" integrator="RK4" gravity="{gravity_str}"/>

          {extra_default}
          {extra_asset}
          {extra_visual}

          <worldbody>
            {extra_worldbody}
            <body name="{name}" pos="0.0 0.0 0.0" >
              <freejoint/>
              <inertial pos="0.0 0.0 0.0" mass="{mass}" fullinertia="{inertia_str}"/>
              {extra_body}
            </body>
          </worldbody>

          <keyframe>
            <key name="{start_key_frame_name}" qpos="{qpos_str}" qvel="{qvel_str}"/>
          </keyframe>
        </mujoco>
        """

        # Create the internal MuJoCo model object from the XML str. If mujoco_assets
        # is provided, pass it so MuJoCo can resolve virtual filenames (e.g., STL
        # meshes) without writing them to disk.
        # noinspection PyArgumentList
        if mujoco_assets is not None:
            self._model: mujoco.MjModel = mujoco.MjModel.from_xml_string(
                self._xml_str, assets=mujoco_assets
            )
        else:
            # noinspection PyArgumentList
            self._model = mujoco.MjModel.from_xml_string(self._xml_str)

        # Set the internal model's time step to be the same as the simulation's.
        self._model.opt.timestep = delta_time

        # Initialize the mutable attributes.
        self._data: mujoco.MjData = mujoco.MjData(self._model)

        # Get and store the body ID and the initial conditions key frame ID.
        self._body_id: int = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_BODY, name
        )
        self._initial_key_frame_id: int = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_KEY, start_key_frame_name
        )

        # Set the internal model's state to the initial conditions.
        mujoco.mj_resetDataKeyframe(self._model, self._data, self._initial_key_frame_id)

        # Run forward kinematics to compute derived quantities (xmat, xpos, etc.)
        # from the initial qpos/qvel. Without this, xmat would be zeros until the
        # first call to mj_step.
        mujoco.mj_forward(self._model, self._data)

        # Store initial state for reset functionality.
        self._initial_qpos: np.ndarray = np.copy(self._data.qpos)
        self._initial_qpos.flags.writeable = False
        self._initial_qvel: np.ndarray = np.copy(self._data.qvel)
        self._initial_qvel.flags.writeable = False

    # --- Immutable: read only properties ---
    @property
    def xml_str(self) -> str:
        return self._xml_str

    @property
    def model(self) -> mujoco.MjModel:
        return self._model

    @property
    def body_id(self) -> int:
        return self._body_id

    @property
    def initial_key_frame_id(self) -> int:
        return self._initial_key_frame_id

    @property
    def initial_qpos(self) -> np.ndarray:
        return self._initial_qpos

    @property
    def initial_qvel(self) -> np.ndarray:
        return self._initial_qvel

    # --- Mutable: read only properties ---
    @property
    def data(self) -> mujoco.MjData:
        return self._data

    # --- Other methods ---
    def apply_loads(
        self,
        forces_E: np.ndarray | Sequence[float | int],
        moments_E_CgP1: np.ndarray | Sequence[float | int],
    ) -> None:
        """Applies loads to the model.

        **Notes:**

        xfrc_applied[0:3] = forces_E: The current force applied to the first Airplane's
        CG (in Earth axes) in Newtons.

        xfrc_applied[3:6] = moments_E_CgP1: The current moment applied to the first
        Airplane's CG (in Earth axes, relative to the first Airplane's CG) in Newton
        meters.

        The loads will persist until the next call to apply_loads or until they are
        explicitly cleared.

        :param forces_E: An array-like of 3 numbers (int or float) representing the
            forces (in Earth axes) to apply to the first Airplane at the first
            Airplane's CG. Can be a tuple, list, or ndarray. Values are converted to
            floats internally. The units are in Newtons.
        :param moments_E_CgP1: An array-like of 3 numbers (int or float) representing
            the moments (in Earth axes, relative to the first Airplane's CG) to apply to
            the first Airplane at the first Airplane's CG. Can be a tuple, list, or
            ndarray. Values are converted to floats internally. The units are in Newton
            meters.
        :return: None
        """
        forces_E = _parameter_validation.threeD_number_vectorLike_return_float(
            forces_E, "forces_E"
        )
        moments_E_CgP1 = _parameter_validation.threeD_number_vectorLike_return_float(
            moments_E_CgP1, "moments_E_CgP1"
        )

        # Pack the force and moment into the model's 6-element xfrc_applied array.
        self._data.xfrc_applied[self._body_id][:] = np.hstack(
            [forces_E, moments_E_CgP1]
        )

    def step(self) -> None:
        """Advances the MuJoCo simulation by one time step.

        Steps the equations of motion forward in time by one time step, taking into
        account all forces, moments, contacts, and constraints in the model.

        :return: None
        """
        mujoco.mj_step(self._model, self._data)

    def get_state(self) -> dict[str, np.ndarray | float]:
        """Extracts the current position, orientation, velocity, and angular velocity of
        the model.

        **Notes:**

        qpos[0:3] = position_E_E: The current position of the first Airplane's CG (in
        Earth axes, relative to the Earth origin) in meters.

        qvel[0:3] = velocity_E__E: The current velocity of the first Airplane's CG (in
        Earth axes, observed from the Earth frame) in meters per second.

        np.rad2deg(qvel[3:6]) = omegas_BP1__E: The current angular velocity of the first
        Airplane's body axes (in the first Airplane's body axes, observed from the Earth
        frame) in degrees per second.

        xmat = R_pas_BP1_to_E: The current orientation of the first Airplane as a
        passive rotation matrix from the first Airplane's body axes to Earth axes.

        We define MuJoCo world coordinates to be identical to Ptera Software Earth axes.

        :return: A dictionary containing the following keys: ``position_E_E``, a (3,)
            ndarray of floats representing the current position of the first Airplane's
            CG (in Earth axes, relative to the Earth origin) in meters;
            ``R_pas_E_to_BP1``, a (3,3) ndarray of floats representing the current
            orientation of the first Airplane as a passive rotation matrix from Earth
            axes to first Airplane's body axes; ``velocity_E__E``, a (3,) ndarray of
            floats representing the current velocity of the first Airplane's CG (in
            Earth axes, observed from the Earth frame) in meters per second;
            ``omegas_BP1__E``, a (3,) ndarray of floats representing the current angular
            velocity of the first Airplane's body axes (in the first Airplane's body
            axes, observed from the Earth frame) in degrees per second; ``time``, a
            float representing the current simulation time in seconds.
        """
        # MuJoCo's xmat is R_pas_BP1_to_E: it transforms vectors from the first
        # Airplane's body axes to Earth axes. To get R_pas_E_to_BP1, we take the
        # transpose.
        R_pas_BP1_to_E = self._data.xmat[self._body_id].reshape(3, 3)
        # REFACTOR: Consider creating an invert_R_pas function in _transformations.py
        #  and calling it here.
        R_pas_E_to_BP1 = R_pas_BP1_to_E.T

        return {
            "position_E_E": np.copy(self._data.qpos[0:3]),
            "R_pas_E_to_BP1": np.copy(R_pas_E_to_BP1),
            "velocity_E__E": np.copy(self._data.qvel[0:3]),
            "omegas_BP1__E": np.rad2deg(np.copy(self._data.qvel[3:6])),
            "time": float(self._data.time),
        }

    def reset(self) -> None:
        """Resets the model's state to the initial conditions, time to zero seconds, and
        removes any applied loads.

        :return: None
        """
        # Reset the model's state to the initial conditions.
        self._data.qpos[:] = self._initial_qpos
        self._data.qvel[:] = self._initial_qvel

        # Reset time to zero seconds.
        self._data.time = 0.0

        # Remove any applied loads.
        self._data.xfrc_applied[:] = 0.0

        # Run forward kinematics to update dependent quantities.
        mujoco.mj_forward(self._model, self._data)
